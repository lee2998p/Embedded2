import argparse
import time
import warnings

import cv2
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from models.BlazeFace.blazeface import BlazeFace
from utils.transform import BaseTransform
from models.SSD.ssd import build_ssd

import sys, os, inspect

from AES import encryption as Encryptor

from threading import Thread

fileCount = None

class FaceDetector:
    def __init__(self, trained_model, detection_threshold=0.75, cuda=True, set_default_dev=False):
        """
        Creates a FaceDetector object
        @param trained_model: A string path to a trained pth file for a ssd model trained in face detection
        @param detection_threshold: The minimum threshold for a detection to be considered valid
        @param cuda: Whether or not to enable CUDA
        @param set_default_dev: Whether or not to set the default device for PyTorch
        """

        self.device = torch.device("cpu")

        if ('.pth' in trained_model and 'ssd' in trained_model):
            self.net = build_ssd('test', 300, 2)
            self.model_name = 'ssd'
            self.net.load_state_dict(torch.load(trained_model, map_location=self.device))
            self.transformer = BaseTransform(self.net.size, (104, 117, 123))


        elif ('.pth' in trained_model and 'blazeface' in trained_model):
            self.net = BlazeFace(self.device)
            self.net.load_weights(trained_model)
            self.net.load_anchors("models/BlazeFace/anchors.npy")
            self.model_name = 'blazeface'
            self.net.min_score_thresh = 0.75
            self.net.min_suppression_threshold = 0.3
            self.transformer = BaseTransform(128, None)

        self.detection_threshold = detection_threshold
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif set_default_dev:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.net.to(self.device)
        self.net.eval()


    def detect(self, image):
        """
        Performs face detection on the image passed
        @param image: A numpy array representing an image
        @return: The bounding boxes of the face(s) that were detected formatted (upper left corner(x, y) , lower right corner(x,y))
        """
        if (self.model_name == 'ssd'):
            x = torch.from_numpy(self.transformer(image)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0)).to(self.device)
            y = self.net(x)
            detections = y.data
            scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            bboxes = []
            j = 0
            while j < detections.shape[2] and detections[0, 1, j, 0] > self.detection_threshold:
                pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
                x1, y1, x2, y2 = pt
                bboxes.append((x1, y1, x2, y2))
                j += 1
            return bboxes

        elif (self.model_name == 'blazeface'):
            img = self.transformer(image)[0].astype(np.float32)
            print (img)
            print (img.shape)


            detections = self.net.predict_on_image(img)
            if isinstance(detections, torch.Tensor):
                detections = detections.cpu().numpy()

            if detections.ndim == 1:
                detections = np.expand_dims(detections, axis=0)

            bboxes = []
            for i in range(detections.shape[0]):
                ymin = detections[i, 0] * image.shape[0]
                xmin = detections[i, 1] * image.shape[1]
                ymax = detections[i, 2] * image.shape[0]
                xmax = detections[i, 3] * image.shape[1]

                img = img / 127.5 - 1.0

                for k in range(6):
                    kp_x = detections[i, 4 + k * 2] * img.shape[1]
                    kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]

                bboxes.append((xmin, ymin, xmax, ymax))
            return bboxes

class VideoCapturer(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        _, self.frame = self.capture.read()
        self.t1 = Thread(target=self.update, args=())
        self.t1.daemon = True
        self.t1.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                _, self.frame = self.capture.read()
            time.sleep(.01)

    def get_frame(self):
        return self.frame


class Classifier:
    def __init__(self, classifier):
        '''
        Performs classification of facial region into three classes - [Goggles, Glasses, Neither]

        Params-
        classifier - Trained classifier model (Currently, mobilenetv2)
        '''

        self.fps = 0
        self.classifier = classifier

        global fileCount
        fileCount = 0

    def classifyFace(self, face):
        '''
        This method initializaes the transforms and classifies the face region

        Params-
        face - Face coordinates

        Returns -
        pred - A tensor containing the index of the highest class probability
        softlabels - A tensor containing all three class probabilities
        '''

        classifier = self.classifier

        if 0 in face.shape:
            pass
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        # Transforms applied to image before passing it to classifier. These should be
        # the same transforms as applied while training model
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomGrayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed_face = transform(pil_face)
        face_batch = transformed_face.unsqueeze(0)
        device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            face_batch = face_batch.to(device)
            labels = classifier(face_batch)
            m = torch.nn.Softmax(1)
            softlabels = m(labels)
            _, pred = torch.max(labels, 1)

        return pred, softlabels

    def classifyFrame(self, img, boxes):
        '''
        This method loops through all the bounding boxes in an image, calls classifyFace method
        to classify face region and finally draws a box around the face.

        Params -
        img - Input video frame
        boxes - Coordinates of the bounding box around the face

        '''

        label = None
        softlabels = None
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = img[y1:y2, x1:x2, :]

            label, softlabels = self.classifyFace(face)


        return label, softlabels


def encryptFace(coordinates, img):
    '''
    This function Encrypts faces

    Params-
    coordinates - Face coordinates returned by face detector
    img - Image to be encrypted

    Returns-
    encryptedImg - Image with face coordinates encrypted
    '''


    encryptor = Encryptor()
    encryptedImg, _ = encryptor.encrypt(coordinates, img)

    return encryptedImg

def encryptFrame(img, boxes):
    try:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            img = encryptFace([(x1, y1, x2, y2)], img)

        #TODO ftp img to remote
        #Lets just write img to filesystem for now
        global fileCount
        face_file_name = os.path.join(args.output_dir, f'{fileCount}.jpg')

        #TODO: Remove this print statement after db integration
        print("writing ", face_file_name)
        fileCount += 1
        cv2.imwrite(face_file_name, img)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', type=str, help="Path to a trained classifier .pth file")
    parser.add_argument('--output_dir', default='encrypted_imgs', type=str, help="Where to output encrypted images")
    args = parser.parse_args()

    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    g = torch.load(args.classifier, map_location=device)
    g.eval()
    class_names = ['Glasses', 'Goggles', 'Neither']

    cap = VideoCapturer() #Instantiate Video Capturer object
    detector = FaceDetector(trained_model=args.trained_model, cuda=args.cuda and torch.cuda.is_available(), set_default_dev=True) #Instantiate Face Detector object
    cl = Classifier(g) #Instantiate Classifier object

    while True:
        start_time = time.time()

        frame = cap.get_frame()
        boxes = detector.detect(frame)

        encryptedImg = frame.copy() #copy for creating encrypted image

        if len(boxes) != 0:
            p1 = Thread(target=encryptFrame, args=(encryptedImg, boxes))
            p1.daemon = True
            p1.start()

        label, softlabels = cl.classifyFrame(frame, boxes)

        fps = 1 / (time.time() - start_time)
        if len(boxes) != 0:
            frame = cv2.putText(frame,
                    'label: %s' % class_names[label],
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0))

        frame = cv2.putText(frame,
                'fps: %.3f' % fps,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0))
        cv2.imshow("Face Detect", frame)

        p1.join()

        if cv2.waitKey(1) == 27:
           break

    cv2.destroyAllWindows()
    exit(0)
