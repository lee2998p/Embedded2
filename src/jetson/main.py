import argparse
import statistics
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from models.BlazeFace.blazeface import BlazeFace
from utils.transform import BaseTransform
from models.SSD.ssd import build_ssd

import sys, os, inspect

from AES import encryption as Encryptor

from threading import Thread
import multiprocessing
from multiprocessing import Process

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
            self.transformer = BaseTransform(128, (104, 117, 123))

        self.detection_threshold = detection_threshold
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif set_default_dev:
            torch.set_default_tensor_type('torch.FloatTensor')

        print(f'Moving network to {self.device.type}')
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
            img = cv2.resize(image, (128, 128)).astype(np.float32)

            detections = self.net.predict_on_image(img)
            if isinstance(detections, torch.Tensor):
                detections = detections.cpu().numpy()

            if detections.ndim == 1:
                detections = np.expand_dims(detections, axis=0)

            print("Found %d faces" % detections.shape[0])

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
        self.goggle_probs = []
        self.glasses_probs = []
        self.neither_probs = []
        self.preds = []
        self.fps = 0
        self.classifier = classifier

        global fileCount
        fileCount = 0

    def classifyFace(self, face):
        classifier = self.classifier

        if 0 in face.shape:
            pass
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

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
            print('Probability labels: {}'.format(softlabels))
            _, pred = torch.max(labels, 1)

        return pred, softlabels

    def classifyFrame(self, img, boxes):
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

            if args.cropped:
                height = face.shape[0]
                face = face[round(0.15 * height):round(0.6 * height), :, :]
                img = cv2.rectangle(img,
                        (x1, y1 + round(0.15 * height)),
                        (x2, y2 - round(0.4 * height)),
                        (0, 255, 0), 2)

            label, softlabels = self.classifyFace(face)

            self.glasses_probs.append(softlabels[0][0].item())
            self.goggle_probs.append(softlabels[0][1].item())
            self.neither_probs.append(softlabels[0][2].item())
            self.preds.append(label.item())

        return label, softlabels

    def printProbs(self):
        goggle_probs = self.goggle_probs
        glasses_probs = self.glasses_probs
        neither_probs = self.neither_probs
        print('num data points', len(goggle_probs))
        if len(goggle_probs) == 50:
            print('Goggle avg pred: {}'.format(sum(goggle_probs) / len(goggle_probs)))
            print('Glasses avg pred: {}'.format(sum(glasses_probs) / len(glasses_probs)))
            print('Neither avg pred: {}'.format(sum(neither_probs) / len(neither_probs)))

            print('Goggle std. dev: {}'.format(statistics.stdev(goggle_probs)))
            print('Glasses std. dev: {}'.format(statistics.stdev(glasses_probs)))
            print('Neither std. dev: {}'.format(statistics.stdev(neither_probs)))

            print('Goggle predictions: {}'.format(preds.count(1)))
            print('Glasses predictions: {}'.format(preds.count(0)))
            print('Neither predictions: {}'.format(preds.count(2)))

            # Ease in copy pasting to the sheet
            print ('\nPaste the following numbers on the sheet: \n')
            print(sum(goggle_probs) / len(goggle_probs))
            print(sum(glasses_probs) / len(glasses_probs))
            print(sum(neither_probs) / len(neither_probs))
            print(statistics.stdev(goggle_probs))
            print(statistics.stdev(glasses_probs))
            print(statistics.stdev(neither_probs))
            print('\n')


def encryptFace(coordinates, img):
    #Accepts list of face coordinates and img,
    #Calls AES to encrypt region
    #For now, returns img with encrypted face(s)

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
        #Lets just write img for now
        global fileCount
        #if fileCount <= 50:
        face_file_name = os.path.join(args.output_dir, f'{fileCount}.jpg')

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
    parser.add_argument('--cropped', default=False, action='store_true',
                        help="Crop out half the face? Make sure your model is trained on cropped images")
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
        cl.printProbs()

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
