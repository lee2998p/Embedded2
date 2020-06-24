import argparse
import time
import warnings
from typing import List, Set, Dict, Tuple, Optional

import cv2
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from models.utils.transform import BaseTransform

import sys
import os
import inspect

from AES import Encryption as Encryptor

from threading import Thread
import multiprocessing
from multiprocessing import Process, Queue

fileCount = None
encryptRet = Queue()

class FaceDetector:
    def __init__(self, detector:str, detection_threshold=0.7, cuda=True, set_default_dev=False):
        """
        Creates a FaceDetector object
        Args:
            detector: A string path to a trained pth file for a ssd model trained in face detection
            detection_threshold: The minimum threshold for a detection to be considered valid
            cuda: Whether or not to enable CUDA
            set_default_dev: Whether or not to set the default device for PyTorch
        """

        self.device = torch.device("cpu")

        if ('.pth' in detector and 'ssd' in detector):
            from models.SSD.ssd import build_ssd

            self.net = build_ssd('test', 300, 2)
            self.model_name = 'ssd'
            self.net.load_state_dict(torch.load(detector, map_location=self.device))
            self.transformer = BaseTransform(self.net.size, (104, 117, 123))


        elif ('.pth' in detector and 'blazeface' in detector):
            from models.BlazeFace.blazeface import BlazeFace


            self.net = BlazeFace(self.device)
            self.net.load_weights(detector)
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


    def detect(self,
               image: np.ndarray):
        """
        Performs face detection on the image passed
        Args:
            image: A 3D numpy array representing an image

        Return:
            The bounding boxes of the face(s) that were detected formatted (upper left corner(x, y) , lower right corner(x,y))
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
        '''
        This class captures videos using open-cv's VideoCapture object
        Args:
            src: This is the connection to the source of the video stream (webcam or raspberry pi camera)
        '''

        self.capture = cv2.VideoCapture(src)
        _, self.frame = self.capture.read()
        self.t1 = Thread(target=self.update, args=())
        self.t1.daemon = True
        self.t1.start()

    def update(self):
        '''Get next frame in video stream'''
        while True:
            if self.capture.isOpened():
                _, self.frame = self.capture.read()
            time.sleep(.01)

    def get_frame(self):
        ''' Return current frame in video stream'''
        return self.frame


class Classifier:
    def __init__(self, classifier):
        '''
        Performs classification of facial region into three classes - [Goggles, Glasses, Neither]
        Args:
            classifier - Trained classifier model (Currently, mobilenetv2)
        '''
        self.fps = 0
        self.classifier = classifier

        global fileCount
        fileCount = 0

    def classifyFace(self,
                    face: np.ndarray):
        '''
        This method initializaes the transforms and classifies the face region
        Args:
            face - A 3D numpy array containing facial region

        Return:
            pred - A tensor containing the index of the highest class probability
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
            _, pred = torch.max(labels, 1)

        return pred

    def classifyFrame(self,
                    img: np.ndarray,
                    boxes: List[Tuple[np.float64]]):
        '''
        This method loops through all the bounding boxes in an image, calls classifyFace method
        to classify face region and finally draws a box around the face.
        Args:
            img - A 3d numpy array containing input video frame
            boxes - Coordinates of the bounding box around the face

        Return:
            label: Classification label (Goggles, Glasses or Neither)
        '''

        label = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            # draw boxes within the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = img[y1:y2, x1:x2, :]

            label.append(int(self.classifyFace(face).data))


        return label


def encryptFace(coordinates: List[Tuple[int]],
                img: np.ndarray):
    '''
    This function Encrypts faces
    Args:
        coordinates - Face coordinates returned by face detector
        img - A 3D numpy array containing image to be encrypted

    Return:
        encryptedImg - Image with face coordinates encrypted
    '''

    encryptor = Encryptor()
    encryptedImg, _ = encryptor.encrypt(coordinates, img)

    return encryptedImg

def encryptFrame(img:np.ndarray,
                boxes:List[Tuple[np.float64]]):
    '''
    This method takes the face coordinates, encrypts the facial region, writes encrypted image to file filesystem
    Args:
        img: A 3D numpy array containing image to be encrypted
        boxes: facial Coordinates
    '''
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


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
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
    detector = FaceDetector(detector=args.detector, cuda=args.cuda and torch.cuda.is_available(), set_default_dev=True) #Instantiate Face Detector object
    cl = Classifier(g) #Instantiate Classifier object

    while True:
        start_time = time.time()

        frame = cap.get_frame()
        boxes = detector.detect(frame)

        encryptedImg = frame.copy() #copy for creating encrypted image

        if len(boxes) != 0:
            p1 = Process(target=encryptFrame, args=(encryptedImg, boxes))
            p1.daemon = True
            p1.start()

            label = cl.classifyFrame(frame, boxes)

            fps = 1 / (time.time() - start_time)

            # Remove line 300-312 before deployment
            index = 0
            for box in boxes:
                frame = cv2.putText(frame,
                            'label: %s' % class_names[label[index]],
                            (int(box[0]), int(box[1]-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255))

                frame = cv2.putText(frame,
                        'fps: %.3f' % fps,
                        (int(box[0]), int(box[1]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255))

                index += 1

            cv2.imshow("Face Detect", frame)

            p1.join()

            if cv2.waitKey(1) == 27:
                p1.terminate()
                p1.join()
                break

    # Remove line 319 before deployment
    cv2.destroyAllWindows()
    exit(0)
