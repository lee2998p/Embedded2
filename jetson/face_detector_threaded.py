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

from BlazeFace_2.blazeface import BlazeFace
from data import BaseTransform
from ssd import build_ssd

import sys
sys.path.append("../Encryption")
from AES import encryption

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
            self.net = BlazeFace()
            self.net.load_weights("blazeface.pth")
            self.net.load_anchors("BlazeFace_2/anchors.npy")
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


def classify(face, classifier):
    if 0 in face.shape:
        pass
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(rgb_face)

    # see what the classifier sees
    #plt.imshow(pil_face)
    #plt.show()

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


def encrypt(coordinates, encryptedImg): 
    #Accepts list of face coordinates and img,
    #Calls AES to encrypt region
    #For now, returns img with encrypted face(s)

    encryptor = encryption()
    encryptedImg, _ = encryptor.encrypt(coordinates, encryptedImg)

    #TODO ftp encryptedImg to remote

    return encryptedImg

if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', type=str, help="Path to a trained classifier .pth file")
    parser.add_argument('--cropped', default=False, action='store_true',
                        help="Crop out half the face? Make sure your model is trained on cropped images")
    parser.add_argument('--encrypt_flag', default=False)
    args = parser.parse_args()
    detector = FaceDetector(trained_model=args.trained_model, cuda=args.cuda and torch.cuda.is_available(),
                            set_default_dev=True)
    cap = cv2.VideoCapture(0)

    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    g = torch.load(args.classifier, map_location=device)
    g.eval()
    class_names = ['Glasses', 'Goggles', 'Neither']

    goggle_probs = []
    glasses_probs = []
    neither_probs = []
    preds = []

    if cap.isOpened():
        while True:
            start_time = time.time()
            _, img = cap.read()
            boxes = detector.detect(img)
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                # draw boxes within the frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                encryptedImg = img.copy() #copy img for separate thread
                encryptedImg = encrypt([(x1, y1, x2, y2)], encryptedImg)

                if args.encrypt_flag:
                    img = encryptedImg #encrypt stream

                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                face = img[y1:y2, x1:x2, :]

                if args.cropped:
                    height = face.shape[0]
                    face = face[round(0.15 * height):round(0.6 * height), :, :]
                    img = cv2.rectangle(img, (x1, y1 + round(0.15 * height)), (x2, y2 - round(0.4 * height)),
                                        (0, 255, 0), 2)

                label, softlabels = classify(face, g)

                glasses_probs.append(softlabels[0][0].item())
                goggle_probs.append(softlabels[0][1].item())
                neither_probs.append(softlabels[0][2].item())
                preds.append(label.item())

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

                img = cv2.putText(img, 'label: %s' % class_names[label], (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 0, 0))
            fps = 1 / (time.time() - start_time)
            img = cv2.putText(img, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imshow("Face Detect", img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        exit(0)
    else:
        print("Unable to open camera")
