import os
import argparse
import time

import cv2
import torch
from PIL import Image
from data import BaseTransform
from torch.autograd import Variable
from torchvision import transforms

from ssd import build_ssd
import warnings

class FaceDetector:
    def __init__(self, trained_model, cuda=True, set_default_dev=False):
        self.net = build_ssd('test', 300, 2)
        self.device = torch.device("cpu")
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif set_default_dev:
            torch.set_default_tensor_type('torch.FloatTensor')

        print(f'Moving network to {self.device.type}')
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(trained_model, map_location=self.device))
        self.net.eval()
        self.transformer = BaseTransform(self.net.size, (104, 117, 123))

    def detect(self, image, threshold=0.50):
        x = torch.from_numpy(self.transformer(image)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0)).to(self.device)
        y = self.net(x)
        detections = y.data
        scale = torch.Tensor([image.shape[1], image.shape[0],
                              image.shape[1], image.shape[0]])
        bboxes = []
        j = 0
        while j < detections.shape[2] and detections[0, 1, j, 0] > threshold:
            pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
            x1, y1, x2, y2 = pt
            bboxes.append((x1, y1, x2, y2))
            j += 1
        return bboxes


def classify(face):
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(rgb_face)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_face = transform(pil_face)
    face_batch = transformed_face.unsqueeze(0)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        face_batch = face_batch.to(device)
        labels = g(face_batch)
        _, pred = torch.max(labels, 1)

    return pred


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained .pth file")
    parser.add_argument('--cuda', '-c', type=bool, help="Enable cuda", default=True)
    args = parser.parse_args()
    detector = FaceDetector(trained_model=args.trained_model, cuda=args.cuda, set_default_dev=True)
    cap = cv2.VideoCapture(0)

    g = torch.load("shufflenet.pth")
    g.eval()
    class_names = ['both', 'glasses', 'goggles', 'neither', 'N/A']

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

                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                face = img[x1:x2, y1:y2]
                label = classify(face)
                img = cv2.putText(img, 'label: %s' % class_names[label], (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            fps = 1 / (time.time() - start_time)
            img = cv2.putText(img, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imshow("Face Detect", img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
