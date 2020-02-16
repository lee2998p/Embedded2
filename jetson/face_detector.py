import os
import argparse
import time

import cv2
import torch
from data import BaseTransform
from torch.autograd import Variable

from ssd import build_ssd


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

    def detect(self, image, threshold=0.35):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained .pth file")
    parser.add_argument('--cuda', '-c', type=bool, help="Enable cuda", default=True)
    args = parser.parse_args()
    detector = FaceDetector(trained_model=args.trained_model, cuda=args.cuda, set_default_dev=True)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            start_time = time.time()
            _, img = cap.read()
            boxes = detector.detect(img)
            for box in boxes:
                x1, y1, x2, y2 = box
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            fps = 1 / (time.time() - start_time)
            img = cv2.putText(img, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imshow("Face Detect", img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
