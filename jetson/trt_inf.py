"""
    #TODO document fix
    Conversions broken: torch.Tensor.pow, torch.Tensor.unsqueeze, torch.Tensor.expand_as
    Conversions fixed: torch.Tensor.pow
"""

import argparse
import time

import cv2
import torch
from torch.autograd import Variable
from torch2trt import torch2trt

from ssd import build_ssd
from data import BaseTransform
from face_detector import FaceDetector
import warnings

warnings.filterwarnings("once")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using tensorRT")
    parser.add_argument('--trained_model', '-t', type=str, required=True,
                        help='Path of .pth file holding trained model')
    args = parser.parse_args()
    detector = FaceDetector(trained_model=args.trained_model, cuda=True, set_default_dev=True)
    size = detector.net.size
    dummy_data = torch.zeros((1, 3, size, size))
    print("Converting network")
    detector.net = torch2trt(detector.net, [dummy_data])
    print("Conversion successful, starting inference")
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
