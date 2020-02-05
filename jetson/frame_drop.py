import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import warnings
import time
from torch.autograd import Variable
from ssd import build_ssd
from data import BaseTransform
from multiprocessing import Process, Pipe
import multiprocessing as mp

SIZE_THRESH = 600

warnings.filterwarnings("once")


# class Detector:
#     def __init__(self, network, device, threshold=0.35):
#         self.network = network
#         self.device = device
#         self.threshold = threshold
#
#     def detect_face(self, pipe: Pipe):
#         print("HI")
#         while True:
#             image = pipe.recv()
#             if type(image) is str and image == "DONE":
#                 break
#             print("LOOP")
#             x = torch.from_numpy(transformer(image)[0]).permute(2, 0, 1)
#             x = Variable(x.unsqueeze(0)).to(self.device)
#             y = self.network(x)
#
#             detections = y.data
#
#             scale = torch.Tensor([image.shape[1], image.shape[0],
#                                   image.shape[1], image.shape[0]])
#
#             boxes = []
#             j = 0
#             while detections[0, 1, j, 0] >= self.threshold:
#                 pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
#                 x1, y1, x2, y2 = pt
#                 if x2 - x1 < SIZE_THRESH and y2 - y1 < SIZE_THRESH:
#                     boxes.append((x1, y1, x2, y2))
#                 j += 1
#
#             pipe.send(boxes)
#         pipe.close()
#         print("BYE")
#

def detect_face(image, network, device, threshold=0.35):
    x = torch.from_numpy(transformer(image)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0)).to(device)
    y = network(x)

    detections = y.data

    scale = torch.Tensor([image.shape[1], image.shape[0],
                          image.shape[1], image.shape[0]])

    boxes = []
    j = 0
    while detections[0, 1, j, 0] >= threshold:
        pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
        x1, y1, x2, y2 = pt
        if x2 - x1 < SIZE_THRESH and y2 - y1 < SIZE_THRESH:
            boxes.append((x1, y1, x2, y2))
        j += 1
    return boxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
    parser.add_argument('--trained_model', default='ssd300_WIDER_100455.pth', type=str, help="Trained state_dict file")
    parser.add_argument('--visual_threshold', default=0.6, type=float, help='Final confidence threshold')
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--encrypt', default=False, type=bool, help='Enable/Disable encryption')
    parser.add_argument('--camera_index', default=0, type=int, help='Index of camera')
    parser.add_argument('--drop_rate', default=15, type=int, help='Take 1 out of the drop rate and process')
    parser.add_argument('--window_name', default="Face Detection -- Frame Drop", type=str,
                        help="Name for the display window")
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:0')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')

    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("Camera failed to open")
        sys.exit(1)
    cv2.namedWindow(args.window_name, cv2.WINDOW_AUTOSIZE)

    num_classes = 2
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Model has been loaded')

    transformer = BaseTransform(net.size, (104, 117, 123))

    encrypt_status = 1
    decrypt_status = 1
    verbose = 0
    frame_count = 0

    consumer, worker = Pipe(duplex=True)

    # detector = Detector(net, device)
    # mp.set_start_method('spawn')
    # worker_proc = Process(target=detector.detect_face, args=(worker,))
    # worker_proc.start()

    boxes = []
    wait_update = False
    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        if frame_count % args.drop_rate == 0:
            boxes = detect_face(image, net, device)

        for box in boxes:
            x1, y1, x2, y2 = box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        end = time.time()
        fps = 1 / (end - start)

        image = cv2.putText(image, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow(args.window_name, image)
        frame_count += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
