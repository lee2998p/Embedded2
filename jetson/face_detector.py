import os
import argparse
import time

import numpy as np
import tensorflow as tf
import cv2
import torch
from PIL import Image
from data import BaseTransform
from torch.autograd import Variable
from torchvision import transforms
import statistics
import matplotlib.pyplot as plt
from layers import Detect
from data import voc, coco, wider_face
import math
from layers import nms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from prior_box_blazeface import PriorBox

from ssd import build_ssd
import warnings


class FaceDetector:
    def __init__(self, trained_model, detection_threshold=0.75, cuda=True, set_default_dev=False):
        """
       Creates a FaceDetector object
       @param trained_model: A string path to a trained pth file for a ssd model trained in face detection
       @param detection_threshold: The minimum threshold for a detection to be considered valid
       @param cuda: Whether or not to enable CUDA
       @param set_default_dev: Whether or not to set the default device for PyTorch
       """
        print(tf.__version__)
        # initialize appropriate model, ssd or blazeface
        if ('.pth' in trained_model):
            self.net = build_ssd('test', 300, 2)
            self.interpreter = None

        elif('.tflite' in trained_model):
            self.net = None
            self.interpreter = tf.lite.Interpreter(model_path=trained_model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.transformer = BaseTransform(self.input_details[0]['shape'][1], (0, 0, 0))

            # get anchors/priors
            file = open('anchors_from_mediapipe.txt', 'r')
            anchors = file.readlines()
            anchors = [(x.strip().split(' ')) for x in anchors]
            anchors = [[float(y) for y in x] for x in anchors]
            file.close()

            anchors = torch.Tensor(anchors)
            anchors = anchors.unsqueeze(0)

            self.priors_mediapipe = anchors


        # set device to cuda if available
        self.cfg = (wider_face)
        print(self.cfg)
        num_classes = 2
        self.device = torch.device("cpu")
        self.detect_boxes = Detect(num_classes, 0, 200, 0.01, 0.45)

        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif set_default_dev:
            torch.set_default_tensor_type('torch.FloatTensor')

        # move ssd to device if using ssd
        if ('.pth' in trained_model):
            print(f'Moving network to {self.device.type}')
            self.net.to(self.device)
            self.net.load_state_dict(torch.load(trained_model, map_location=self.device))
            self.net.eval()
            self.transformer = BaseTransform(self.net.size, (104, 117, 123))


    def detect(self, image, threshold=0.2):

        # ssd
        if self.net:
            x = torch.from_numpy(self.transformer(image)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0)).to(self.device)

            start = time.clock()
            y = self.net(x)
            end = time.clock()

            print('time ', end-start)

            # print('y.shape', y.shape)
            # print('y type', type(y))
            detections = y.data
            # print('detection.shape', detections.shape)
            # print('detection type', type(detections))
            scale = torch.Tensor([image.shape[1], image.shape[0],
                                  image.shape[1], image.shape[0]])
            bboxes = []
            j = 0
            while j < detections.shape[2] and detections[0, 1, j, 0] > threshold:
                pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
                x1, y1, x2, y2 = pt
                bboxes.append((x1, y1, x2, y2))
                j += 1
            print(bboxes)
            return bboxes

        # blazeface
        if self.interpreter:
            x = torch.from_numpy(self.transformer(image)[0])
            x = Variable(x.unsqueeze(0)).to(self.device)
            # print('input details ', self.input_details)
            input_shape = self.input_details[0]['shape']
            input_data = np.array(x, dtype=np.float32)
            
            # print('intended input shape', input_shape)
            # print('actual input shape', input_data.shape)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            start = time.clock()
            self.interpreter.invoke()
            end = time.clock()

            print('time, ' , end-start)
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            score_data = self.interpreter.get_tensor(self.output_details[1]['index'])

            output_shape = self.output_details[0]['shape']
            # print(self.output_details)
            # print('intended output shape', output_shape)
            # print('actual output shape', output_data.shape)

            boxes_decoded = magically_decode_boxes(self.priors_mediapipe, output_data)

            score_data = keep_top_scores(score_data)
            # print('score data.shape', score_data.shape)
            # minval = 100
            # maxval = -100
            # score_data_reshaped = score_data.squeeze()
            # for each in score_data_reshaped.tolist():
            #     if each < minval:
            #         minval = each
            #     if each > maxval:
            #         maxval = each
            #
            # print('minval ', minval)
            # print('maxval ', maxval)
            # print('boxes decoded shape ', boxes_decoded.shape)
            # print('score data shape ', score_data.shape)
            detections = convertToDetections(boxes_decoded, score_data, threshold = 0.2, flip_vertically=False)
            output = torch.zeros(1, 2, 200, 5)
            # print('score data type', type(score_data))
            # print('score data shape', score_data.shape)
            score_data = torch.Tensor(score_data)
            score_data = score_data.squeeze()

            ids, count = nms(detections, score_data)
            output = torch.cat((score_data[ids[:count]].unsqueeze(1),
                           detections[ids[:count]]), 1)
            # print('output shape before contiguous', output.shape)
            # flt = output.contiguous().view(1, -1, 5)
            # print('flt.shape ', flt.shape)
            # print('flt ', flt)
            # _, idx = flt[:, :, 0].sort(1, descending=True)
            # _, rank = idx.sort(1)
            # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
            detections = output

            scale = torch.Tensor([image.shape[1], image.shape[0],
                                  image.shape[1], image.shape[0]])
            # SOME jank code to display whatever bounding boxes we can at this point without properly ranking after nms
            bboxes = []
            j = 0
            # print(detections)
            while j < detections.shape[1] and detections[j, 0] > 0.5:
                print('CONFIDENCE:  ', detections[j,0])
                pt = (detections[j, 1:5] * scale).cpu().numpy()
                print(pt)

                y1, x1, y2, x2 = pt
                bboxes.append((x1, y1, x2, y2))
                j += 1
            # print(bboxes)
            return bboxes

        #@todo: detection of output data
            # print(detections[0:5])

def convertToDetections(boxes, scores, threshold, flip_vertically):
    detections = []
   # print(boxes.shape[0])
    for i in range(boxes.shape[0]):
       #if scores[i].item() > threshold:
            ymin = boxes[i][0].item()
            xmin = boxes[i][1].item()
            ymax = boxes[i][2].item()
            xmax = boxes[i][3].item()

            detection = []
            detection.append(xmin)
            if (flip_vertically):
                detection.append(1-ymax)
            else:
                detection.append(ymin)
            detection.append(xmax)
            detection.append(ymax)

            # add keypoints
            for j in range(4,16):
                detection.append(boxes[i][j].item())

            detections.append(detection)

    # print((detections))
    detections = torch.Tensor(detections)
    #print('shape of detections ', detections.shape)
    return detections

def keep_top_scores(scores):
    """
    refer to: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    apply sigmoid to raw scores (which can be negative)
    filter by scores and keep top k
    :param scores: (1,896,1) tensor
    :return: tensor of top k scores
    """
    # print(scores)
    scores = scores.squeeze(0)
    for i in range(scores.shape[0]):
        if scores[i].item() < -100:
            scores[i] = -100
        elif scores[i].item() > 100:
            scores[i] = 100

        scores[i] = 1/(1+math.exp(-scores[i]))


    return scores
    # print(scores)


def magically_decode_boxes(anchors, raw_boxes):
    """
    decodes raw boxes based on anchors
    refer to: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    function DecodeBoxes
    :return: displayable boxes
    """
    raw_boxes = raw_boxes.squeeze(0)
    anchors = anchors.squeeze(0)

    # output boxes
    boxes = []

    # print(anchors.shape)
    # print(raw_boxes.shape)
    for i in range(anchors.shape[0]):
        x_center = raw_boxes[i][0]
        y_center = raw_boxes[i][1]
        w = raw_boxes[i][2]
        h = raw_boxes[i][3]

        # anchor could be y,x,h,w or x,y,w,h
        x_center = x_center / 128 * anchors[i][2] + anchors[i][0];
        y_center = y_center / 128 * anchors[i][3] + anchors[i][1];

        apply_exponential = False
        if apply_exponential:
            h = math.exp(h / 128) * anchors[i][2]
            w = math.exp(w / 128) * anchors[i][3]
        else:
            h = h / 128 * anchors[i][2]
            w = w / 128 * anchors[i][3]


        ymin = y_center - h / 2.0
        xmin = x_center - w / 2.0
        ymax = y_center + h / 2.0
        xmax = x_center + w / 2.0

        box = []

        box.append(ymin.item())
        box.append(xmin.item())
        box.append(ymax.item())
        box.append(xmax.item())

        #keypoint
        keypoints = raw_boxes[i][4:16]
        for k in range(len(keypoints)):
            # print(keypoints[k])
            if k%2 == 0:
                keypoint_x = keypoints[k] / 128 * anchors[i][2] + anchors[i][0]
                box.append(keypoint_x)
            else:
                keypoint_y = keypoints[k] / 128 * anchors[i][2] + anchors[i][1]
                box.append(keypoint_y)

        boxes.append(box)

    boxes = torch.Tensor(boxes)
    # print('boxes.shape', boxes.shape)
    # print(boxes)
    # minval = 100
    # maxval = -100
    # boxes_reshaped = boxes.view(1, 14336)
    # print('boxes reshaped shape ', boxes_reshaped.shape)
    # boxes_reshaped = boxes_reshaped.squeeze()
    # for each in boxes_reshaped.tolist():
    #     if each < minval:
    #         minval = each
    #     if each > maxval:
    #         maxval = each
    #
    # print('minval ', minval)
    # print('maxval ', maxval)
    return boxes

def classify(face, classifier):
    if 0 in face.shape:
        pass
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(rgb_face)
    plt.imshow(pil_face)
    #plt.show()

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

        # using old; fix error TODO
        _, pred = torch.max(labels, 1)

    return pred, softlabels


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', type=str, help="Path to a trained classifier .pth file")
    parser.add_argument('--cropped', default=False, action='store_true',
                        help="Crop out half the face? Make sure your model is trained on cropped images")
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
                # face = img[x1:x2, y1:y2]
                face = img[y1:y2, x1:x2, :]
               # print('face ', face.shape)
                if args.cropped:
                    height = face.shape[0]
                    face = face[round(0.15 * height):round(0.6 * height), :, :]
                    img = cv2.rectangle(img, (x1, y1 + round(0.15 * height)), (x2, y2 - round(0.4 * height)),
                                        (0, 255, 0), 2)

                label, softlabels = classify(face, g)

                glasses_probs.append(softlabels[0][0].item())
                goggle_probs.append(softlabels[0][1].item())
                neither_probs.append(softlabels[0][2].item())
                # pred = max(labels)

                print('num data points', len(goggle_probs))
                if len(goggle_probs) == 50:
                    print('Goggle avg pred: {}'.format(sum(goggle_probs) / len(goggle_probs)))
                    print('Glasses avg pred: {}'.format(sum(glasses_probs) / len(glasses_probs)))
                    print('Neither avg pred: {}'.format(sum(neither_probs) / len(neither_probs)))

                    print('Goggle std. dev: {}'.format(statistics.stdev(goggle_probs)))
                    print('Glasses std. dev: {}'.format(statistics.stdev(glasses_probs)))
                    print('Neither std. dev: {}'.format(statistics.stdev(neither_probs)))

                    #Ease in copy pasting to the sheet
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
