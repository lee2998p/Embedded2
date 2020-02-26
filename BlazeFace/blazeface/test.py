from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#import torchvision.transforms as transforms
#from torch.autograd import Variable
from wider_face import WIDER_ROOT, WIDER_CLASSES as labelmap
from PIL import Image
from wider_face import WIDERAnnotationTransform, WIDERDetection, WIDER_CLASSES
from data import BaseTransform
import torch.utils.data as data
from model import *
import time
import cv2

parser = argparse.ArgumentParser(description='Blazeface MultiBox Detection')
parser.add_argument('--trained_model', default='weights/4000.pth',   #Default path needs to be added
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--wider_root', default=WIDER_ROOT, help='Location of WIDER root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def drawBoundingBox(img, result):
    for box in result:
        x1, y1, x2, y2 = (box[0], box[1], box[2], box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 6)


    return img


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test2.txt'
    print(filename)
    num_images = len(testset)
    #path = '/Users/ishaghodgaonkar/Embedded2/BlazeFace/WIDER/WIDER_test/images'
    path = 'WIDER/WIDER_test/images'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    print (files)

    for i in range(num_images):
        # print(i)
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(files[i])
        img_id, annotation = testset.pull_anno(i)

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        print ("x shape:", x.shape)
        x = x.unsqueeze(0)

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        results = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.3:
                if pred_num == 0:
                    with open(filename, mode='w') as fi:
                        fi.write('PREDICTIONS: '+ img_id + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                results.append(pt)
                coords = (pt[0], pt[1], pt[2], pt[3])
                #append to a text file
                pred_num += 1
                with open(filename, mode='a') as fi:
                    fi.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

        try:
            boxed_img = drawBoundingBox(img, results)
            cv2.imshow("image:", boxed_img)
            cv2.waitKey(2)
            time.sleep(3)
        except UnboundLocalError:
            pass


def test_on_my_face():
    # load net
    num_classes = 1 + 1 # +1 background
    net = BlazeFace('test', num_classes) # initialize Blazeface
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    transform = BaseTransform(128, (104, 117, 123))

    # load data
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        #x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True

        y = net(x)      # forward pass
        detections = y.data
        results = []
        pred_num = 0

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                results.append(pt)
                # append to a text file
                pred_num += 1
                j += 1

        try:
            boxed_img = drawBoundingBox(img, results)
            cv2.imshow("image:", boxed_img)
            cv2.waitKey(2)
        except UnboundLocalError:
            pass        # evaluation



def test_wider():
    # load net
    num_classes = 1 + 1 # +1 background
    net = BlazeFace('test', num_classes) # initialize Blazeface
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = WIDERDetection(WIDER_ROOT, ['wider_test'], None, WIDERAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation

    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(128, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    #test_wider()
    test_on_my_face()
