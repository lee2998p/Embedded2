from __future__ import print_function
import torch
import torch.nn as nn
import os
import time
import argparse
import numpy as np
from model import BlazeFace
from wider_face import WIDERAnnotationTransform, WIDERDetection, WIDER_CLASSES
from wider_face import WIDER_ROOT
from data import wider_face
from utils.augmentations import SSDAugmentation
from torch.autograd import Variable
import torch.utils.data as data





parser = argparse.ArgumentParser(
    description='Blazeface Evaluation on Custom Data')
parser.add_argument('--trained_model',
                    default='weights/210.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--wider_root', default=WIDER_ROOT,
                    help='Location of WIDER root directory')

args = parser.parse_args()



def test_net(net, dataset):
    '''
    All detections are collected into:
        all_boxes[cls][image] = N x 5 array of detections in
        (x1, y1, x2, y2, score)
    '''
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(WIDER_CLASSES)+1)]

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        detections = net(x).data
        print (detections)


    '''
    output_dir = get_output_dir('blazeface', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    for i in range(num_images):
        x = Variable(im.unsqueeze(0))

        detections = net(x).data
        #print(detections)
        # skip j = 0, because it's the background class

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)
    '''


def detection_collate(batch=32):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == '__main__':
    # load net
    num_classes = 1 + 1                      # +1 for background
    net = BlazeFace('test', num_classes)            # initialize Blazeface
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()
    print('Finished loading model!')

    #TODO: Change this to custom dataset
    cfg = wider_face
    # load data
    custom_dataset = WIDERDetection(root=WIDER_ROOT, image_set='wider_train', transform=SSDAugmentation(cfg['min_dim']))


    test_net(net, custom_dataset)