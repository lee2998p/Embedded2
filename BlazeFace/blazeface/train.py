from data import wider_face
from data import HOME
from wider_face import WIDERDetection
from wider_face import WIDER_ROOT
from wider_face import WIDER_ROOT
from layers import MultiBoxLoss
from utils.augmentations import SSDAugmentation
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import argparse
from model import *
import numpy as np
import time


parser = argparse.ArgumentParser(
    description='BlazeFace Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='WIDER', choices=['VOC', 'COCO', 'WIDER'],
                    type=str)
parser.add_argument('--dataset_root', default=HOME + '/PyTorch_BlazeFace/WIDER',
                    help='Dataset root directory path')
# parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
#                     help='Pretrained base model')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--epochs', default=50,
                    help='Num epochs for training')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')

def train():

    net = BlazeFace(phase='train', num_classes=2)

    if args.dataset == 'WIDER':
        cfg = wider_face
        dataset = WIDERDetection(root=WIDER_ROOT,
                                 transform=SSDAugmentation(cfg['min_dim'],
                                                           ))
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)

    # for param in net.parameters():
        # print(param.data, param.size)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()

    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    num_iterations = len(dataset) // args.batch_size

    print('Training BlazeFace on:', dataset.name)
    print('Using the specified args:')
    print(args)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)

    epochs = args.epochs

    for epoch in range(epochs):
        batch_iterator = iter(data_loader)
        for iteration in range(args.start_iter, cfg['max_iter']):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            images, targets = next(batch_iterator)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
                # forward
            t0 = time.time()
            # print(images[0].shape)
            # print("here")
            out = net(images)
            # print("forward prop done")
            # backprop
            optimizer.zero_grad()

            loss_l, loss_c = criterion(out, targets)
            print(loss_l)
            print(loss_c)

            loss = loss_l + loss_c
            loss.backward()

            optimizer.step()
            t1 = time.time()
            # loc_loss += loss_l.data[0]
            # conf_loss += loss_c.data[0]

            loc_loss += loss_l.data
            conf_loss += loss_c.data
            print(loc_loss)
            print(conf_loss)

            if iteration % 1 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')

            if iteration != 0 and iteration % 1 == 0:
                        print('here')
                        print('Saving state, iter:', iteration)
                        torch.save(net.state_dict(), args.save_folder +
                                   repr(iteration) + '.pth')

        torch.save(net.state_dict(),
                           args.save_folder + 'Blaze_' + args.dataset + repr(epoch) + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def detection_collate(batch):
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

train()
