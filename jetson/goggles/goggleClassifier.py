from __future__ import print_function, division

import argparse
import copy
import os
import sys
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import prettytable as pt
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models

import params

# TODO use Skorch for cross-validation

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("results.txt", "w+")
        open("results.txt", )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass


# custom dataset for applying different transforms to train and val data
class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, item):
        return self.map(self.dataset[item][0]), self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


class GoggleClassifier:
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomGrayscale(1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomGrayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def __init__(self, data_location, test_mode, model, device):
        self.device = device

        if not test_mode:
            # choose which model to train/evaluate
            model_ft = self.get_model()
            # model_ft = model_ft.load_state_dict(torch.load('3classv2_Apr_6.pth'))
            model_ft = model_ft.to(device)

            data_loaders, dataset_sizes, class_names = self.load_data(data_location)

            # hyperparameters
            lr = params.hparams['lr']
            momentum = params.hparams['momentum']
            step_size = params.hparams['step_size']
            gamma = params.hparams['gamma']
            num_epochs = params.hparams['num_epochs']

            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

            print('Running model with lr={}, momentum={}, step_size={}, gamma={}, num_epochs={}\n'.format(lr, momentum,
                                                                                                          step_size,
                                                                                                          gamma,
                                                                                                          num_epochs))

            # trains the model
            model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_loaders,
                                        dataset_sizes, num_epochs=num_epochs)
            torch.save(model_ft, 'trained_model.pth')

            # show some statistics
            self.get_metrics(model_ft, data_loaders, class_names)
        else:
            # WIP code. We usually do testing directly in face_detector.py
            self.model = self.load_model(model)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomGrayscale(1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def load_model(self, path):
        # load pth file
        if path is not None:
            self.model = torch.load(path)
            return self.model
        else:
            print('Must supply a .pth file location')
            exit(1)

    def classify(self, face):
        # this is the same code as in face_detector
        if 0 in face.shape:
            pass
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        transformed_face = self.transform(pil_face)
        face_batch = transformed_face.unsqueeze(0)
        with torch.no_grad():
            face_batch = face_batch.to(self.device)
            labels = self.model(face_batch)
            m = torch.nn.Softmax(1)
            softlabels = m(labels)
            print('Probability labels: {}'.format(softlabels))

            # using old; fix error TODO
            _, pred = torch.max(labels, 1)

        return pred, softlabels

    def load_data(self, data_location):
        dataset = datasets.ImageFolder(os.path.abspath(data_location))
        class_names = dataset.classes

        val_size = int(VAL_SPLIT * len(dataset))
        train_size = len(dataset) - val_size
        face_datasets = {}
        face_datasets['train'], face_datasets['val'] = torch.utils.data.random_split(dataset, [train_size, val_size])

        face_datasets['train'] = MapDataset(face_datasets['train'], self.data_transforms['train'])
        face_datasets['val'] = MapDataset(face_datasets['val'], self.data_transforms['val'])

        # code for oversampling if we have a class imbalance
        # class_count = np.unique(face_datasets['train'].targets, return_counts=True)[1]
        # weight = 1. / class_count
        # samples_weight = weight[face_datasets['train'].targets]
        # samples_weight = torch.from_numpy(samples_weight)
        # train_sampler (oversampling) instead of random sampling to handle class imbalance
        # train_sampler = WeightedRandomSampler(samples_weight, int(sum(class_count)))

        data_loaders = {'train': DataLoader(face_datasets['train'], batch_size=4,
                                            shuffle=True, num_workers=4),
                        'val': DataLoader(face_datasets['val'], batch_size=4,
                                          shuffle=True, num_workers=4)}
        dataset_sizes = {x: len(face_datasets[x]) for x in ['train', 'val']}

        print('class_names are {}'.format(class_names))
        return data_loaders, dataset_sizes, class_names

    def get_model(self):
        model = models.mobilenet_v2(pretrained=True)

        # freeze all the layers, then make a new classifier layer to match # classes
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, NUM_CLASSES)
        )
        return model

    def train_model(self, model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, num_epochs=10):
        since = time.time()
        epoch_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # train and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # iterate over data
                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        # print('Outputs are {}'.format(outputs))
                        # print('Labels are {}'.format(labels))

                        loss = criterion(outputs, labels)

                        # backward
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))


                if phase == 'train':
                    scheduler.step()
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/val', epoch_acc, epoch)

                # Save checkpoints
                if epoch != 0 and epoch % 10 == 0:
                    print('Saving state, epoch:', epoch)
                    torch.save(model.state_dict(),
                               repr(epoch) + '.pth')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s \n'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Final val Acc: {:4f}'.format(epoch_acc))

        # load final epoch's weights
        model.load_state_dict(copy.deepcopy(model.state_dict()))
        return model

    def get_metrics(self, model, data_loaders, class_names):
        model.eval()
        full_correct = []
        full_pred = []

        # collect true and predicted labels for sklearn
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    full_correct.append(labels[j].item())
                    full_pred.append(preds[j].item())

        acc = skm.accuracy_score(full_correct, full_pred)
        fone_score = skm.f1_score(full_correct, full_pred, average="weighted")
        precision = skm.precision_score(full_correct, full_pred, average="weighted")
        recall = skm.recall_score(full_correct, full_pred, average="weighted")
        cm = skm.confusion_matrix(full_correct, full_pred)

        print('F1-score: {}'.format(fone_score))
        print('Precision: {}'.format(precision))
        print('Recall: {}\n'.format(recall))

        # hyperparameter printing will be more interesting when we do tuning
        writer.add_hparams(params.hparams, {'hparam/accuracy': acc, 'hparam/f1_score': fone_score,
                                            'hparam/precision': precision, 'hparam/recall': recall})
        #writer.add_text('Confusion matrix', cm)
        writer.flush()

        print("------------------Confusion matrix------------------")
        x = pt.PrettyTable()
        x.add_column('', class_names)
        for i in range(NUM_CLASSES):
            col = []
            for j in range(NUM_CLASSES):
                col.append(cm[j][i])
            x.add_column(class_names[i], col)

        print(x)
        print('Columns are actual labels, rows are predicted labels\n\n')


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--directory', type=str, help='(Relative) Directory location of dataset', default='dataset')
    parser.add_argument('--test_mode', action='store_true', help='Test classifier. Must be used with --model',
                        default=False)
    parser.add_argument('--model', type=str, help='(Relative) location of model to load', default=None)
    args = parser.parse_args()

    # 3 classes, train/val split 80/20
    NUM_CLASSES = 3
    VAL_SPLIT = .2

    # TensorBoard writer
    writer = SummaryWriter()

    # uncomment this line if you want results saved to both a text file and stdout
    # sys.stdout = Logger()
    print("Time: {}".format(time.asctime(time.localtime())))

    warnings.filterwarnings("ignore")
    plt.ion()  # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    gc = GoggleClassifier(data_location=args.directory, test_mode=args.test_mode, model=args.model, device=device)

    exit(0)
