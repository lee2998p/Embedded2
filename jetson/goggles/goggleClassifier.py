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
import pandas as pd
import prettytable as pt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, models

# TODO: rewrite the code to better follow these practices: https://gist.github.com/sloria/7001839
# TODO use Skorch instead of manual cross-validation

# dunno if this is the right spot
NUM_CLASSES = 3


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


class FacesDataset(Dataset):
    """Glasses/goggles dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.pics = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # rewrote this from tutorial; for some reason they return single_img as a dictionary and not a tuple
        # using Image.open and .convert('RGB') is what ImageFolder source does
        img_name = os.path.join(self.root_dir,
                                self.pics.iloc[item, 0])
        image = Image.open(img_name)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.pics.iloc[item, 1]
        # the line below caused errors with CrossEntropyLoss and Mobilenet.
        # see here if other models have issues
        # label = np.array([label])
        single_img = (image, label)

        return single_img


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

    def __init__(self, gmodel, pretrained, data_location, image_folder, train_mode, device):
        if train_mode:
            # choose which model to train/evaluate
            model_ft = self.name_to_model(gmodel, pretrained)
            # model_ft = model_ft.load_state_dict(torch.load('3classv2_Apr_6.pth'))
            model_ft = model_ft.to(device)

            data_loaders, dataset_sizes, class_names = self.load_data(image_folder, data_location)

            # hyperparameters
            lr = 0.001
            momentum = 0.9
            step_size = 10
            gamma = 0.25
            num_epochs = 50

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

            # shows a picture and its label
            # (code from tutorial, could look a bit nicer)
            # self.visualize_model(model_ft, data_loaders, class_names)

            torch.save(model_ft, 'trained_model.pth')

            # show some statistics
            self.get_metrics(model_ft, data_loaders, class_names)
        else:
            self.model = torch.load(pretrained, map_location=device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomGrayscale(1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def load_model(self, path):
        # load pth file
        self.model = torch.load(path)

        return self.model

    def classify(self, face):
        # TODO assertion error after a certain amount of time?
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

    """ ------------------------ EVERYTHING BELOW IS FOR TESTING CLASSIFICATION --------------------------------- """

    def load_data(self, image_folder, data_location):

        if image_folder:
            # when using ImageFolder structure
            # print(os.path.abspath(data_location))
            # print(os.path.join(data_location, 'train'))
            face_datasets = {x: datasets.ImageFolder(os.path.join(os.path.abspath(data_location), x),
                                                     self.data_transforms[x])
                             for x in ['train', 'val']}
            data_loaders = {x: DataLoader(face_datasets[x], batch_size=4,
                                          shuffle=True, num_workers=4)
                            for x in ['train', 'val']}
            dataset_sizes = {x: len(face_datasets[x]) for x in ['train', 'val']}
            class_names = face_datasets['train'].classes
            print('class_names are {}'.format(class_names))
        else:
            # when using csv file TODO update to generalize this
            # might be re-transforming every image for every access? might be unnecessary?
            full_face_dataset = FacesDataset('harderDataset.csv', 'csvharderDataset', self.data_transforms['val'])
            train_dataset, val_dataset = random_split(full_face_dataset, (491, len(full_face_dataset.pics) - 491))

            data_loaders = {'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
                            'val': DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)}
            dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
            class_names = ['both', 'glasses', 'goggles', 'neither']

        return data_loaders, dataset_sizes, class_names

    def name_to_model(self, name, pretrained):
        if name == "resnet":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            return model
        elif name == "mobilenet":
            model = models.mobilenet_v2(pretrained=pretrained)

            print('Using Mobilenet')

            # freeze all the layers, then make a new classifier layer to match # classes
            for param in model.parameters():
                param.requires_grad = False

            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, NUM_CLASSES)
            )
            return model
        elif name == "resnext":
            model = models.resnext50_32x4d(pretrained=pretrained)
            print("TODO CHANGE NUMBER OF CLASSES FOR RESNEXT")
            return model
        elif name == "shufflenet":
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            return model

        print("Couldn't match model")
        return None

    def imshow(self, inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def train_model(self, model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, num_epochs=10):
        since = time.time()

        best_acc = 0.0

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
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s \n'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load final epoch's weights
        model.load_state_dict(copy.deepcopy(model.state_dict()))
        return model

    def get_metrics(self, model, data_loaders, class_names):
        model.eval()
        # cm stands for confusion matrix
        cm = np.zeros((4, 4))
        num_correct = 0
        full_correct = []
        full_pred = []

        # create confusion matrix
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    full_correct.append(labels[0].item())
                    full_pred.append(preds[j].item())

                    if labels[j] == preds[j]:
                        num_correct += 1
                        cm[labels[j]][labels[j]] += 1
                    else:
                        cm[preds[j]][labels[j]] += 1

        print('F1-score: {}'.format(f1_score(full_correct, full_pred, average="weighted")))
        print('Precision: {}'.format(precision_score(full_correct, full_pred, average="weighted")))
        print('Recall: {}\n'.format(recall_score(full_correct, full_pred, average="weighted")))

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
    parser.add_argument('--model', type=str, help='Select which model to run, default is mobilenetV2',
                        default='mobilenet')
    parser.add_argument('--pretrained', type=bool, help='Use pretrained model?', default=True)
    parser.add_argument('--directory', type=str, help='(Relative) Directory location of dataset', default='dataset')
    parser.add_argument('--im', type=bool, help='Is the data sorted into ImageFolder structure?', default=False)
    parser.add_argument('--test_mode', type=str, help='Testing classifier?', default=False)
    args = parser.parse_args()

    # uncomment this line if you want results logged to a text file and stdout
    # sys.stdout = Logger()
    print("Time: {}".format(time.asctime(time.localtime())))

    warnings.filterwarnings("ignore")
    plt.ion()  # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    gc = GoggleClassifier(gmodel=args.model, pretrained=args.pretrained, data_location=args.directory,
                          image_folder=args.im, train_mode=args.test_mode)

    exit(0)
