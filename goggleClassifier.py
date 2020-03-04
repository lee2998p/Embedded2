from __future__ import print_function, division
import argparse
import os
import time
import copy
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, models
import prettytable as pt

# TODO: rewrite the code to better follow these practices: https://gist.github.com/sloria/7001839
# TODO use Skorch instead of manual cross-validation
# TODO print metrics to file

# dunno if this is the right spot
NUM_CLASSES = 4

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
        #label = np.array([label])
        single_img = (image, label)

        return single_img


class GoggleClassifier:

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def __init__(self, gmodel, pretrained, data_location, image_folder, test_mode):

        if test_mode:
            # choose which model to train/evaluate
            model_ft = self.name_to_model(gmodel, pretrained)
            model_ft = model_ft.to(device)

            data_loaders, dataset_sizes, class_names = self.load_data(image_folder, data_location)

            criterion = nn.CrossEntropyLoss()
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

            # trains the model
            model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_loaders,
                                        dataset_sizes, num_epochs=15)

            # shows a picture (code from tutorial, could look a bit nicer)
            # self.visualize_model(model_ft, data_loaders, class_names)

            torch.save(model_ft, 'trained_model.pth')

            # show some statistics
            self.get_metrics(model_ft, data_loaders)

    def load_model(self, path):
        # load pth file
        self.model = torch.load(path)
        return self.model

    def classify(self, image):
        # run inference on one image
        self.model.eval()
        label = self.model(image)

        return label

    """ ------------------------ EVERYTHING BELOW IS FOR TESTING CLASSIFICATION --------------------------------- """

    def load_data(self, image_folder, data_location):

        if image_folder:
            # when using imageFolderharderDataset
            face_datasets = {x: datasets.ImageFolder(os.path.join(data_location, x),
                                                     self.data_transforms[x])
                             for x in ['train', 'val']}
            data_loaders = {x: DataLoader(face_datasets[x], batch_size=4,
                                          shuffle=True, num_workers=4)
                            for x in ['train', 'val']}
            dataset_sizes = {x: len(face_datasets[x]) for x in ['train', 'val']}
            class_names = face_datasets['train'].classes
            # when using imageFolderharderDataset
        else:
            # when using csvharderDataset (custom FacesDataset)
            # might be re-transforming every image for every access? might be unnecessary?
            full_face_dataset = FacesDataset('harderDataset.csv', 'csvharderDataset', self.data_transforms['val'])
            train_dataset, val_dataset = random_split(full_face_dataset, (491, len(full_face_dataset.pics) - 491))

            data_loaders = {'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
                            'val': DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)}
            dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
            class_names = ['both', 'glasses', 'goggles', 'neither']
            # when using csvharderDataset (custom FacesDataset)

        inputs, classes = next(iter(data_loaders['train']))
        #out = torchvision.utils.make_grid(inputs)
        return data_loaders, dataset_sizes, class_names

    def name_to_model(self, name, pretrained):
        if name == "resnet":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            return model
        elif name == "mobilenet":
            model = models.mobilenet_v2(pretrained=pretrained)
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
            # this gets stuck at 52% accuracy (everything is glasses) for some reason
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

        best_model_wts = copy.deepcopy(model.state_dict())
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

                # deep copy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(self, model, data_loaders, class_names, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}, actual: {}'.format(class_names[preds[j]], labels[j]))
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return

            model.train(mode=was_training)

    # view some images and get metrics of model
    def get_metrics(self, model, data_loaders):
        model.eval()
        # cm stands for confusion matrix
        cm = np.zeros((4, 4))
        num_correct = 0

        # create confusion matrix
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    if labels[j] == preds[j]:
                        num_correct += 1
                        cm[labels[j]][labels[j]] += 1
                    else:
                        cm[preds[j]][labels[j]] += 1

        # calculate metrics
        # precision, recall, F1 are columns; rows are each label
        metric_matrix = np.zeros((4, 3))
        for i in range(4):
            # precision
            metric_matrix[i][0] = cm[i][i] / (cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3])
            # recall
            metric_matrix[i][1] = cm[i][i] / (cm[0][i] + cm[1][i] + cm[2][i] + cm[3][i])
            # f1-score
            metric_matrix[i][2] = (2 * metric_matrix[i][0] * metric_matrix[i][1]) / (
                    metric_matrix[i][0] + metric_matrix[i][1])

        print("Number correct: {}".format(num_correct))

        print("-------------------PrettyTable------------------")
        x = pt.PrettyTable()
        # x.field_names = ["Predicted"]
        # x.add_column("Actual")
        # Predicted title should be horizontal, Actual should be vertical
        x.field_names = ["", "Both", "Glasses", "Goggles", "Neither"]
        x.add_row(["Both", cm[0][0], cm[0][1], cm[0][2], cm[0][3]])
        x.add_row(["Glasses", cm[1][0], cm[1][1], cm[1][2], cm[1][3]])
        x.add_row(["Goggles", cm[2][0], cm[2][1], cm[2][2], cm[2][3]])
        x.add_row(["Neither", cm[3][0], cm[3][1], cm[3][2], cm[3][3]])
        print(x)

        print("^^^ Double-check that this table is correct.")
        print("Print out which is predicted, which is actual.")


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--model', type=str, help='Select which model to run, default is mobilenetV2',
                        default='mobilenet')
    parser.add_argument('--pretrained', type=bool, help='Use pretrained model?', default=True)
    parser.add_argument('--directory', type=str, help='(Relative) Directory location of dataset', default='imageFolderharderDataset')
    parser.add_argument('--im', type=bool, help='Is the data sorted into ImageFolder structure?', default=False)
    parser.add_argument('--test_mode', type=str, help='Testing classifier?', default=False)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    plt.ion()  # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    gc = GoggleClassifier(gmodel=args.model, pretrained=args.pretrained, data_location=args.directory, image_folder=args.im, test_mode=args.test_mode)

    exit(0)
