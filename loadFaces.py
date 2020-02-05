from __future__ import print_function, division
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import torchvision
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class FacesDataset(Dataset):
    """Glasses/goggles dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.pics = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pics.iloc[item, 0])
        image = io.imread(img_name)
        label = self.pics.iloc[item, 1]
        # make the one item into an array :)
        label = np.array([label])
        single_img = {'image': image, 'label': label}

        if self.transform:
            single_img = self.transform(single_img)

        return single_img


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

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
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)


# view some images and get metrics of model
def get_metrics(model):
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
        metric_matrix[i][2] = (2 * metric_matrix[i][0] * metric_matrix[i][1]) / (metric_matrix[i][0] + metric_matrix[i][1])

    print("Confusion matrix")
    print('[{}] [{}] [{}] [{}]'.format(cm[0][0], cm[0][1], cm[0][2], cm[0][3]))
    print('[{}] [{}] [{}] [{}]'.format(cm[1][0], cm[1][1], cm[1][2], cm[1][3]))
    print('[{}] [{}] [{}] [{}]'.format(cm[2][0], cm[2][1], cm[2][2], cm[2][3]))
    print('[{}] [{}] [{}] [{}]'.format(cm[3][0], cm[3][1], cm[3][2], cm[3][3]))
    print("Number correct: {}".format(num_correct))
    print("Metrics")
    print('[{}] [{}] [{}]'.format(metric_matrix[0][0], metric_matrix[0][1], metric_matrix[0][2]))
    print('[{}] [{}] [{}]'.format(metric_matrix[1][0], metric_matrix[1][1], metric_matrix[1][2]))
    print('[{}] [{}] [{}]'.format(metric_matrix[2][0], metric_matrix[2][1], metric_matrix[2][2]))
    print('[{}] [{}] [{}]'.format(metric_matrix[3][0], metric_matrix[3][1], metric_matrix[3][2]))


data_transforms = {
    # TODO check which transforms we want
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

face_datasets = {x: datasets.ImageFolder(os.path.join('pics', x),
                                         data_transforms[x])

                 for x in ['train', 'val']}
data_loaders = {x: DataLoader(face_datasets[x], batch_size=4,
                              shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dataset_sizes = {x: len(face_datasets[x]) for x in ['train', 'val']}
class_names = face_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs, classes = next(iter(data_loaders['train']))

out = torchvision.utils.make_grid(inputs)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# changed 2 --> 4 because there are 4 classes?
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# trains the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

# shows a few pictures (code from tutorial)
#visualize_model(model_ft)

# show some statistics
get_metrics(model_ft)
