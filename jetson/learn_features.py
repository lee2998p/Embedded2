import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

# 80/20 training/validation split
VAL_SPLIT = 0.2


def tensor_loader(path):
    """Required for the custom dataset created in load_data."""
    return torch.load(path)


# TODO use train_model from goggleClassifier instead once file reorganization is done
def train_model(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, num_epochs=10):
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
            # else:
            writer.add_scalar('Loss/val', epoch_loss, epoch)
            writer.add_scalar('Accuracy/val', epoch_acc, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(epoch_acc))

    # load final epoch's weights
    model.load_state_dict(copy.deepcopy(model.state_dict()))
    return model


def load_data(data_location):
    """
    Create a Pytorch Dataloader for activation maps output by the face detector.
    This is quite similar to goggleClassifier's load_data method, but no transforms are required here.
    @param data_location: Directory in DatasetFolder structure containing .pt files to train on.
    @return: The Pytorch Dataloader, size of training and validation datasets, and dataset class names.
    """
    dataset = datasets.DatasetFolder(data_location, tensor_loader, ('.pt',))
    class_names = dataset.classes

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    splits = torch.utils.data.random_split(dataset, [train_size, val_size])
    tensor_datasets = {'train': splits[0],
                       'val': splits[1]}

    # TODO batch_size could be 4... how to transform correctly?
    data_loaders = {'train': DataLoader(tensor_datasets['train'], batch_size=1,
                                        shuffle=True, num_workers=4),
                    'val': DataLoader(tensor_datasets['val'], batch_size=1,
                                      shuffle=True, num_workers=4)}
    dataset_sizes = {x: len(tensor_datasets[x]) for x in ['train', 'val']}

    print('class_names are {}'.format(class_names))
    return data_loaders, dataset_sizes, class_names


parser = argparse.ArgumentParser(description='Run classification on a dataset')
parser.add_argument('--directory', type=str, help='(Relative) Directory location of dataset', default='dataset')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
args = parser.parse_args()

# TensorBoard writer
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

model = nn.Sequential(
    # nn.conv2d here?
    nn.Flatten(),
    nn.Dropout(0.2),
    # 6144 is 96 x 8 x 8 (size of the activation map)
    nn.Linear(6144, 1000),
    # 3 is the number of classes
    nn.Linear(1000, 3)
)

model = model.to(device)

# placeholder values for hyperparams for now
data_loaders, dataset_sizes, class_names = load_data(args.directory)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 1)
trained_model = train_model(model, nn.CrossEntropyLoss(), optimizer,
                            lr_scheduler, data_loaders, dataset_sizes, num_epochs=100)
torch.save(trained_model, 'actmap_model.pth')

exit(0)
