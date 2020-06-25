import argparse
import copy
import json
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

# 80/20 training/validation split
VAL_SPLIT = 0.2

"""Train a custom CNN classifier trained on activation maps output by Blazeface."""


def tensor_loader(path):
    """Required for the custom dataset created in load_data."""
    return torch.load(path)


# TODO use train_model from goggle_classifier instead once file reorganization is done
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
    This is quite similar to goggle_classifier's load_data method, but no transforms are required here.
    @param data_location: Directory in DatasetFolder structure containing .pt files to train on.
    @return: The Pytorch Dataloader, size of training and validation datasets, and dataset class names.
    """
    dataset = datasets.DatasetFolder(data_location, tensor_loader, ('.pt',))

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    splits = torch.utils.data.random_split(dataset, [train_size, val_size])
    tensor_datasets = {'train': splits[0],
                       'val': splits[1]}

    # TODO batch_size could be 4... need to understand how transforms are handled
    data_loaders = {'train': DataLoader(tensor_datasets['train'], batch_size=1,
                                        shuffle=True, num_workers=4),
                    'val': DataLoader(tensor_datasets['val'], batch_size=1,
                                      shuffle=True, num_workers=4)}
    dataset_sizes = {x: len(tensor_datasets[x]) for x in ['train', 'val']}

    return data_loaders, dataset_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--directory', type=str, help='Relative directory location of dataset', default='dataset')
    args = parser.parse_args()

    # TensorBoard writer
    writer = SummaryWriter()

    # the CNN architecture to train on. The exact layers involved is an open question.
    model = nn.Sequential(
        # One or more convolutional layers here?
        nn.Flatten(),
        nn.Dropout(0.2),
        # 6144 is 96 x 8 x 8 (size of the activation map)
        nn.Linear(6144, 1000),
        # 3 is the number of classes
        nn.Linear(1000, 3)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with open("act_map_params.json", "r") as params_file:
        params = json.load(params_file)

    data_loaders, dataset_sizes = load_data(args.directory)
    trained_model = train_model(model, data_loaders, dataset_sizes, params)
    torch.save(trained_model, 'act_map_model.pth')

    exit(0)
