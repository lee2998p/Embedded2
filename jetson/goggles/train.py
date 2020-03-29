import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from goggleClassifier_lightweight import SimpleCNN
from torchvision import transforms
from GoggleDataloader import GoggleDataset, RandomCrop, Rescale, ToTensor, LABELS_ONEHOT
from torch.utils.data.dataloader import DataLoader
from typing import Union
import numpy as np
from tqdm import tqdm


def train_network(network: SimpleCNN, loss_fn: nn.CrossEntropyLoss, optimizer: Union[optim.SGD, optim.Adam],
                  dataloader: DataLoader,
                  num_epoch=10, device=None):
    overall_prog = tqdm(total=num_epoch)
    epoch_prog = tqdm(total=len(dataloader.dataset))
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, batched_sample in enumerate(dataloader):
            inputs, labels = batched_sample['image'], batched_sample['label']
            if device is not None:
                inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            print(f"type outputs {outputs.size()}, type labels {type(labels)}")
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 250 == 0:
                print(f'[{epoch + 1}, {(i + 1):5d}] loss: {(running_loss / 250):.3f}')
                running_loss = 0.0
            epoch_prog.update(dataloader.batch_size)
        epoch_prog.clear()
        overall_prog.update(1)
    print("Done training")
    torch.save(network.state_dict(), 'simple_cnn.pth')


def get_data_loader(dataset, batch_size=2, split=0.2, shuffle_dataset=True, seed=1337):
    ds_size = len(dataset)
    idxz = list(range(ds_size))
    split_idx = int(ds_size * split)
    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(idxz)
    train_idx, val_idx = idxz[split_idx:], idxz[:split_idx]
    train_sample = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sample = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sample)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sample)
    return train_loader, val_loader


if __name__ == "__main__":
    net = SimpleCNN(32, 4)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net.to(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dataset = GoggleDataset("/home/jrmo/data/faces/labeled",
                            transform=transforms.Compose([Rescale(128), RandomCrop(100), ToTensor()]))

    train_loader, val_loader = get_data_loader(dataset)
    train_network(net, loss_fn, optimizer, train_loader, device=device)
