import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from GoggleDataloader import GoggleDataset, RandomCrop, Rescale, ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm

from jetson.goggles.SimpleClassifier import SimpleCNN


# TODO allow for continuation of training
# TODO capture sigkill to dump model to file and exit


def train_network(network, loss_fn, optimizer, dataloader, num_epoch, cp_file_path, cp_file_name, cp_rate, device=None):
    # Pretty progress bar
    epoch_prog = tqdm(total=len(dataloader.dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}')
    for epoch in range(num_epoch):
        running_loss = 0.0
        # Show progress
        epoch_prog.set_description(f'Training epoch {epoch}/{num_epoch} -- running loss {running_loss}')

        for i, batched_sample in enumerate(dataloader):
            inputs, labels = batched_sample['image'], batched_sample['label']
            # Make sure we're using the right device
            # Moving whole dataset to gpu might be faster, but take more memory
            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            print(outputs.shape)
            import sys

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 250 == 249:
                running_loss = 0.0
            epoch_prog.update(dataloader.batch_size)

        epoch_prog.refresh()
        epoch_prog.reset()

        if epoch % cp_rate == 0:
            network_save_path = os.path.join(cp_file_path, f"{cp_file_name}_{epoch}.pth")
            torch.save(network.state_dict(), network_save_path)

    print("Done training, saving network")
    network_save_path = os.path.join(cp_file_path, f"{cp_file_name}_b{dataloader.batch_size}_e{epoch}.pth")
    print(f"Saving final model to {network_save_path}")
    torch.save(network.state_dict(), network_save_path)
    epoch_prog.close()

    return network


def get_data_loader(dataset, batch_size=2, split=0.2, shuffle_dataset=True, seed=852):
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


def test_network(net, testloader, device=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['image'], data['label']

            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)
            print(inputs.shape)
            outputs = net(inputs)
            _, preds = torch.max(outputs.data, 1)

            print("*" * 20)
            print(f"Preds: {preds}\nLabels: {labels}")
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {100 * (correct / total)} %%')


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for goggle classifier")
    parser.add_argument('--cuda', '-c', type=bool, default=True)
    parser.add_argument('--data_dir', '-d', type=str, help='The directory with the labeled data', required=True)
    parser.add_argument('--output_dir', '-o', type=str, help='The directory to output models and checkpoints',
                        required=True)
    parser.add_argument('--model_name', '-n', type=str, help='The base name for the model files',
                        default='goggle_classifier')
    parser.add_argument('--rate', '-r', type=int, default=10,
                        help='How many epochs to process between saving checkpoints')
    parser.add_argument('--num_epoch', '-e', type=int, default=20, help="The number of epochs to train for")
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="The size of batches to use")
    parser.add_argument('--state_dict', '-s', type=str, default=None, help="State dict to load from")
    parser.add_argument('--test_only', '-t', type=bool, default=False,
                        help='Only run the testing segment of the script')

    args = parser.parse_args()
    net = SimpleCNN()
    if args.state_dict is not None:
        net.load_state_dict(torch.load(args.state_dict))
        print(f"Loaded network from state dict at: {args.state_dict}")
    print(f"Network has {net.param_count()} parameters")

    device = torch.device('cpu')
    if torch.cuda.is_available():
        if args.cuda:
            device = torch.device('cuda:0')
            net.to(device)
            print('Network on GPU')
        else:
            print("Cuda is available, you should use it")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dataset = GoggleDataset(args.data_dir, transform=transforms.Compose([Rescale(128), RandomCrop(100), ToTensor()]))
    train_loader, val_loader = get_data_loader(dataset, batch_size=args.batch_size)
    print(f"Training network for {args.num_epoch} epochs, ",
          f"with a batch size of {args.batch_size}, ",
          f"over {len(train_loader.dataset)} images")

    if not args.test_only:
        net = train_network(net, loss_fn, optimizer, train_loader, args.num_epoch, args.output_dir, args.model_name,
                            args.rate, device=device)

    test_network(net, testloader=val_loader, device=device)
