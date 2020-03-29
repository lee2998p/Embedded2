import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

warnings.filterwarnings("once")

# LABELS_ONEHOT = {'both': [1, 0, 0, 0], 'goggles': [0, 1, 0, 0], 'glasses': [0, 0, 1, 0], 'none': [0, 0, 0, 1]}

LABELS_ONEHOT = {'both': 0, 'goggles': 1, 'glasses': 2, 'none': 3}

class GoggleDataset(Dataset):
    """Goggle Dataset"""

    def __init__(self, root_dir, transform=None):
        """

        @param root_dir (string): The root directory containing the data, with the images in the folder that is their label
        @param transform (callable, optional): Optional transform to be applied on sample
        """
        self.root_dir = root_dir
        self.dataset = {}
        for folder in [f.path for f in os.scandir(root_dir) if f.is_dir()]:
            label = folder.split('/')[-1]
            self.dataset[label] = [f.path for f in os.scandir(folder) if
                                   "jpg" in f.path.lower() or "png" in f.path.lower()]
        self.paths = [path for folder in self.dataset.values() for path in folder]
        self.transform = transform

    def __len__(self):
        return sum([len(k) for k in self.dataset.values()])

    def __getitem__(self, idx):

        def _get_sample(sample_path):
            label = sample_path.split('/')[-2]
            img = cv2.imread(sample_path)
            sample = {"image": img, 'label': label.lower()}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # Support list slicing
        if isinstance(idx, slice):
            return [_get_sample(path) for path in self.paths[idx]]

        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.paths[idx]
        return _get_sample(path)

    class GoogleDatasetIter:
        def __init__(self, gds):
            self._ds = gds
            self._idx = 0

        def __next__(self):
            if self._idx < len(self._ds):
                result = self._ds[self._idx]
                self._idx += 1
                return result
            raise StopIteration

    def __iter__(self):
        return self.GoogleDatasetIter(self)


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_w, new_h = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(img, (new_h, new_w))
        return {'image': img, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        nums = torch.rand(2)
        top = int(nums[0] * (h - new_h))
        left = int(nums[1] * (w - new_w))
        img = img[top:top + new_h, left:left + new_w]
        if img.shape[0] != 100 or img.shape[1] != 100:
            print(f"\tleft is {left}, top is {top}")

        return {"image": img, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Convert from numpy dim (HxWxC) ordering to torch dim (CxHxW) ordering
        img = img.transpose((2, 0, 1))
        # Keep the image on cpu and transfer for training
        return {'image': torch.from_numpy(img).float().to('cpu'), 'label': torch.tensor(LABELS_ONEHOT[label])}


if __name__ == "__main__":

    def show_batch(sample_batch):
        img_batch, label_batch = sample_batch['image'], sample_batch['label']
        batch_size = len(img_batch)
        print(f"batch size is {batch_size}")
        grid = utils.make_grid(img_batch)
        plt.title('Batch from dataloader')
        plt.imshow(np.flip(grid.numpy().transpose(1, 2, 0), 2))
        plt.axis('off')
        plt.ioff()
        plt.show()


    ds = GoggleDataset("/home/jrmo/data/faces/labeled/",
                       transform=transforms.Compose([Rescale(128), RandomCrop(100), ToTensor()]))
    dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)
    label_count = {}
    for sample in dataloader:
        if not sample['label'][0] in label_count.keys():
            label_count[sample['label'][0]] = 0
        label_count[sample['label'][0]] += 1

    print(f"Dataset contents: {len(dataloader.dataset)} images")
    for label in list(label_count.keys()):
        print(f"{label}: {label_count[label] / len(dataloader.dataset) * 100:.4}%")
