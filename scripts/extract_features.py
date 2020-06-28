"""quick-and-dirty way to load images through blazeface and
save activation maps after backbone2 as .pt files"""

from __future__ import print_function, division

import argparse

import numpy as np
import torch
from BlazeFace_2.blazeface import BlazeFace
from data import BaseTransform
from torch.utils.data import Dataset
from torchvision import transforms
from torch import multiprocessing

from goggle_classifier import load_data, train_model
# TODO import FaceDetector from main

VAL_SPLIT = .2
activation = {}

"""Save activation maps of images run partway through a CNN. Currently specific to Blazeface. 
Use this script before running learn_features.py"""


# custom dataset for applying different transforms to train and val data
class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, item):
        return self.map(self.dataset[item][0]), self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


"""Transformations to perform on images before running inference. Blazeface requires 128x128 images."""
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
}


# TODO use from main
# might need to rewrite slightly to work with both
class FaceDetector:
    def __init__(self, trained_model, detection_threshold=0.75, cuda=True, set_default_dev=False):
        """
        Creates a FaceDetector object
        @param trained_model: A string path to a trained pth file for a ssd model trained in face detection
        @param detection_threshold: The minimum threshold for a detection to be considered valid
        @param cuda: Whether or not to enable CUDA
        @param set_default_dev: Whether or not to set the default device for PyTorch
        """

        self.device = torch.device("cpu")
        self.net = BlazeFace()
        self.net.load_weights("blazeface.pth")
        self.net.load_anchors("BlazeFace_2/anchors.npy")
        self.model_name = 'blazeface'
        self.net.min_score_thresh = 0.75
        self.net.min_suppression_threshold = 0.3
        self.transformer = BaseTransform(128, (104, 117, 123))

        self.detection_threshold = detection_threshold
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if set_default_dev:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
        elif set_default_dev:
            torch.set_default_tensor_type('torch.FloatTensor')

        print(f'Moving network to {self.device.type}')
        self.net.to(self.device)

        self.net.eval()

    def detect(self, image):
        """
        Performs face detection on the image passed
        @param image: A numpy array representing an image
        @return: The bounding boxes of the face(s) that were detected formatted (upper left corner(x, y) , lower right corner(x,y))
        """

        img = image.squeeze()

        detections = self.net.predict_on_image(img)
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()

        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        print("Found %d faces" % detections.shape[0])

        bboxes = []
        for i in range(detections.shape[0]):
            ymin = detections[i, 0] * image.shape[0]
            xmin = detections[i, 1] * image.shape[1]
            ymax = detections[i, 2] * image.shape[0]
            xmax = detections[i, 3] * image.shape[1]

            img = img / 127.5 - 1.0

            for k in range(6):
                kp_x = detections[i, 4 + k * 2] * img.shape[1]
                kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]

            bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes


def run_inference(detector, data_location, data_loaders,  class_names):
    """
    Run all images in the dataset through the face detector, saving activation maps as .pt files.
    @param detector: The FaceDetector. Must be a Blazeface model for now.
    @param data_location: Directory in Imagefolder structure containing the images to train on.
    @param data_loaders: A dictionary containing a 'train' DataLoader
    and a 'val' DataLoader (returned by load_data).
    @param class_names: List of class names. Used for saving .pt files in the same folder as their respective image.
    """

    i = 0
    for phase in ['train', 'val']:
        # iterate over all data
        for inputs, labels in data_loaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # TODO blazeface never appears to detect any faces...
            # the activation map is saved when .detect is called, so we need to call this even though it's not used
            output = detector.detect(inputs)

            # TODO I think there's a nicer way to write this file name, something like os.path.join?
            # preferably should/could have the same name as the original image, but with a .pt extension.
            # does the name get ignored when loading though?
            torch.save(activation['backbone2'],
                       data_location + '/' + class_names[labels.item()] + '/act_map' + str(i) + '.pt')
            i += 1


def get_activation(name):
    # TODO find reference for why this is the correct way
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description='Run classification on a dataset')
    parser.add_argument('--directory', type=str, help='(Relative) Directory location of dataset', default='dataset')
    parser.add_argument('--trained_model', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")

    args = parser.parse_args()

    # TODO I think this is necessary when using CUDA b/c we have two processes running?
    multiprocessing.set_start_method('spawn')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO should just import FaceDetector with minor changes to limit duplicate code
    detector = FaceDetector(trained_model=args.trained_model, cuda=args.cuda and torch.cuda.is_available(),
                            set_default_dev=True)

    if detector.model_name == 'blazeface':
        # saves activation map after backbone2 into activation['backbone2']
        detector.net.backbone2.register_forward_hook(get_activation('backbone2'))

    data_loaders, dataset_sizes, class_names = load_data(args.directory, data_transforms)

    run_inference(detector, args.directory, data_loaders, class_names)

    exit(0)
