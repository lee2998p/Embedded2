import cv2
import numpy as np


class BaseTransform:
    '''
    Transform image to desired size before passing to face detector

    Params:
    size (Type: int) - Desired input size to face detector
    mean (Type: tuple of length 3 for 3 channels) - mean intensity values to perform normalization.

    '''

    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return self.base_transform(image, self.size, self.mean), boxes, labels


    def base_transform(self, image, size, mean):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x
