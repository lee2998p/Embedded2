import cv2
import numpy as np


class BaseTransform:
    def __init__(self, size, mean):
        '''
        Transform image to desired size before passing to face detector

        Params:
        size (Type: int) - Desired input size to face detector
        mean (Type: tuple of length 3 for 3 channels) - mean intensity values to perform normalization.

        '''

        self.size = size
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
        else:
            self.mean = None

    def __call__(self, image, boxes=None, labels=None):
        return self.base_transform(image), boxes, labels


    def base_transform(self, image):
        x = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        if self.mean is not None:
            x -= self.mean
        x = x.astype(np.float32)
        return x
