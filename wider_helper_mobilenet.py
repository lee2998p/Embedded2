import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from torchvision import transforms

#function to fetch and return input images and target annotations
class WiderFaceDetection(data.Dataset):
    #outputs of __init__ -> list of image paths, list of lists (different images) of lists (different faces) of annotations (for each face)
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc #widerface preprocessing function
        self.imgs_path = [] #list of image paths
        self.words = [] #list of lists of labels (declared below)
        self.num_faces = [] #list of number of faces in each image
        f = open(txt_path,'r') #reads label.txt (in train/val folder)
        lines = f.readlines() #stores list of individual lines
        isFirst = True 
        new_im = False
        labels = [] #list of 4 bounding box numbers (x1,y1,w,h) + 6 image-mods (bl, exp, ill, inv, oc, ps) // image-mods elaborated bottom of code 
        for line in lines: #deals with one line at a time from list 'lines'
            line = line.rstrip() #removes empty spaces
            if new_im: #records the number of faces (from the line after the '#' line)
                self.num_faces.append(line)
                new_im = False
            elif line.startswith('#'):
                new_im = True
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy() #shallow-copies list
                    self.words.append(labels_copy)
                    labels.clear() #clears list
                path = line[1:] #???? the folder name contains the thing we are taking out.....!!! (error, removed)
                # path = line #???? the folder name contains the thing we are taking out.....!!! (error-replaced)
                path = txt_path.replace('label.txt','images/') + path #saves image path
                self.imgs_path.append(path)
            else:
                line = line.split(' ') #list of numbers (that are separated by space) in the line
                # label = [float(x) for x in line] #???? the values are integers, float typecast not needed (error, removed)
                label = [x for x in line] #???? the values are integers, float typecast not needed (error, replaced)
                labels.append(label) #list of lists of annotation data for the many faces in the image

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path) #number of images

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index]) #image at index
        height, width, _ = img.shape

        labels = self.words[index] #list of 10 annotation criteria for the image
        annotations = np.zeros((0, 10)) #initialize annotation array with 10 columns and 0 rows
        if len(labels) == 0:
            print("\n\nNO ANNOTATIONS FOUND!\n\n")
            return annotations
        for label in labels:
            annotation = np.zeros((1, 10)) #initialization of 1 x 10 numpy array
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            # image_mods
            annotation[0, 4] = label[4]    # blur
            annotation[0, 5] = label[5]    # expression
            annotation[0, 6] = label[6]    # illumination
            annotation[0, 7] = label[7]    # occlusion
            annotation[0, 8] = label[9]    # pose
            annotation[0, 9] = label[9]    # invalid

            annotations = np.append(annotations, annotation, axis=0) #add a row to annoation array. Shape => (self.num_faces x 10)
        target = np.array(annotations) #shouldn't change any dimension; just a copy essentially
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return (torch.from_numpy(img), target) #consider using "img = torch.from_numpy(img).float().to(device)" instead


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    # batch = tuple(batch) #forceful typecasting, (error, added)
    # print("BATCH TYPE: ", type(batch))
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup): #checks is tup is a pytorch tensor
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

# Image-modification information (Included for clarification)
# blur:
#   clear->0
#   normal blur->1
#   heavy blur->2

# expression:
#   typical expression->0
#   exaggerate expression->1

# illumination:
#   normal illumination->0
#   extreme illumination->1

# occlusion:
#   no occlusion->0
#   partial occlusion->1
#   heavy occlusion->2

# pose:
#   typical pose->0
#   atypical pose->1

# invalid:
#   false->0(valid image)
#   true->1(invalid image)
