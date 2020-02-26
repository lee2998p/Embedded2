from data import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



WIDER_CLASSES = ('face', )

#WIDER_ROOT = osp.join(HOME, "/PyTorch_BlazeFace/")
WIDER_ROOT = osp.join("WIDER/")

class WIDERAnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDER_CLASSES, range(len(WIDER_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height

                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            #look here for potential problems
            #from voc0712.
            # x1 = bndbox[0]
            # y1 = bndbox[1]
            # width = bndbox[2]
            # height = bndbox[3]
            #
            # bndbox[0] = x1
            # bndbox[1] = x1 + width
            # bndbox[2] = y1
            # bndbox[3] = y1 + height

            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            #original


            #res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res


class WIDERDetection(data.Dataset):

    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, image_set='wider_train', transform = None, target_transform=WIDERAnnotationTransform(), dataset_name = "WIDER_FACE"):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        #self._annopath = os.path.join('/Users/ishaghodgaonkar/Embedded2/BlazeFace/', 'WIDER/annotations', '%s')
        self._annopath = os.path.join(os.getcwd(), '../WIDER/annotations', '%s')
        if image_set == 'wider_train':
            print("TRAINING DATASET")
            #self._imgpath = os.path.join('/Users/ishaghodgaonkar/Embedded2/BlazeFace/', 'WIDER/WIDER_train/images', '%s')
            self._imgpath = os.path.join(os.getcwd(), '../WIDER/WIDER_train/images', '%s')
        else:
            print("TESTING DATASET")
            #self._imgpath = os.path.join('/Users/ishaghodgaonkar/Embedded2/BlazeFace/',  'WIDER/WIDER_test/images', '%s')
            self._imgpath = os.path.join(os.getcwd(),  '../WIDER/WIDER_test/images', '%s')
        self.ids = list()
        self.full_ids = list()
#        with open(os.path.join('/Users/ishaghodgaonkar/Embedded2/BlazeFace/', 'WIDER/wider_face_split/img_list.txt'), 'r') as f:
        with open(os.path.join(os.getcwd(), '../WIDER/wider_face_split/img_list.txt'), 'r') as f:
          self.ids = [(((tuple(line.split('/')))[1]).split('.'))[0] + '.xml' for line in f]

    #    with open(os.path.join('/Users/ishaghodgaonkar/Embedded2/BlazeFace/', 'WIDER/wider_face_split/img_list.txt'), 'r') as f:
        with open(os.path.join(os.getcwd(), '../WIDER/wider_face_split/img_list.txt'), 'r') as f:
          self.full_ids = [tuple(line.split()) for line in f]

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        img_id = self.ids[index]
        full_id = self.full_ids[index]
        #
        # print(img_id)
        # print(full_id)

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % (full_id[0].split('.')[0] + '.jpg'))
        # print(img)
        height, width, channels = img.shape

        if self.target_transform is not None:

            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width


    def pull_tensor(self, index):
            '''Returns the original image at an index in tensor form
            Note: not using self.__getitem__(), as any transformations passed in
            could mess up this functionality.
            Argument:
                index (int): index of img to show
            Return:
                tensorized version of img, squeezed
            '''
            return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def pull_image(self, index):
            '''Returns the original image object at index in PIL form
            Note: not using self.__getitem__(), as any transformations passed in
            could mess up this functionality.
            Argument:
                index (int): index of img to show
            Return:
                PIL img
            '''
            # print(index)
            # img_id = self.ids[index]
            # full_id = self.full_ids[index]
            # print(self._imgpath % (full_id[0].split('.')[0] + '.jpg'))
            return cv2.imread(index)

    def pull_anno(self, index):
            '''Returns the original annotation of image at index
            Note: not using self.__getitem__(), as any transformations passed in
            could mess up this functionality.
            Argument:
                index (int): index of img to get annotation of
            Return:
                list:  [img_id, [(label, bbox coords),...]]
                    eg: ('001718', [('dog', (96, 13, 438, 332))])
            '''
            img_id = self.ids[index]
            print ("id:", img_id)
            anno = ET.parse(self._annopath % img_id).getroot()
            gt = self.target_transform(anno, 1, 1)
            return img_id, gt


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
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
