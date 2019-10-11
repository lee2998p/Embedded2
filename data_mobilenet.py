import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

#Added from second pre-processing attempt
from data_augment import Rescale

#Added from RetinaFace GitHub
from wider_helper import WiderFaceDetection, detection_collate

#Added from Widerface for testing
from tqdm import tqdm

def get_loaders(dataroot, val_batch_size, train_batch_size, input_size, workers):
    #parameters for pre-processing function (called as argument in train_data)
    rgb_mean = (104, 117, 123) # bgr order
    img_dim = 640

    #path for training data
    path_train = os.getcwd()
    path_train = os.path.join(path_train, 'dataroot')
    path_train = os.path.join(path_train, 'train')
    path_train = os.path.join(path_train, 'label.txt')

    #path for validation data
    path_val = os.getcwd()
    path_val = os.path.join(path_val, 'dataroot')
    path_val = os.path.join(path_val, 'val')
    path_val = os.path.join(path_val, 'label.txt')

    train_data = WiderFaceDetection(path_train)#, preproc(img_dim, rgb_mean)) #torch dataset object made using WiderDetection class
   
    # #Added recently for dimension matching (pre-processing)
    # ReSc = Rescale(img_dim) 
    # train_data = ReSc(train_data)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=workers, collate_fn=detection_collate) #pin-memory?? (error, removed)
    
    val_data = WiderFaceDetection(path_val)# , preproc(img_dim, rgb_mean)) #torch dataset object made using WiderDetection clas

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True,
                                               num_workers=workers, collate_fn=detection_collate) #pin-memory?? (error, removed)

    return train_loader, val_loader


    #PREVIOUS CODE

                                                   
    # val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'))#, 
    #                                            #transform=get_transform(False, input_size))

    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False, 
    #                                            num_workers=workers, pin_memory=True)

    # train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'))#,
    #                                           # transform=get_transform(input_size=input_size))
    
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
    #                                            num_workers=workers, pin_memory=True)