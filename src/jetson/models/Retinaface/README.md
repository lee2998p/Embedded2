# net.py
This file describes functions about returning convolutional layers.
## conv_bn
## conv_bn_no_relu
## conv_bn1X1
## conv_dw
## SSH
### __init__
### forward
## FPN
## MobileNetV1


# retinaface.py
This file describes functions about making layers describing facial images.
## ClassHead
## BboxHead
## LandmarkHead
## RetinaFace
## load_model

# data
This folder consists of __init__.py and config.py. 

__init__.py imports the configurations from the models.Retinaface class. 

config.py defines a library of configurations, including inference configuration and MobileNetV1 and Resnet50 backbone configurations for training.

# layers

## functions
### prior_box.py
This file defines a class PriorBox. The class consists of __init__ and forward functions.
#### __init__
This function initializes the variables of priorbox according to the configuration of the training models. Furthermore, for each source feature map, it calculates the priorbox coordinates.
#### forward
This function first loops through the feature maps to compute the anchor using dense face localizations, min_size and image_size. Then, it passes the anchors to torch land and returns the forward pass of the prior box tensor.

## modules
### multibox_loss.py
This file defines a class MultiBoxLoss. The class consists of __init__ and forward functions.
