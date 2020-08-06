## net.py
This file describes functions about returning convolutional layers.
#### conv_bn
This funcion returns one layer of batch normalized convolution layer (filter=3x3) applying relu activation.
#### conv_bn_no_relu
This function returns one layer of batch normalized convolution layer (filter=3x3) without applying relu activation.
#### conv_bn1X1
This function returns one layer of batch normalized convolution layer (filter=1x1) applying relu activation.
#### conv_dw
This function returns layer of batch normalized depthwise convolution layer applying relu activation.
#### SSH
This class defines single stage headless face detector.

The __init__ function initializes batch normalized convolutional layers using the number of input and output channels, the stride and the leakiness.

The forward function receives tensor outputted by backbone. Using this input, it returns output tensor after applying network layers(batch normalized convolutional) and operations.
#### FPN
This class defines feature pyramid network.

The __init__ function initializes batch normalized convolutional layers using the list of input channels, the number of output channels, the stride and the leakiness.

The forward function first initalizes 3 different batch normalized convolutional layers(output1, output2, output3); each initalized layer uses respective input channel from the input channel list. Then, output2 is updated, involving the interpolation of output3 and the merge of output2. Afterward, output1 is updated, involving the interpolation of output2 and the merge of output1. Finally, the forward function returns output1, output2 and output3.
#### MobileNetV1
This class defines MobileNetV1 model used as backbone for face detection.

The __init__ function first initalizes different stages of sequential batch normalized convolution layers and batch normalized depthwise convolution layer. It also initializes operational functions.

The forward function receives input tensor x: x is either a single image tensor or a batch of them. The forward function applied the stagges of network layers and performs operations. Finally, it returns the output tensor after passing through MobileNetV1 backbone.
## retinaface.py
This file describes functions about making layers describing facial images.
#### ClassHead
#### BboxHead
#### LandmarkHead
#### RetinaFace
#### load_model

# data
This folder consists of __init__.py and config.py. 

__init__.py imports the configurations from the models.Retinaface class. 

config.py defines a library of configurations, including inference configuration and MobileNetV1 and Resnet50 backbone configurations for training.

# layers

## functions
### prior_box.py
This file defines a class PriorBox. 

The __init__ function initializes the variables of priorbox according to the configuration of the training models. Furthermore, for each source feature map, it calculates the priorbox coordinates.

The forward function first loops through the feature maps to compute the anchor using dense face localizations, min_size and image_size. Then, it passes the anchors to torch land and returns the forward pass of the prior box tensor.

## modules
### multibox_loss.py
This file defines a class MultiBoxLoss. 

The __init__ function initializes variables related to comparing ground truth boxes and priorboxes.

The forward function computes the SSD weighted loss: the calculations include confidence target indices, localization target and hard negative mining. 
