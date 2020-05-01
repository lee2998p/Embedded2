# Embedded Computer Vision 2
## Personal Protective Equipment (PPE) Usage Detection
#### The system obfuscates faces after detection and classification to protect individual privacy
#### Features:
* Mobilenet-v2 based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Runs on Jetson Nano in real time

#### Primary scripts:
* jetson/goggles/goggleClassifier.py should be run with --test_mode=True --im=True --directory=path/to/imagefolder
** The image folder should contain a train and val folder, both of which should be in Pytorch [Imagefolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder) structure
* jetson/face_detector.py should be run with --trained_model=path/to/ssd_model.pth --classifier=path/to/trained_classifier.pth --cuda (if you have a GPU)
** We have been using ssd300_WIDER_100455.pth as the SSD model. The classifier model will be any other .pth file stored on the [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z).
