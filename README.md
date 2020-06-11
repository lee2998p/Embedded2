[![Build Status](https://travis-ci.com/PurdueCAM2Project/Embedded2.svg?branch=master)](https://travis-ci.com/PurdueCAM2Project/Embedded2)

# Embedded Computer Vision 2
## Personal Protective Equipment (PPE) Usage Detection
#### The system obfuscates faces after detection and classification to protect individual privacy
#### Features:
* Mobilenet-v2 based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Runs on Jetson Nano in real time

#### Primary scripts:
`jetson/goggles/goggleClassifier.py --directory=path/to/imagefolder`
`jetson/face_detector.py --trained_model=path/to/ssd_model.pth --classifier=path/to/trained_classifier.pth --cuda`
* goggleClassifier.py is how we train our goggle classifier. The model is saved into a .pth file that is loaded as the trained_model of face_detector.py. face_detector.py will detect your face and classify whether you are wearing goggles, glasses, or neither.
* The image folder should be in Pytorch [Imagefolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder) structure.
* Only include --cuda with face_detector if you have a GPU
* We have been using ssd300_WIDER_100455.pth as the SSD model. The classifier model will be any other .pth file stored on the [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z).
* We recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for this project. Once you have it installed you can run `conda env create -f environment.yml` from the Embedded2 directory for the necessary packages.
