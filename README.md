[![Build Status](https://travis-ci.com/PurdueCAM2Project/Embedded2.svg?branch=master)](https://travis-ci.com/PurdueCAM2Project/Embedded2)

# Embedded Computer Vision 2
The system is used to detect the usage of Personal Protection Equipment (PPE), specifically goggles, in labs that require them. The system is run in real time on Jetson Nano and uses a Rasspberry Pi camera to record footage in the lab. To ensure individual privacy is protected, the system obfuscates faces after detection and classification. Images are stored in a remote storage drive and image metadata are stored on a SQL database.

### Features:
* Retinaface based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Image metadata is stored on a SQL database server
* Images are transfered to remote computer using SFTP
* Runs on Jetson Nano in real time

# Table of Contents
- [Installation](#Installation)
- [Usage](#Contributing)
- [Credits](#Credits)
- [Builds](#Builds)
- [License](#License)

## Installation
1. Clone the project and enter the folder 
```shell
$ git clone https://github.com/PurdueCAM2Project/Embedded2.git
$ cd Embedded2
```
2. The classiifier model (.pth file) can be found on [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z). Downloadand place it in the ```Embedded2/src/jetson``` directory.
3. There is a requirement.txt file with all the necessary dependencies. We, however, recommend using Conda for this project. Once you have conda installed, run the following command to setup the enviroment with necessary dependicies.
```shell
$ conda env create -f environment.yml
```
4. Add the Embedded2 folder to PYTHONPATH by adding the following line in your .bashrc file:
```export PYTHONPATH=/path/Embedded2```
## Usage:
1. We recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for this project. Once you have it installed, you can run `conda env create -f environment.yml` from the Embedded2 directory for the necessary packages.

2. Make sure that the image folder is in Pytorch [Imagefolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder) structure.

3. Run the following script.

`scripts/goggle_classifier.py --directory=path/to/imagefolder`
`scripts/face_extractor.py --trained_model=path/to/ssd_model.pth --classifier=path/to/trained_classifier.pth --cuda`
* goggle_classifier.py trains our goggle classifier. The model is saved into a .pth file that is loaded as the trained_model of face_extractor.py. 
* face_extractor.py detects the face and classifies whether the person is wearing goggles, glasses, or neither.
* We have been using ssd300_WIDER_100455.pth as the SSD model. The classifier model will be any other .pth file stored on the [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z).
* Only include --cuda with face_detector if you have a GPU

 4. The image is sent to one of the three types of detector: blazeface, retinaface or ssd. Make sure that cuda is enabled and calssifier is activiated. The encrypted images are outputted after detection and classification.

## Credits:
* [Crontabs](https://github.com/robdmc/crontabs)

## Builds:
* Travis CI

## License: