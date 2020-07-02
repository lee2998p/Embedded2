[![Build Status](https://travis-ci.com/PurdueCAM2Project/Embedded2.svg?branch=master)](https://travis-ci.com/PurdueCAM2Project/Embedded2)

# Embedded Computer Vision 2
System is used to detect usage of Personal Protection Equipment (PPE), specifically goggles, in labs that require them. System is ran on Jetson Nano and uses a Rasspberry Pi camera to record footage in the lab. To ensure individual privacy is protected, system obfuscates faces after detection and classification. Images are stored in a remote storage drive and image metadata are stored on a SQL database.

#### Features
* Retinanet based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Image metadata is stored on a SQL database server
* Images are stored on a remote computer
# Table of Contents
- [Description](#Embedded-Computer-Vision-2)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Usage](#Contributing)
- [Credits](#Credits)
- [License](#License)
# Installation
First step:
'''shell
$ git clone https://github.com/PurdueCAM2Project/Embedded2.git
$ cd Embedded2/src/jetson
'''
The classifier model will be any other .pth file stored on the [Drive](https://drive.google.com/drive/u/1/folders/1ZeKVygo-RyIDL_EnxeYJR8tk-xqzgi3Z).
# Usage
# Contributing
# Credits
# License
