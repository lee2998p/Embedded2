# Embedded Computer Vision 2
## Personal Protective Equipment (PPE) Usage Detection
#### The system obfuscates faces after detection and classification to protect individual privacy
#### Features:
* Mobilenet-v2 based SSD performs face detection
* CNN performs classification of detected faces to determine if PPE is being used
* Faces are encrypted using AES
* Runs on Jetson Nano in real time