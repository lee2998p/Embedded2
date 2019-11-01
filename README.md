# Embedded2

Mobilenet-v2 Implementation (temporary usage description)

Testing code 1:

Predicts bounding boxes for images in demo file. You would have to specify filenames in the code.

python test_mobilenet_img.py --trained_model 'weights_mobilenet/50epochs-  pretrained-wface.pth' 

Testing Code 2:

Predicts bounding boxes for video camera input. 

python test_mobilenet_img.py --trained_model 'weights_mobilenet/50epochs-  pretrained-wface.pth' 
---
Training Code (on Widerface training dataset):
-> TODO: add XML annotations and Widerface training images
python train_mobilenet.py --ngpu 1 --num_workers 24 --batch_size 32 --pretrained './weights_mobilenet/vgg16_reducedfc.pth' --save_folder './weights_mobilenet/log_save' --max_epoch 300
