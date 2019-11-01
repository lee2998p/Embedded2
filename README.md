# Mobilenet-v2 Implementation (temporary usage description)
## Testing code 1:  (Predicts bbx for demo images)
TODO: Add imgs in demo_mobilenet. Add filenames in code

python test_mobilenet_img.py --trained_model 'weights_mobilenet/50epochs-  pretrained-wface.pth' 

## Testing Code 2:  (Predicts bbx for video cam frames)

python test_mobilenet_img.py --trained_model 'weights_mobilenet/50epochs-  pretrained-wface.pth' 

## Training Code:  (Trains on Widerface training data)

TODO: Add XML annotations (https://github.com/akofman/wider-face-pascal-voc-annotations) and Widerface training images (http://shuoyang1213.me/WIDERFACE/) to data_mobilenet folders

python train_mobilenet.py --ngpu 1 --num_workers 24 --batch_size 32 --pretrained './weights_mobilenet/vgg16_reducedfc.pth' --save_folder './weights_mobilenet/log_save' --max_epoch 300
