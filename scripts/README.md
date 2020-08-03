Description of files in the 'scripts' folder and the functions used in the files

# automatic_notification.py
The file uses Cron.schedule() to run the send_email function every day.
#### send_email
The function first retrieves the current timestamp. Then, it forms an email message consisting of information about the sender and the receiver, the email's subject, body messages and timestamp. It connects to a server in order to send the mail. After the mail is sent, the server session is quit.

# face_extractor.py
The file parses an argument of input directory, output directory, trained model, images, rate and horizontal orientation. It calls get_images or get_Videos to get the files. Then, it calls crop_faces_from_images or crop_faces_from_videos to crop and save the face images.
#### get_images
The function gets filenames of images from the input directory and returns a list of the image filenames.
#### get_Videos
The function gets filenames of videos from the input directory and returns a list of the video filenames.
#### crop_and_save_img
The function runs the frame through FaceDetector to create a bounding box around the face and crop the image. Then, it saves the cropped face image.
#### crop_faces_from_images
The function iterates through the image files and crops and saves face images.
#### crop_faces_from_videos
The function first iterates through the video files. For each video, the function iterates through the video frames and crops and saves face images. If the video is shot horizontally, the function flips it so that it's in the right orientation.

# goggle_classifier.py
The file first parses the argument for training the Mobilenet classifier. It then initializes the TensorBoard writer. It other loads a pretrained model or calls get_model to use a pretrained Mobilenet model with layers frozen. Then, it performs training and validation augmentations and then outputs the results from the training.
### MapDataset
This class contains custom dataset for applying different transforms to training and validation data. It consists of the __init__, __getitem__ and __len__ functions.
#### __init__
This function initializes variables regarding the dataset.
#### __getitem__
This function gets the item and returns the mapping of the item in the dataset.
#### __len__
This function returns the number of elements in the dataset.

#### classifier_transforms
Data augmentation options used to train and validate the classifier

#### get_model
This function first initializes Mobilenet and freezes relevant layers. Then, it returns the pretrained Mobilenet model with the relevant layers frozen.
#### load_data
This function first loads the image data from the data location specified. Then, it uses MapDataset to perform data augmentations, creating variables regarding 'train' and 'val'. Fianlly, it returns these new variables: 'train' and 'val' Dataloader, sizes of 'train' and 'val' datasets and the names of dataset classes
#### train_model
This function first initializes hyperparemeters used to train the model. For each epoch, it trains and validates the data. As it iterates through the data, it goes through forward propagation unless the phase is 'train'; in this case, it will go through backward propagation. It prints the loss and accuracy of train or val. There are checkpoints every 10 epochs; these checkpoints help us make the comparison among the trained model and the overfit and underfit models. Finally, it returns the trained model.
#### get_metrics
It prints statistics from the final epoch of training.
