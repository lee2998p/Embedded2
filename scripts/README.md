Description of files and the functions used in the files

# automatic_notification.py
The file uses Cron.schedule() to run the send_email function every day.
## send_email
The function first retrieves the current timestamp. Then, it forms an email message consisting of information about the sender and the receiver, the email's subject, body messages and timestamp. It connects to a server in order to send the mail. After the mail is sent, the server session is quit.

# face_extractor.py
The file parses an argument of input directory, output directory, trained model, images, rate and horizontal orientation. It calls get_images or get_Videos to get the files. Then, it calls crop_faces_from_images or crop_faces_from_videos to crop and save the face images.
## get_images
The function gets filenames of images from the input directory and returns a list of the image filenames.
## get_Videos
The function gets filenames of videos from the input directory and returns a list of the video filenames.
## crop_and_save_img
The function runs the frame through FaceDetector to create a bounding box around the face and crop the image. Then, it saves the cropped face image.
## crop_faces_from_images
The function iterates through the image files and crops and saves face images.
## crop_faces_from_videos
The function first iterates through the video files. For each video, the function iterates through the video frames and crops and saves face images. If the video is shot horizontally, the function flips it so that it's in the right orientation.
