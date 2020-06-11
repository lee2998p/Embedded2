import argparse
from glob import glob
import hashlib
import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from face_detector import FaceDetector

warnings.filterwarnings('once')


# for videos, saves filenames and hashes
# for images, saves filenames
def get_files(input_dir, hash_file=None, images=False):
    if images:
        img_ext = ['.jpg', '.JPG', '.png', '.PNG']
        imgs = [glob(f"{input_dir}/*{e}", recursive=args.recursive) for e in img_ext]
        images = [img for subimg in imgs for img in subimg]
        return images
    else:
        video_ext = ['.mp4', '.mov', '.MOV', '.MP4']
        videos = [glob(f"{input_dir}/*{e}", recursive=args.recursive) for e in video_ext]
        # Reduce the 2d list from before to a 1d list
        videos = [vid for subvid in videos for vid in subvid]

        hashes = []
        for video in videos:
            hasher = hashlib.md5()
            buf_size = 65536
            with open(video, 'rb') as video_file:
                buf = video_file.read(buf_size)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = video_file.read(buf_size)
            hashes.append(hasher.hexdigest())

        if hash_file is None:
            return list(zip(videos, hashes))
        else:
            with open(hash_file, 'r') as f:
                processed_hashes = [line.strip() for line in f]
            # Filter out the videos that have already been processed
            return [(file, file_hash) for (file, file_hash) in zip(videos, hashes) if file_hash not in processed_hashes]


# run the frame through FaceDetector and save the face region
def save_cropped_img(frame, file_num):
    if frame is not None and not 0:
        boxes = face_detector.detect(frame)
        for box in boxes:
            # Get individual coordinates as integers
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]
            if face is None or 0 in face.shape:
                continue
            face_file_name = os.path.join(args.output_dir, f'{file_num}.jpg')
            cv2.imwrite(face_file_name, face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract faces in data to label at a later time using a neural net trained to detect faces")
    parser.add_argument("--input_dir", default="videos", type=str, help="The directory holding the video")
    parser.add_argument('-R', '--recursive', default=False, type=bool, help="Search the input dir recursively")
    parser.add_argument('--output_dir', default='face_imgs', type=str, help="Where to output the extracted faces")
    parser.add_argument('--hash_file', default=None,
                        help='File containing md5 hashes of videos that have been processed')
    parser.add_argument('--trained_model', default='ssd300_WIDER_100455.pth', type=str, help="Trained state_dict file")
    parser.add_argument('--rate', default=5, type=int, help="Run the network on 1/rate frames in the video")
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='Run the neural network with cuda enabled (will be disabled if cuda isn\'t available')
    parser.add_argument('--images', default=False, action='store_true',
                        help='Run images through the neural net instead of videos')
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:0')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')

    face_detector = FaceDetector(trained_model=args.trained_model)

    # This lets us break the generation up into different sessions if we want
    print("Finding files")
    files_and_hashes = get_files(args.input_dir, args.hash_file, args.images)
    print("Processing files")

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    file_num = 0

    if args.images:
        for i in files_and_hashes:
            image = cv2.imread(i)
            save_cropped_img(np.float32(image), file_num)
            file_num += 1
    else:
        hash_file = args.hash_file
        if hash_file is None:
            hash_file = "processed_hashes.txt"
        print(f"FILES: {files_and_hashes}")
        with open(hash_file, "w+") as hash_out:
            for video_file, file_hash in files_and_hashes:
                print(f"Opening {video_file} :: {file_hash}")
                video = cv2.VideoCapture(video_file)
                file_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                for frame_num in tqdm(range(file_len)):
                    ret, frame = video.read()

                    if frame_num % args.rate == 0:
                        # If the video is shot horizontally, flip it so it's in the right orientation
                        # frame = cv2.flip(cv2.transpose(frame), 1)
                        save_cropped_img(frame, file_num)
                        file_num += 1

                video.release()
                hash_out.write(f"{file_hash}\n")

    exit(0)
