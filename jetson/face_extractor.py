import argparse
from glob import glob
import os
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from face_detector_threaded import FaceDetector

warnings.filterwarnings('once')
IMAGE_EXT = ['.jpg', '.JPG', '.png', '.PNG']
VIDEO_EXT = ['.mp4', '.MP4', 'mov', '.MOV']


def get_images():
    """Get filenames of images from the input directory."""
    files = [glob(f"{args.input_dir}/*{e}") for e in IMAGE_EXT]
    return files[0]


def get_videos():
    """Get filenames of videos from the input directory."""
    files = [glob(f"{args.input_dir}/*{e}") for e in VIDEO_EXT]
    return files[0]


# run the frame through FaceDetector and save the face region
def save_cropped_img(frame, file_num):
    """Run frame through FaceDetector and save the cropped face image."""
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


parser = argparse.ArgumentParser(
    description="Extract faces from images or videos.")
parser.add_argument("--input_dir", default="videos", type=str, help="Input directory containing the videos/images.")
parser.add_argument('--output_dir', default='face_imgs', type=str, help="Output directory for the extracted faces.")
parser.add_argument('--trained_model', default='blazeface.pth', type=str, help="Path to the face detector model.")
parser.add_argument('--rate', default=5, type=int, help="Run the network on 1/rate frames in the video.")
parser.add_argument('--cuda', default=False, action='store_true',
                    help='Use CUDA-enabled GPU for face detection.')
parser.add_argument('--horiz', default=False, action='store_true', help='Rotate the video. If your output images '
                                                                        'are all sideways, enable this.')
parser.add_argument('--images', default=False, action='store_true',
                    help='Extract from images instead of videos.')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device('cpu')

face_detector = FaceDetector(trained_model=args.trained_model)

filenames = get_images() if args.images else get_videos()
print("Processing files")

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

file_num = 0

if args.images:
    for i in filenames:
        image = cv2.imread(i)
        save_cropped_img(np.float32(image), file_num)
        file_num += 1
else:
    for video_file in filenames:
        print(f"Opening {video_file}")
        video = cv2.VideoCapture(video_file)
        file_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(file_len)):
            ret, frame = video.read()

            if frame_num % args.rate == 0:
                # If the video is shot horizontally, flip it so it's in the right orientation
                if args.horiz:
                    frame = cv2.transpose(frame)
                save_cropped_img(frame, file_num)
                file_num += 1

        video.release()

exit(0)
