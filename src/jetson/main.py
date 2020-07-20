import argparse
import os
import time
import datetime
import warnings
import json
from multiprocessing import Process, Queue, Value

import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms

from src.jetson.face_detector import FaceDetector 
from src.jetson.video_capturer import VideoCapturer
from src.jetson.classifier import Classifier
from src.jetson.encryptor import Encryptor
from src.db import data_insertion

fileCount = Value('i', 0)
encryptRet = Queue()  # Shared memory queue to allow child encryption process to return to parent
DETECTOR_TYPES = ['blazeface', 'retinaface', 'ssd']

def writeImg(img, output_dir, write_imgs):
    """
    This method is used to write an image to an output directory
    Args:
        img: A 3D numpy array containing image to be written
        output_dir: directory to be written to
    Ret:
        face_file_name: os path to written file
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    global fileCount
    face_file_name = os.path.join(output_dir, f'{fileCount.value}.jpg')

    if write_imgs:
        cv2.imwrite(face_file_name, img)
    with fileCount.get_lock():
        fileCount.value += 1

    return face_file_name


def encryptWorker(encryptor, img, boxes, output_dir, write_imgs):
    """
    This method is intended to be spawned as a separate process to handle encrypting and writing of individual frames
    Args:
        encryptor: an encryptor object that contains an AES encryptor object and decryption key
        img: A 3D numpy array containing an image to be enrypted and written
        boxes: facial Coordinates
        output_dir: directory to be written to
    """
    encryptedImg, init_vec_list = encryptor.encryptFrame(img, boxes)
    if write_imgs:
        writtenImg = writeImg(encryptedImg, output_dir, write_imgs)
        encryptRet.put([writtenImg, init_vec_list])


def drawFrame(boxes, frame, fps):
    """
    This method is used to draw the video detection frame viewable by the user
    Args:
        boxes: facial Coordinates
        frame: current frame from video capturer being processed
        fps: frames per second the detector is capable of detecting, classifying, and encrypting
    """
    class_names = ['Glasses', 'Goggles', 'Neither']
    index = 0
    for box in boxes:
        frame = cv2.putText(frame,
                            'label: %s' % class_names[label[index]],
                            (int(box[0]), int(box[1] - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255))

        frame = cv2.putText(frame,
                            'fps: %.3f' % fps,
                            (int(box[0]), int(box[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255))

        index += 1

    cv2.imshow("Face Detect", frame)


if __name__ == "__main__":
    warnings.filterwarnings("once")

    with open('src/jetson/config.json') as file:
        args = json.load(file)

    detector = args["DETECTOR"]
    detector_type = args["DETECTOR_TYPE"]
    cuda = args["CUDA"]
    classifier = args["CLASSIFIER"]
    write_imgs = args["WRITE_IMGS"]
    output_dir = args["OUTPUT_DIR"]
    gstreamer = args["GSTREAMER"]

    if detector_type not in DETECTOR_TYPES:
        print('Please include a valid detector type (\'blazeface\', \'ssd\', or \'retinaface\'')
        exit(1)

    device = torch.device('cpu')
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    g = torch.load(classifier, map_location=device)
    g.eval()

    capturer = VideoCapturer(gstreamer)
    detector = FaceDetector(detector=detector, detector_type=detector_type,
                            cuda=cuda and torch.cuda.is_available(), set_default_dev=True)
    classifier = Classifier(g, cuda)
    encryptor = Encryptor()

    run_face_detection: bool = True
    while run_face_detection:  # main video detection loop that will iterate until ESC key is entered
        start_time = time.time()
        image_date = datetime.date.today()
        image_time = datetime.time()
        frame = capturer.get_frame()
        boxes = detector.detect(frame)
        encryptedImg = frame.copy()  # copy memory for encrypting image separate from unencrypted image

        if len(boxes) != 0:
            p1 = Process(target=encryptWorker, args=(encryptor, encryptedImg, boxes, output_dir, write_imgs))
            p1.daemon = True
            p1.start()

            label = classifier.classifyFrame(frame, boxes)

            if write_imgs:
                image_name, init_vec_list = encryptRet.get()
                data_insertion.data_insert(image_name, image_date, image_time, init_vec_list, boxes, output_dir)

            fps = 1 / (time.time() - start_time)
            drawFrame(boxes, frame, fps)

            # remove frame creation and drawing before deployment

            p1.join()
            if cv2.waitKey(1) == 27:
                run_face_detection = False

    capturer.close()
    cv2.destroyAllWindows()
    exit(0)
