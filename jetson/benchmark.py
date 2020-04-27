import argparse
import statistics
import time
import warnings

import cv2
import torch

from .face_detector import FaceDetector, classify

from .CodeAES import Encryptor

if __name__ == "__main__":
    warnings.filters('once')
    parser = argparse.ArgumentParser(description="Benchmarking goggle detection speed")
    parser.add_argument('--detector', '-d', type=str, required=True)
    parser.add_argument('--classifier', '-c', type=str, required=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--files', '-f', required=True, nargs='+')
    parser.add_argument('--cropped', default=False)
    parser.add_argument('--encryption', '-e', default=False)
    args = parser.parse_args()

    detector = FaceDetector(trained_model=args.detector, cuda=args.cuda and torch.cuda.is_available(),
                            set_default_dev=True)

    cap = cv2.VideoCapture(0)

    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    goggle_classifier = torch.load(args.classifier, map_location=device)
    goggle_classifier.eval()

    detector_times = []
    classifier_times = []
    overall_times = []
    if args.encryption:
        encryption_times = []
        encryptor = Encryptor()

    for video_file in args.files:
        video = cv2.VideoCapture(video_file)
        ret, frame = video.read()
        while ret:
            detector_start_time = time.time()
            boxes = detector.detect(frame)
            detector_end_time = time.time()
            detector_times.append(detector_end_time - detector_start_time)
            if args.encryption:
                encryption_start = time.time()
                encryptor.encrypt(boxes, frame)
                encryption_end = time.time()
                encryption_times.append(encryption_end - encryption_start)
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2, :]
                if args.cropped:
                    height = face.shape[0]
                    face = face[round(0.15 * height):round(0.6 * height), :, :]
                classifier_start_time = time.time()
                label, softlabels = classify(face, goggle_classifier)
                classifier_end_time = time.time()
                classifier_times.append(classifier_end_time - classifier_start_time)
            overall_end = time.time()
            overall_times.append(overall_end - detector_start_time)

    print(f"Face detection time per frame mean {sum(detector_times) / len(detector_times)}s "
          f"+/- std. dev. {statistics.stdev(detector_times)}s\n"
          f"range {min(detector_times)}s - {max(detector_times)}s")

    print(f"Goggle classification time per frame mean: {sum(classifier_times) / len(classifier_times)}s "
          f"+/- std. dev. {statistics.stdev(classifier_times)}s\n"
          f"range {min(classifier_times)}s - {max(classifier_times)}s")

    if args.encryption:
        print(f'Encryption processing time per frame mean: {sum(encryption_times) / len(encryption_times)}s '
              f'+/- std. dev {statistics.stdev(encryption_times)}s\n'
              f'range {min(encryption_times)}s - {max(encryption_times)}s')

    print(f'Overall processing time per frame mean: {sum(overall_times) / len(overall_times)}s '
          f'+/- std. dev {statistics.stdev(overall_times)}s\n'
          f'range {min(overall_times)}s - {max(overall_times)}s')
