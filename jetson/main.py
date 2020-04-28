from jetson.goggles.goggleClassifier import GoggleClassifier
from jetson.face_detector import FaceDetector
from jetson.FrameComparator import FrameComparator
import cv2
import argparse
import torch
import time


def run_networks(img, fd: FaceDetector, gc: GoggleClassifier, cropped: bool):
    bboxes = fd.detect(img)
    class_results = []
    for box in bboxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        if cropped:
            y1 = y1 + round(0.15 * y1)
            y2 = y2 - round(0.4 * y2)

        face = img[y1:y2, x1:x2]
        label, softlabels = gc.classify(face)
        class_results.append(label)

    return class_results


if __name__ == "__main__":
    # TODO assign abbreviations where appropriate
    parser = argparse.ArgumentParser(description="Main script for running the whole project")
    parser.add_argument("--cuda", type=bool, default=True, help="Choose to enable GPU acceleration")
    parser.add_argument("--detector_pth", type=str, required=True,
                        help="The location of a pth file for a trained face detector")
    parser.add_argument("--classifier_pth", type=str, required=True,
                        help="The location of a pth file for a trained goggle classifier")
    parser.add_argument("--classifier_name", type=str, required=True, help="The name of the classifier architecture")
    parser.add_argument("--rate", type=float, default=2,
                        help="The target frame rate (should be set lower than maximum throughput to save resources)")  # This should be fps
    parser.add_argument("--detect_thresh", type=float, default=0.75,
                        help="The threshold to consider detection a success")
    parser.add_argument("--cropped", type=bool, default=True,
                        help="Enable tighter cropping of faces for goggle classification")
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), "Cuda is not available at this time"

    assert 0 <= args.detect_thresh <= 1, "Detection threshold must be between 0.0 and 1.0"

    if not args.cuda and torch.cuda.is_available():
        print("Cuda is disabled but available")

    face_detector = FaceDetector(args.detector_pth, detection_threshold=args.detect_thresh, cuda=args.cuda,
                                 set_default_dev=True)

    goggle_classifier = GoggleClassifier(args.classifier_name, args.classifier_pth, train_mode=False,
                                         device=torch.device("cuda:0") if args.cuda else torch.device("cpu"))

    assert goggle_classifier.model is not None, "Goggle classifier model failed to load"

    cap = cv2.VideoCapture(0)
    running = True
    check_face = True  # Start true to get detector to run at least once
    time_delta = 1 / args.rate

    # Need to grab a frame to initialize the frame comparator
    ret, frame = cap.read()
    frame_comp = FrameComparator(frame, change_time=15, change_thresh=45)

    last_nn_run = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
        else:
            if check_face:
                if time.time() - last_nn_run > time_delta:
                    goggle_usage = run_networks(img=frame, fd=face_detector, gc=goggle_classifier, cropped=args.cropped)
                    last_nn_run = time.time()
                    if len(goggle_usage) == 0:
                        check_face = False
                        frame_comp.set_ref_frame(frame)
                    else:
                        # TODO handle classifications
                        for ppe_status in goggle_usage:
                            print(ppe_status)

            else:
                check_face = frame_comp.check_change(frame)

    cap.release()
