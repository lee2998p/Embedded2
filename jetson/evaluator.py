import os
import cv2
import argparse
import torch
import time
import warnings
import json


from face_detector import FaceDetector, classify


class Evaluator():
    def __init__(self, cuda, detector, classifier, input_directory):
        if cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda:0')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.device = torch.device('cpu')

        self.detector = FaceDetector(trained_model=detector, cuda=cuda and torch.cuda.is_available(),
                                set_default_dev=True)
        self.classifier = torch.load(classifier, map_location=self.device)
        self.classifier.eval()
        self.video_filenames = self.get_video_files(input_directory)
        self.results = {}
        #self.results.from_list(self.video_filenames)


        for video_file in self.video_filenames:
            self.class_label = self.get_class_label(self.video_filenames)
            self.cap = cv2.VideoCapture(video_file)
            if self.cap.isOpened():
                results = self.evaluate_classifications(self.cap, self.class_label)
                self.results[video_file] = {}
                self.results[video_file]["Accuracy"] = results[0]
                self.results[video_file]["Inference Time"] = results[1]
            else:
                print (f"Unable to open video {video_file}")
                continue

    def infer(self, video_capture):
        inference_dict = {"Goggles": 0, "Glasses": 0, "Neither": 0}
        frame_counter = 0
        start_time = time.time()

        while True:
            ret, img = video_capture.read()

            if not ret:
                break

            boxes = self.detector.detect(img)
            preds = []
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                face = img[y1:y2, x1:x2, :]
                label, softlabels = classify(face, self.classifier, self.device)

                preds.append(label.item())

                inference_dict["Goggles"] += preds.count(1)
                inference_dict["Glasses"] += preds.count(0)
                inference_dict["Neither"] += preds.count(2)

            frame_counter += 1

        total_time = time.time() - start_time
        average_inference_time = total_time / frame_counter



        return inference_dict, average_inference_time


    def get_class_label(self, filename):
        ''' Get class label [Goggles / Glasses / Neither] that the image belongs to '''

        class_label = ''
        if '/Goggles/' in filename or '/goggles/' in filename:
            class_label = 'Goggles'
        elif '/Glasses/' in filename or '/glasses/' in filename:
            class_label = 'Glasses'
        else:
            class_label = 'Neither'

        return class_label



    def evaluate_classifications(self, video_capture, class_label):
        inferences, inference_time = self.infer(video_capture)

        percentage_of_correct_predictions = inferences[class_label] / sum(inferences.values())

        return percentage_of_correct_predictions, inference_time


    def evaluate_detections(self, boxes):

        pass


    def get_video_files(self, directory):
        filenames = []
        for dirName, subdirList, fileList in os.walk(directory):
            for filename in fileList:
                ext = '.' + filename.split('.')[-1]
                if ext in ['.mov','.mp4','.avi']:
                    filenames.append(dirName + '/' + filename)

        return filenames


def main():
    evaluator = Evaluator(args.cuda, args.detector, args.classifier, args.input_directory)
    print (evaluator.results)

    with open(args.output_file, 'w') as json_file:
        json.dump(evaluator.results, json_file, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-t', type=str, default='model_weights/blazeface.pth', help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', default='model_weights/ensemble_100epochs.pth', type=str, help="Path to a trained classifier .pth file")
    parser.add_argument('--output_file', type=str, default='test_results/test1.json', help="Path to a directory to store evaluation log")
    parser.add_argument('--input_directory', type=str, help="Path to a directory containing video files")

    args = parser.parse_args()

    main()

    exit()
