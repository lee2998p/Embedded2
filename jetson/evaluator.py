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

        '''
        This class evaluates face detection and goggle classsification performance.
        Goggle Classification accuracy is given by average class accuracy and individual
        video accuracy.
        Face detection accuracy is give by precision and recall values.

        Parameters:
        cuda: A bool value that specifies if cuda shall be used
        detector: A string path to a .pth weights file for a face detection model
        classifier: A string path to a .pth weights file for a goggle classsification model
        input_directory: Directory containing test videos to run Evaluator on
        '''

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
        self.results = {'Goggles':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos' : 0,
                             'individual_video_results': {}
                            },
                        'Glasses':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos' : 0,
                             'individual_video_results': {}
                            },
                        'Neither':
                            {'average_class_accuracy': 0.0,
                             'number_of_videos' : 0,
                             'individual_video_results': {}
                            }
                       }
        self.class_label = ''
        self.condition = ''
        self.cap = ''

        self.evaluate()


    def evaluate(self):
        '''
        This method evaluates every video file in the input directory containing test videos.
        It stores all the results in a dict called self.results as it calls the record_results method.
        To understand the format of self.results dict, check the class constructor
        '''
        total_videos_processed = 0

        for video_file in self.video_filenames:
            self.class_label = self.get_class_label(video_file)
            self.condition = self.get_condition(video_file)
            self.cap = cv2.VideoCapture(video_file)
            if self.cap.isOpened():
                result = self.evaluate_classifications()
                self.record_results(result, video_file)
                total_videos_processed += 1
                print (f"{video_file} : Done")

            else:
                print (f"Unable to open video {video_file}")
                continue

        self.calculate_average_class_accuracy()
        print (f"\n {total_videos_processed} videos processed!")


    def calculate_average_class_accuracy(self):
        '''
        This method calculates the average class accuracy for each class and stores it in the
        self.results dict.
        '''
        for class_label in self.results:
            self.results[class_label]['average_class_accuracy'] = self.results[class_label]['average_class_accuracy'] / self.results[class_label]['number_of_videos']

    def record_results(self, result, video_file):
        '''
        This method records all the results in the self.results dict
        '''
        self.results[self.class_label]['number_of_videos'] += 1
        self.results[self.class_label]['average_class_accuracy'] += result[0]
        self.results[self.class_label]['individual_video_results'][video_file] = {}
        self.results[self.class_label]['individual_video_results'][video_file]["accuracy"] = result[0]
        self.results[self.class_label]['individual_video_results'][video_file]["inference_time"] = result[0]
        self.results[self.class_label]['individual_video_results'][video_file]["condition"] = self.condition


    def infer(self):
        '''
        This method Performs inference on a video (frame by frame) by using the face detection
        and goggle classification models
        It returns:
        1) inference_dict contains the number of inferences for each class.
        2) average inference time is a float containing the average inference time for the whole video
        '''

        inference_dict = {"Goggles": 0, "Glasses": 0, "Neither": 0}
        frame_counter = 0
        start_time = time.time()

        while True:
            ret, img = self.cap.read()

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
        '''
        Get class label [Goggles / Glasses / Neither] that the image belongs to
        '''

        class_label = ''
        if '/Goggles/' in filename or '/goggles/' in filename:
            class_label = 'Goggles'
        elif '/Glasses/' in filename or '/glasses/' in filename:
            class_label = 'Glasses'
        else:
            class_label = 'Neither'

        return class_label

    def get_condition(self, filename):
        '''
        Get condition [Ideal, low_lighting etc. ] that the image belongs to
        '''
        return (filename.split('/')[-2])


    def evaluate_classifications(self):
        '''
        This method returns the accuracy (percentage_of_correct_predictions) of the
        predictions for a video
        '''
        inferences, inference_time = self.infer()
        percentage_of_correct_predictions = inferences[self.class_label] / sum(inferences.values())


        return percentage_of_correct_predictions, inference_time


    def evaluate_detections(self, boxes, ground_truth):
        '''
        This method calculates the recall and precision of face detection for a video
        '''
        pass


    def get_video_files(self, input_directory):
        '''
        This method gets all the video files in the input directory
        '''

        filenames = []
        for dirName, subdirList, fileList in os.walk(input_directory):
            for filename in fileList:
                ext = '.' + filename.split('.')[-1]
                if ext in ['.mov','.mp4','.avi']:
                    filenames.append(dirName + '/' + filename)

        return filenames

    def get_evaluator_results(self):
        '''
        This method returns the dict containing all the test results (self.results)
        '''

        return self.results

def main():
    evaluator = Evaluator(args.cuda, args.detector, args.classifier, args.input_directory)
    individual_video_results = evaluator.get_evaluator_results()


    with open(args.output_file, 'w') as json_file:
        json.dump(individual_video_results, json_file, indent=4)

    print (f"\n Output saved at {args.output_file}")

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
