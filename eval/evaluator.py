import os
import cv2
import argparse
import torch
import time
import warnings
import json
import numpy as np


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


        if os.path.exists("test_results/det_results_ideal.txt"):
            os.remove("test_results/det_results_ideal.txt")


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
        self.video = ''

        self.evaluate()


    def evaluate(self):
        '''
        This method evaluates every video file in the input directory containing test videos.
        It stores all the results in a dict called self.results as it calls the record_results method.
        To understand the format of self.results dict, check the class constructor
        '''
        total_videos_processed = 0
        for video_file in self.video_filenames:
            self.video = video_file
            print (f"Processing {self.video} ...")


            self.class_label = self.get_class_label()
            self.condition = self.get_condition()
            self.cap = cv2.VideoCapture(self.video)
            if self.cap.isOpened():
                classification_result = self.evaluate_classifications() #Also contains boxes
                self.record_results(classification_result)
                total_videos_processed += 1
                print (f"{self.video} : Done")

            else:
                print (f"Unable to open video {self.video}")
                continue
        self.calculate_average_class_accuracy()
        #self.record_detections('test_results/det_output.txt')
        detection_results = self.evaluate_detections('ground_truth_detections_ideal/',"test_results/det_results_ideal.txt")


        print (f"\n {total_videos_processed} videos processed!")


    def calculate_average_class_accuracy(self):
        '''
        This method calculates the average class accuracy for each class and stores it in the
        self.results dict.
        '''
        for class_label in self.results:
            if self.results[class_label]['number_of_videos'] > 0:
                self.results[class_label]['average_class_accuracy'] = self.results[class_label]['average_class_accuracy'] / self.results[class_label]['number_of_videos']

    def record_results(self, result):
        '''
        This method records all the results in the self.results dict
        '''
        self.results[self.class_label]['number_of_videos'] += 1
        self.results[self.class_label]['average_class_accuracy'] += result[0]
        self.results[self.class_label]['individual_video_results'][self.video] = {}
        self.results[self.class_label]['individual_video_results'][self.video]["accuracy"] = result[0]
        self.results[self.class_label]['individual_video_results'][self.video]["inference_time"] = result[1]
        self.results[self.class_label]['individual_video_results'][self.video]["condition"] = self.condition

    def record_detections(self, file, detections):
        f = open(file, "a+")
        for detection in detections:
            for element in detection:
                f.write(str(element))
                f.write("|")
            f.write("\n")
        f.close()


    def infer(self):
        '''
        This method Performs inference on a video (frame by frame) by using the face detection
        and goggle classification models
        It returns:
        1) inference_dict contains the number of inferences for each class.
        2) average inference time is a float containing the average inference time for the whole video
        '''
        bboxes = []
        inference_dict = {"Goggles": 0, "Glasses": 0, "Neither": 0}
        frame_counter = 0
        start_time = time.time()

        while True:
            ret, img = self.cap.read()

            if not ret:
                break
            frame_id = self.video.strip('.avi').strip('.mp4').strip('.MOV').strip('.mov').split('/')[-1] + "_" + str(frame_counter)
            boxes = self.detector.detect(img)  #Also contains confidence
            preds = []
            for box in boxes:
                x1, y1, x2, y2, conf = [b for b in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                if isinstance(conf, torch.Tensor): #This is true for ssd
                    conf = conf.numpy()
                    #print (conf)

                #assert isinstance(conf, np.float32)


                face = img[int(y1):int(y2), int(x1):int(x2), :]
                label, softlabels = classify(face, self.classifier, self.device)

                preds.append(label.item())

                inference_dict["Goggles"] += preds.count(1)
                inference_dict["Glasses"] += preds.count(0)
                inference_dict["Neither"] += preds.count(2)

                bboxes.append([frame_id, x1, y1, x2, y2, conf])

            frame_counter += 1

        total_time = time.time() - start_time
        if frame_counter > 0:
            average_inference_time = total_time / frame_counter
        else:
            average_inference_time = -1 #Empty video file

        self.record_detections("test_results/det_results_ideal.txt", bboxes)
        return inference_dict, average_inference_time


    def get_class_label(self):
        '''
        Get class label [Goggles / Glasses / Neither] that the image belongs to
        '''

        class_label = ''
        if '/Goggles/' in self.video or '/goggles/' in self.video:
            class_label = 'Goggles'
        elif '/Glasses/' in self.video or '/glasses/' in self.video:
            class_label = 'Glasses'
        else:
            class_label = 'Neither'

        return class_label

    def get_condition(self):
        '''
        Get condition [Ideal, low_lighting etc. ] that the image belongs to
        '''
        return (self.video.split('/')[-2])


    def evaluate_classifications(self):
        '''
        This method returns the accuracy (percentage_of_correct_predictions) of the
        predictions for a video
        '''
        inferences, inference_time = self.infer()
        if sum(inferences.values()) == 0:
            percentage_of_correct_predictions = 0
        else:
            percentage_of_correct_predictions = inferences[self.class_label] / sum(inferences.values())


        return percentage_of_correct_predictions, inference_time

    def get_ground_truth_detections(self, directory):
        GT = {}

        for file in os.listdir(directory):
            f = open(directory+file, "r")
            key = file.strip('.txt')
            content = f.readlines()
            f.close()

            content = [list(map(float, x.strip(' \n').split(' '))) for x in content]
            GT[key] = content


        return GT

    def evaluate_detections(self, annotations_location, detection_location, ovthresh = 0.5):
        '''
        This method calculates the recall and precision of face detection for a video
        '''

        GT_detections = self.get_ground_truth_detections(annotations_location)
        with open(detection_location, 'r') as f:
            lines = f.readlines()

        total_GT = 0
        for frame_id in GT_detections:
            total_GT += len(GT_detections[frame_id])

        if any(lines) == 1:
            splitlines = [x.strip().split('|') for x in lines]

            '''
            for x in splitlines:
                if x[0] not in GT_detections:
                    splitlines.remove(x)
            '''

            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[5]) for x in splitlines])
            BB = np.array([[float(z) for z in x[1:5]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            print (nd)
            for d in range(nd):
                try:
                    R = GT_detections[image_ids[d]]
                    bb = BB[d, :].astype(float)
                    ovmax = -np.inf
                    BBGT = np.asarray(R, dtype=np.float32)
                    if BBGT.size > 0:
                        ixmin = np.maximum(BBGT[:, 0], bb[0])
                        iymin = np.maximum(BBGT[:, 1], bb[1])
                        ixmax = np.minimum(BBGT[:, 2], bb[2])
                        iymax = np.minimum(BBGT[:, 3], bb[3])
                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                               (BBGT[:, 2] - BBGT[:, 0]) *
                               (BBGT[:, 3] - BBGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        #jmax = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        tp[d] = 1.
                    else:
                        fp[d] = 1.

                except KeyError:
                    continue

            print ("total_GT: ", total_GT)
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(total_GT)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        print ("precision: ", prec)
        print ("recall: ", rec)

        return prec, rec



    def get_video_files(self, input_directory):
        '''
        This method gets all the video files in the input directory
        '''

        filenames = []
        for dirName, subdirList, fileList in os.walk(input_directory):
            for filename in fileList:
                ext = '.' + filename.split('.')[-1]
                if ext in ['.mov','.mp4','.avi','.MOV']:
                    filenames.append(dirName + '/' + filename)

        return filenames

    def get_evaluator_results(self):
        '''
        This method returns the dict containing all the test results (self.results)
        '''

        return self.results

def main():
    if not args.input_directory:
        raise Exception("Invalid input directory")
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
    parser.add_argument('--annotation_path', type=str, help="Path to annotation files")

    args = parser.parse_args()

    main()

    exit()
