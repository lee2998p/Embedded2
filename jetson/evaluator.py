import os
import cv2
import argparse
import torch


from face_detector import FaceDetector


class Evaluator():
    def __init__(self):
        self.detector = FaceDetector(trained_model=args.detector, cuda=args.cuda and torch.cuda.is_available(),
                                set_default_dev=True)
        self.classifier = torch.load(args.classifier, map_location=device)
        self.classifier.eval()
        if args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda:0')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.device = torch.device('cpu')

        self.cap = cv2.VideoCapture(0)



    def load_evaluation_data(directory):
        filenames = []
        for dirName, subdirList, fileList in os.walk(directory):
            for filename in fileList:
                ext = '.' + filename.split('.')[-1]
                if ext in ['.mov','.mp4','.avi']:
                    filenames.append(dirName + '/' + filename)

        return filenames


    def evaluate():
        pass


    def record_evaluationa():
        pass


def main():
    pass



if __name__ == "__main__":
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument('--detector', '-t', type=str, required=True, help="Path to a trained ssd .pth file")
    parser.add_argument('--cuda', '-c', default=False, action='store_true', help="Enable cuda")
    parser.add_argument('--classifier', type=str, help="Path to a trained classifier .pth file")

    args = parser.parse_args()

    main()
