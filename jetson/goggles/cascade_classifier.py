import torch
import torch.nn as nn
import torch.functional as F
from jetson.goggles.SimpleClassifier import SimpleCNN


class MultiStageGoggleClassifier(nn.Module):
    def __init__(self):
        super(MultiStageGoggleClassifier, self).__init__()
        self.onFace_classifier = SimpleCNN(n_class=2)
        self.goggle_classifier = SimpleCNN(n_class=2)
        self.saved_features = None
        list(self.onFace_classifier.features[-1].modules())[-1].register_forward_hook(self.face_features_hook)

    def face_features_hook(self, layer, input, output):
        self.saved_features = output

    def forward(self, x):
        outputs = self.onFace_classifier.forward(x)
        _, preds = torch.max(outputs.data, 1)
        # Get the indicies where we have a goggle detection
        idxz = (preds == 1).nonzero().squeeze()
        features = self.saved_features[idxz]
        print(preds)
        print(idxz)
        if 0 not in features.shape:
            print("SECOND CLASSIFIER")
            print(features.shape)
            goggle_preds = self.goggle_classifier(features)
            print("ADDING")
            preds[idxz] += goggle_preds


if __name__ == "__main__":
    msgc = MultiStageGoggleClassifier()
    print(msgc.onFace_classifier)
    x = torch.zeros([1, 3, 100, 100])
    msgc(x)
