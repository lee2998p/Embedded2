from typing import Any, T_co

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SimpleCNN(nn.Module):
    def __init__(self, n_feat, n_class=4):
        super().__init__()
        self.layers = nn.Sequential(self.conv_block(3, 32, kernel_size=5), self.conv_block(n_feat, 32, kernel_size=5, padding=1),
                                    self.conv_block(32, 64, kernel_size=3, padding=1))
        self.fc = nn.Sequential(nn.Linear(565504, 1024), nn.Sigmoid(), nn.Linear(1024, n_class))

    def conv_block(self, in_d, out_d, *args, **kwargs):
        return nn.Sequential(nn.Conv2d(in_d, out_d, *args, **kwargs), nn.BatchNorm2d(out_d), nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
