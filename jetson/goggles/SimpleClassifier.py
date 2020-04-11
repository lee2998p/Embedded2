from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F


# Creates a bottleneck layer
def _bn_func_factory(conv, norm, relu):
    def bn_layer(*input):
        concat_feat = torch.cat(input, 1)
        bn_output = relu(norm(conv(concat_feat)))
        return bn_output

    return bn_layer


class _Layer(nn.Module):
    def __init__(self, num_input_feat, growth_rate):
        super(_Layer, self).__init__()
        self.add_module('conv0', nn.Conv2d(num_input_feat, 4 * growth_rate, kernel_size=3, padding=1))
        self.add_module('norm0', nn.BatchNorm2d(4 * growth_rate))
        self.add_module('relu0', nn.ReLU(inplace=True))

        self.add_module('conv1', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1))
        self.add_module('norm1', nn.BatchNorm2d(growth_rate))
        self.bn_func = _bn_func_factory(self.conv0, self.norm0, self.relu0)

    def forward(self, *prev_feats):
        if any(feat.requires_grad for feat in prev_feats):
            bn_out = cp.checkpoint(self.bn_func, *prev_feats)
        else:
            bn_out = self.bn_func(*prev_feats)
        return self.norm1(self.conv1(bn_out))


class _Block(nn.Module):
    def __init__(self, num_chan, growth_rate, block_num):
        super(_Block, self).__init__()
        self.add_module('block' + block_num, _Layer(num_chan, growth_rate))

    def forward(self, init_feat):
        feats = [init_feat]
        for name, layer in self.named_children():
            new_feats = layer(*feats)
            feats.append(new_feats)
        return torch.cat(feats, 1)


class SimpleCNN(nn.Module):
    def __init__(self, growth_rate=4, block_config=2, n_class=4):
        super().__init__()
        n_feat = 32
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, n_feat, kernel_size=3, stride=1, padding=1, bias=True)),
            ('pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=1))]))

        self.features.add_module(f'block{0}', _Block(n_feat, growth_rate * block_config, f'{0}'))
        n_feat += growth_rate * block_config

        self.fc = nn.Linear(n_feat, n_class)

        # Initialize the model
        for m in self.named_parameters():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.weight.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        poold = F.avg_pool2d(out, kernel_size=51)
        out = poold.view(features.size(0), -1)
        out = self.fc(out)
        return out

    def param_count(self):
        return sum(reduce(lambda a, b: a * b, x.size()) for x in self.parameters())


if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
    print(f"Model has {model.param_count()} parameters")
