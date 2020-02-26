from torch import nn
from data import wider_face
import torch.nn.functional as F
import torch
from layers.functions import PriorBox
from torch.autograd import Variable
from layers.functions import Detect

class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.out_channels = oup1

        self.stride = stride
        assert stride in [1, 2]

        self.use_double_block = oup2 is not None
        self.use_pooling = self.stride != 1

        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            # pw-linear
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)

        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )
            self.out_channels = oup2

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)


def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


class BlazeFace(nn.Module):
    """Constructs a BlazeFace model

    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self, phase, num_classes):
        super(BlazeFace, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.conv_1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.blaze_1 = BlazeBlock(24, 24)
        self.blaze_2 = BlazeBlock(24, 24)
        self.blaze_3 = BlazeBlock(24, 48, stride=2)
        self.blaze_4 = BlazeBlock(48, 48)
        self.blaze_5 = BlazeBlock(48, 48)
        self.blaze_6 = BlazeBlock(48, 24, 96, stride=2)
        self.blaze_7 = BlazeBlock(96, 24, 96)
        self.blaze_8 = BlazeBlock(96, 24, 96)
        self.blaze_9 = BlazeBlock(96, 24, 96, stride=2)
        self.blaze_10 = BlazeBlock(96, 24, 96)
        self.blaze_11 = BlazeBlock(96, 24, 96)

        self.apply(initialize)

        self.head = mbox([self.blaze_9, self.blaze_10], [2, 6], 2)

        self.loc = nn.ModuleList(self.head[0])
        self.conf = nn.ModuleList(self.head[1])

        self.cfg = (wider_face)
        # print(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = self.relu(h)
        h = self.blaze_1(h)
        h = self.blaze_2(h)
        h = self.blaze_3(h)
        h = self.blaze_4(h)
        h = self.blaze_5(h)
        h = self.blaze_6(h)
        h = self.blaze_7(h)
        h1 = self.blaze_8(h)
        h = self.blaze_9(h1)

        h = self.blaze_10(h)
        h2 = self.blaze_11(h)

        # @todo: need to cache outputs from each detection layer, not just h(final output)
        # these will be stored in h ( should be a list )

        # @todo: second argument to multibox ([6, 6, 4, 4, 4, 6] is wrong and based on SSD, but
        # I can't seem to find the correct priorbox numbers for multibox

        # @ todo: once these issues are fixed and code works till returning output, training should work

        loc = list()
        conf = list()
        for (x, l, c) in zip([h1, h2], self.loc, self.conf):
            # print(l)
            # print(x)
            # print('l(x) shape:',  l(x).shape)
            # print(f"x shape: {x.shape}")
            print('l(x)', type(l(x)))
            print('l', type(l))
            print('x', type(x))
            print('x shape', x.shape)
            # print('l(x) shape', l(x).shape)

            # print('type self.loc', type(loc))
            print('l(x) permuted shape', l(x).permute(0, 2, 3, 1).shape)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        o = loc[0]
        print('o shape', (o.view(o.size(0 ), -1)).shape)
        loc = torch.cat([o.view(o.size(0 ), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        print ("loc shape:", loc.shape)
        print ("conf shape:", conf.shape)
        print ("conf.size(0):", conf.size(0))
        print ("loc.size(0)", loc.size(0))
        if self.phase == "test":
            print ("In test mode")
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                conf.view(conf.size(0), -1, self.num_classes),  # conf preds
                self.priors # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        # print(output)
        # print(output[0].shape)
        # print(output[1].shape)
        # print(output[2].shape)
        return output



def mbox(layers, cfg, num_classes):
    # @todo: this function should only take in the layers that produce anchor boxes
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(layers):
        # print(type(v))
        print('out channels in mbox loss', v.out_channels)
        print(cfg)
        # @ todo: find right number for 2nd argument to nn.Conv2d (not 6, which is hardcoded)
        # should be number of anchor boxes at that layer, hence takes into account "cfg" argument
        # loc_layers += [nn.Conv2d(v.out_channels,
        #                      cfg[k] * 4, kernel_size=3, padding=1)]
        # conf_layers += [nn.Conv2d(v.out_channels,
        #                       cfg[k] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(v.out_channels,
                             cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                              cfg[k] * 2, kernel_size=3, padding=1)]

    return (loc_layers, conf_layers)
