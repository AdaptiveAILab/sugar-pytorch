"""Frozen model architectures for the golden parity tests.

Verbatim copies of the experiment model definitions as they were at migration
time:

- ``VGG16``          -- experiments/cnn/cnn/vgg.py
- ``resnet18_cifar`` -- experiments/cnn/cnn/resnet18.py

Vendored so the package test suite (and the golden fixtures it regenerates)
does not depend on the experiments tree; the recorded fixtures stay stable
even if the experiment models evolve.  The torchvision import inside
``resnet18_cifar`` is kept local to the function so this module stays
importable with torch alone.
"""

import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_classes=10, bn=True):
        super(VGG16, self).__init__()

        def conv_block(in_ch, out_ch, pool=False, bn=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch) if bn else nn.Identity(),
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*[l for l in layers if not isinstance(l, nn.Identity)])

        self.layer1 = conv_block(3, 64, bn=bn)
        self.layer2 = conv_block(64, 64, pool=True, bn=bn)
        self.layer3 = conv_block(64, 128, bn=bn)
        self.layer4 = conv_block(128, 128, pool=True, bn=bn)
        self.layer5 = conv_block(128, 256, bn=bn)
        self.layer6 = conv_block(256, 256, bn=bn)
        self.layer7 = conv_block(256, 256, pool=True, bn=bn)
        self.layer8 = conv_block(256, 512, bn=bn)
        self.layer9 = conv_block(512, 512, bn=bn)
        self.layer10 = conv_block(512, 512, pool=True, bn=bn)
        self.layer11 = conv_block(512, 512, bn=bn)
        self.layer12 = conv_block(512, 512, bn=bn)
        self.layer13 = conv_block(512, 512, pool=True, bn=bn)
        self.gap = nn.AdaptiveAvgPool2d(1)
        fc_layers = [nn.Dropout(0.5), nn.Linear(512, 512)]
        if bn:
            fc_layers.append(nn.BatchNorm1d(512))
        fc_layers.append(nn.ReLU())
        self.fc1 = nn.Sequential(*fc_layers)
        self.fc2 = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def resnet18_cifar(num_classes=10):
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m
