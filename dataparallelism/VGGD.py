import time
import torch
from torch import nn as nn
from collections import OrderedDict
import os

class vgg16bn(torch.nn.Module):
    def __init__(self):
        super(vgg16bn, self).__init__()
        self.model = nn.Sequential(OrderedDict([
                        ("block1_conv1", nn.Conv2d(3, 64, 3, padding = 1)),
                        ("block1_conv1_batchnorm", nn.BatchNorm2d(64)),
                        ("block1_conv1_ReLU", nn.ReLU(True)),
                        ("block1_conv2", nn.Conv2d(64, 64, 3, padding = 1)),
                        ("block1_conv2_batchnorm", nn.BatchNorm2d(64)),
                        ("block1_conv2_ReLU", nn.ReLU(True)),
                        ("block1_MaxPool", nn.MaxPool2d(2, 2)),
                        ("block2_conv1", nn.Conv2d(64, 128, 3, padding = 1)),
                        ("block2_conv1_batchnorm", nn.BatchNorm2d(128)),
                        ("block2_conv1_ReLU", nn.ReLU(True)),
                        ("block2_conv2", nn.Conv2d(128, 128, 3, padding = 1)),
                        ("block2_conv2_batchnorm", nn.BatchNorm2d(128)),
                        ("block2_conv2_ReLU", nn.ReLU(True)),
                        ("block2_MaxPool", nn.MaxPool2d(2, 2)),
                        ("block3_conv1", nn.Conv2d(128, 256, 3, padding = 1)),
                        ("block3_conv1_batchnorm", nn.BatchNorm2d(256)),
                        ("block3_conv1_ReLU", nn.ReLU(True)),
                        ("block3_conv2", nn.Conv2d(256, 256, 3, padding = 1)),
                        ("block3_conv2_batchnorm", nn.BatchNorm2d(256)),
                        ("block3_conv2_ReLU", nn.ReLU(True)),
                        ("block3_conv3", nn.Conv2d(256, 256, 3, padding = 1)),
                        ("block3_conv3_batchnorm", nn.BatchNorm2d(256)),
                        ("block3_conv3_ReLU", nn.ReLU(True)),
                        ("block3_MaxPool", nn.MaxPool2d(2, 2)),
                        ("block4_conv1", nn.Conv2d(256, 512, 3, padding = 1)),
                        ("block4_conv1_batchnorm", nn.BatchNorm2d(512)),
                        ("block4_conv1_ReLU", nn.ReLU(True)),
                        ("block4_conv2", nn.Conv2d(512, 512, 3, padding = 1)),
                        ("block4_conv2_batchnorm", nn.BatchNorm2d(512)),
                        ("block4_conv2_ReLU", nn.ReLU(True)),
                        ("block4_conv3", nn.Conv2d(512, 512, 3, padding = 1)),
                        ("block4_conv3_batchnorm", nn.BatchNorm2d(512)),
                        ("block4_conv3_ReLU", nn.ReLU(True)),
                        ("block4_MaxPool", nn.MaxPool2d(2, 2)),
                        ("block5_conv1", nn.Conv2d(512, 512, 3, padding = 1)),
                        ("block5_conv1_batchnorm", nn.BatchNorm2d(512)),
                        ("block5_conv1_ReLU", nn.ReLU(True)),
                        ("block5_conv2", nn.Conv2d(512, 512, 3, padding = 1)),
                        ("block5_conv2_batchnorm", nn.BatchNorm2d(512)),
                        ("block5_conv2_ReLU", nn.ReLU(True)),
                        ("block5_conv3", nn.Conv2d(512, 512, 3, padding = 1)),
                        ("block5_conv3_batchnorm", nn.BatchNorm2d(512)),
                        ("block5_conv3_ReLU", nn.ReLU(True)),
                        ("block5_MaxPool", nn.MaxPool2d(2, 2)),
                        ("AvgPool", nn.AdaptiveAvgPool2d((7, 7))),
                        ("flatten", nn.Flatten()),
                        ("fc1", nn.Linear(512 * 7 * 7, 4096)),
                        ("ReLU", nn.ReLU(True)),
                        ("Dropout", nn.Dropout(0.5)),
                        ("fc2", nn.Linear(4096, 4096)),
                        ("ReLU", nn.ReLU(True)),
                        ("Dropout", nn.Dropout(0.5)),
                        ("prediction", nn.Linear(4096, 1000))
                        ]))

    def forward(self, x):
        x = self.model(x)
        return x
