import time
import torch
from torch import nn as nn
from collections import OrderedDict
import os

class vgg19bn_imagenet(torch.nn.Module):
    def __init__(self):
        super(vgg19bn_imagenet, self).__init__()
        self.model = nn.Sequential(OrderedDict([
                        ("block1_conv1", nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block1_conv1_batchnorm", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block1_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block1_conv2", nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))),
                        ("block1_conv2_batchnorm", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block1_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block1_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block2_conv1", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block2_conv1_batchnorm", nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block2_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block2_conv2", nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
                        ("block2_conv2_batchnorm", nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block2_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block2_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block3_conv1", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv1_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv2", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv2_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv3", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv3_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv4", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv4_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block3_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block4_conv1", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv1_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv2_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv3_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv4", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv4_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block4_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block5_conv1", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv1_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv2_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv3_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv4", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv4_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block5_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                        
                        #classifier
                        ("AvgPool", nn.AdaptiveAvgPool2d(output_size=(7, 7))),

                        ("flatten", nn.Flatten()),

                        ("fc1", nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True)),
                        ("fc1_ReLU", nn.ReLU(inplace=True)),
                        ("fc1_Dropout", nn.Dropout(p=0.5, inplace=False)),
                        ("fc2", nn.Linear(in_features=4096, out_features=4096, bias=True)),
                        ("fc2_ReLU", nn.ReLU(inplace=True)),
                        ("fc2_Dropout", nn.Dropout(p=0.5, inplace=False)),
                        ("prediction", nn.Linear(in_features=4096, out_features=1000, bias=True))
                        ]))

    def forward(self, x):
        x = self.model(x)
        return x

class vgg19bn_cifar10(torch.nn.Module):
    def __init__(self):
        super(vgg19bn_cifar10, self).__init__()
        self.model = nn.Sequential(OrderedDict([
                        ("block1_conv1", nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block1_conv1_batchnorm", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block1_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block1_conv2", nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))),
                        ("block1_conv2_batchnorm", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block1_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block1_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block2_conv1", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block2_conv1_batchnorm", nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block2_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block2_conv2", nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
                        ("block2_conv2_batchnorm", nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block2_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block2_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block3_conv1", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv1_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv2", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv2_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv3", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv3_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block3_conv4", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block3_conv4_batchnorm", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block3_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block3_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block4_conv1", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv1_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv2_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv3_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block4_conv4", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block4_conv4_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block4_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block4_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                        ("block5_conv1", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv1_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv1_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv2_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv2_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv3_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv3_ReLU", nn.ReLU(inplace=True)),

                        ("block5_conv4", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                        ("block5_conv4_batchnorm", nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                        ("block5_conv4_ReLU", nn.ReLU(inplace=True)),

                        ("block5_MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                        
                        #classifier
                        ("AvgPool", nn.AdaptiveAvgPool2d(output_size=(7, 7))),

                        ("flatten", nn.Flatten()),

                        ("fc1", nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True)),
                        ("fc1_ReLU", nn.ReLU(inplace=True)),
                        ("fc1_Dropout", nn.Dropout(p=0.5, inplace=False)),
                        ("fc2", nn.Linear(in_features=4096, out_features=4096, bias=True)),
                        ("fc2_ReLU", nn.ReLU(inplace=True)),
                        ("fc2_Dropout", nn.Dropout(p=0.5, inplace=False)),
                        ("prediction", nn.Linear(in_features=4096, out_features=10, bias=True))
                        ]))

    def forward(self, x):
        x = self.model(x)
        return x