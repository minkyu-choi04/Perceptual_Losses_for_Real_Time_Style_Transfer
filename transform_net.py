import torch
import torch.nn as nn
import numpy as np

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(ConvLayer, self).__init__()
        self.upsample_f = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample)
        self.padding = nn.ReflectionPad2d(int(np.floor(kernel_size / 2)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, feature):
        y = feature
        if self.upsample_f:
            y = self.upsample(y)
        y = self.padding(y)
        y = self.conv(y)
        return y


class BasicResidualBlock(nn.Module):
    '''Residual block
    Original Code is from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Some parts are modified. 
    '''
    
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = ConvLayer(inplanes, planes, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.process = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 128),
            BasicResidualBlock(128, 128),
            ConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ConvLayer(32, 3, kernel_size=9, stride=1),
            nn.Sigmoid()
            )
        '''
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.res1 = BasicResidualBlock(128, 128)
        self.res2 = BasicResidualBlock(128, 128)
        self.res3 = BasicResidualBlock(128, 128)
        self.res4 = BasicResidualBlock(128, 128)
        self.res5 = BasicResidualBlock(128, 128)

        self.deconv1 = ConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2 = ConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1, upsample=1)

        self.relu = nn.ReLU()
        '''

    def forward(self, x):
        return self.process(x)


