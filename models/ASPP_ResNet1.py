# Liquid Argon Computer Vision
# Atrous Spatial Pyramid Pooling Architecture (ASPP)
# U-Net with ResNet modules

# e_4 & e_5 skip connections with image pooling

# Import Script
import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
from numbers import Integral

# python,numpy
import os
import sys
import commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import warnings

#####


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

        self.bypass = None
        self.bnpass = None
        if inplanes != planes or stride > 1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnpass = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.bypass is not None:
            outbp = self.bypass(x)
            outbp = self.bnpass(outbp)
            out += outbp
        else:
            out += x

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # residual path
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # if stride >1, then we need to subsamble the input
        if stride > 1:
            self.shortcut = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):

        if self.shortcut is None:
            bypass = x
        else:
            bypass = self.shortcut(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)

        residual = self.conv3(residual)
        residual = self.bn3(residual)

        out = bypass + residual
        out = self.relu(out)

        return out


class DoubleResNet(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DoubleResNet, self).__init__()
        # self.res1 = Bottleneck(inplanes,planes,stride)
        # self.res2 = Bottleneck(  planes,planes,     1)
        self.res1 = BasicBlock(inplanes, planes, stride)
        self.res2 = BasicBlock(planes, planes, 1)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        return out


class ConvTransposeLayer(nn.Module):
    def __init__(self, deconv_inplanes, deconv_outplanes, res_outplanes):
        super(ConvTransposeLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(deconv_inplanes, deconv_outplanes, kernel_size=4, stride=2, padding=1, bias=False)
        self.res = DoubleResNet(res_outplanes + deconv_outplanes, res_outplanes, stride=1)

    def forward(self, x, skip_x):
        out = self.deconv(x, output_size=skip_x.size())
        # concat skip connections
        out = torch.cat([out, skip_x], 1)
        out = self.res(out)
        return out


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes=16, nkernels=16):
        super(ASPP, self).__init__()

        stride = 1
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nkernels = nkernels

        # Block 1
        self.B1_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=stride, padding=0, dilation=1, bias=True)
        self.B1_bn = nn.BatchNorm2d(self.nkernels)
        self.B1_relu = nn.ReLU(inplace=True)

        # Block 2
        self.B2_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=True)
        self.B2_bn = nn.BatchNorm2d(self.nkernels)
        self.B2_relu = nn.ReLU(inplace=True)

        # Block 3
        self.B3_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=3, dilation=3, bias=True)
        self.B3_bn = nn.BatchNorm2d(self.nkernels)
        self.B3_relu = nn.ReLU(inplace=True)

        # Block 4
        self.B4_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=5, dilation=5, bias=True)
        self.B4_bn = nn.BatchNorm2d(self.nkernels)
        self.B4_relu = nn.ReLU(inplace=True)

        # Block 5
        self.B5_gp = torch.nn.MaxPool2d(kernel_size=3, stride=stride, padding=1, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):

        # Atrous Spatial Pyramid Pooling
        # Block 1
        b1 = self.B1_conv(x)
        b1 = self.B1_bn(b1)
        b1 = self.B1_relu(b1)

        # Block 2
        b2 = self.B2_conv(x)
        b2 = self.B2_bn(b2)
        b2 = self.B2_relu(b2)

        # Block 3
        b3 = self.B3_conv(x)
        b3 = self.B3_bn(b3)
        b3 = self.B3_relu(b3)

        # Block 4
        b4 = self.B4_conv(x)
        b4 = self.B4_bn(b4)
        b4 = self.B4_relu(b4)

        # Block 5
        b5 = self.B5_gp(x)

        # Concatenation along the depth
        x = torch.cat((b1, b2, b3, b4, b5), 1)

        return x


class ASPP_post(nn.Module):  # from ASPP_combine
    def __init__(self, inplanes, outplanes):
        super(ASPP_post, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nkernels = outplanes
        self.outplanes = outplanes

        self.ASPP_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ASPP_bn = nn.BatchNorm2d(self.nkernels)
        self.ASPP_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Final 1x1 convolution for rich feature extraction
        x = self.ASPP_conv(x)
        x = self.ASPP_bn(x)
        x = self.ASPP_relu(x)

        return x


class ASPP_ResNet(nn.Module):

    def __init__(self, num_classes=3, in_channels=3, inplanes=16, showsizes=False):
        self.inplanes = inplanes
        super(ASPP_ResNet, self).__init__()

        # Class variables
        stride = 1
        stride_mp = 2

        self.nkernels = 16
        self.inplanes = inplanes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.showsizes = showsizes

        # Encoder

        # Stem 1
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=True)  # initial conv layer
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Encoding Layer
        self.enc_layer1 = self._make_encoding_layer(self.inplanes * 1, self.inplanes * 2, stride=1)  # 16->32
        self.enc_layer2 = self._make_encoding_layer(self.inplanes * 2, self.inplanes * 4, stride=2)  # 32->64
        self.enc_layer3 = self._make_encoding_layer(self.inplanes * 4, self.inplanes * 8, stride=2)  # 64->128
        self.enc_layer4 = self._make_encoding_layer(self.inplanes * 8, self.inplanes * 16, stride=2)  # 128->256
        self.enc_layer5 = self._make_encoding_layer(self.inplanes * 16, self.inplanes * 32, stride=2)  # 256->512

        # Atrous Spatial Pyramid Pooling (ASPP)

        self.ASPP_layer_enc3 = self.ASPP_layer(self.inplanes * 8)
        self.ASPP_combine_enc3 = self.ASPP_combine(self.inplanes * 4, self.inplanes * 8)

        self.ASPP_layer_enc4 = self.ASPP_layer(self.inplanes * 16)
        self.ASPP_combine_enc4 = self.ASPP_combine(self.inplanes * 20, self.inplanes * 16)

        self.ASPP_layer_enc5 = self.ASPP_layer(self.inplanes * 32)
        self.ASPP_combine_enc5 = self.ASPP_combine(self.inplanes * 36, self.inplanes * 32)

        # Decoding Layer
        self.dec_layer5 = self._make_decoding_layer(self.inplanes * 64, self.inplanes * 16, self.inplanes * 32)  # 512->256
        self.dec_layer4 = self._make_decoding_layer(self.inplanes * 32, self.inplanes * 8, self.inplanes * 8)  # 256->128
        self.dec_layer3 = self._make_decoding_layer(self.inplanes * 8, self.inplanes * 4, self.inplanes * 4)  # 128->64
        self.dec_layer2 = self._make_decoding_layer(self.inplanes * 4, self.inplanes * 2, self.inplanes * 2)  # 64->32
        self.dec_layer1 = self._make_decoding_layer(self.inplanes * 2, self.inplanes, self.inplanes)  # 32->16

        # Final conv stem (7x7) = (3x3)^3
        self.nkernels = 16
        self.conv10 = nn.Conv2d(self.inplanes, self.nkernels, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn10 = nn.BatchNorm2d(self.nkernels)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(self.inplanes, num_classes, kernel_size=7, stride=1, padding=3, bias=True)

        # we use log softmax in order to more easily pair it with
        self.softmax = nn.LogSoftmax(dim=1)  # should return [b,c=3,h,w], normalized over, c dimension

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoding_layer(self, inplanes, planes, stride=2):
        return DoubleResNet(inplanes, planes, stride=stride)

    def _make_decoding_layer(self, inplanes, deconvplanes, resnetplanes):
        return ConvTransposeLayer(inplanes, deconvplanes, resnetplanes)

    def ASPP_layer(self, inplanes):
        return ASPP(inplanes)

    def ASPP_combine(self, inplanes, outplanes):
        return ASPP_post(inplanes, outplanes)

    def forward(self, x):

        if self.showsizes:
            print "input: ", x.size(), " is_cuda=", x.is_cuda

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu1(x)
        x = self.pool1(x0)  # Still in memory

        if self.showsizes:
            print "after conv1, x0: ", x0.size()

        # Encoding Layer
        e1 = self.enc_layer1(x)
        e2 = self.enc_layer2(e1)
        e3 = self.enc_layer3(e2)
        e4 = self.enc_layer4(e3)
        e5 = self.enc_layer5(e4)
        if self.showsizes:
            print "after encoding: "
            print "  e1: ", e1.size()
            print "  e2: ", e2.size()
            print "  e3: ", e3.size()
            print "  e4: ", e4.size()
            print "  e5: ", e5.size()

        # ASPP

        # e3_ASPP = self.ASPP_layer_enc3(e3)
        # e3_ASPP = self.ASPP_combine_enc3(e3_ASPP)
        # e3_skip = torch.cat((e3_ASPP, e3), 1)

        e4_ASPP = self.ASPP_layer_enc4(e4)
        if self.showsizes:
            print "e4_ASPP size after ASPP_layer:", e4_ASPP.size()

        e4_ASPP = self.ASPP_combine_enc4(e4_ASPP)  # [10, 320, 32, 32]
        if self.showsizes:
            print "e4_ASPP size after ASPP_combine:", e4_ASPP.size()

        e4_skip = torch.cat((e4_ASPP, e4), 1)  # [10, 512, 32, 32]
        if self.showsizes:
            print "e4_skip size after cat:", e4_skip.size()

        e5_ASPP = self.ASPP_layer_enc5(e5)
        if self.showsizes:
            print "e5_ASPP size after ASPP_layer:", e5_ASPP.size()

        e5_ASPP = self.ASPP_combine_enc5(e5_ASPP)  # [10, 512, 16, 16]
        if self.showsizes:
            print "e5_ASPP size after ASPP_combine:", e5_ASPP.size()

        e5_skip = torch.cat((e5_ASPP, e5), 1)  # [10, 1024, 16, 16]
        if self.showsizes:
            print "e5_skip size after cat:", e5_skip.size()

        # Decoding Layer
        d5 = self.dec_layer5(e5_skip, e4_skip)

        d4 = self.dec_layer4(d5, e3)
        if self.showsizes:
            print "  dec4: ", d4.size(), " iscuda=", d4.is_cuda

        d3 = self.dec_layer3(d4, e2)
        if self.showsizes:
            print "  dec3: ", d3.size(), " iscuda=", d3.is_cuda

        d2 = self.dec_layer2(d3, e1)
        if self.showsizes:
            print "  dec2: ", d2.size(), " iscuda=", d2.is_cuda

        d1 = self.dec_layer1(d2, x0)

        if self.showsizes:
            print "  dec1: ", d1.size(), " iscuda=", d1.is_cuda

        x = self.conv10(d1)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.conv11(x)

        x = self.softmax(x)
        if self.showsizes:
            print "  softmax: ", x.size()

        return x
