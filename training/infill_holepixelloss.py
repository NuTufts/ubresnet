import os,sys

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

# -------------------------------------------------------------------------
# HolePixelLoss
# This loss mimics nividia's pixelwise loss for holes (L1)
# used in the infill network
# how well does the network do in dead regions?
# -------------------------------------------------------------------------

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class HolePixelLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        #self.mean = torch.mean.cuda()

    def forward(self,predict,labels_basic,adc):
        """
        predict: (b,1,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        """
        _assert_no_grad(labels_basic)
        _assert_no_grad(adc)
        print "labels_basic: ",labels_basic.shape
        print "adc: ",adc.shape
        print "predict: ",predict.shape

        # adc is true adc acc_values
        # labels_basic has 1 for holes, else 0

        # calculate loss per pixel
        pixelloss= abs(labels_basic*(predict - adc))

        # get total loss
        loss = pixelloss.sum()
        return loss
