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

class InfillLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=False, ignore_index=-100 ):
        super(InfillLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        #self.mean = torch.mean.cuda()

    def forward(self,predict,labels_basic,adc,weights):
        """
        predict: (b,1,h,w) tensor with output from logsoftmax
        adc:  (b,h,w) tensor with true adc values
        labels_basic: (b,h,w) tensor with zeros and ones marking dead regions
        (1 = dead region)
        psi1: (b,1,x,y) tensor of activation map from pool1 layer

        """
        _assert_no_grad(labels_basic)
        _assert_no_grad(weights)
        _assert_no_grad(adc)
        # _assert_no_grad(psi1)

        labels_basic = labels_basic.reshape(labels_basic.size(0),1,labels_basic.size(1),labels_basic.size(2))
        weights = weights.reshape(weights.size(0),1,weights.size(1),weights.size(2))

        # print "labels_basic: ",labels_basic.shape
        # print "adc: ",adc.shape
        # print "predict: ",predict.shape
        #
        # print "location of pred in loss: ",predict.get_device()
        # print "location of labels in loss: ",labels_basic.get_device()
        # print "location of adc in loss: ",adc.get_device()
        # print "location of weights in loss: ",weights.get_device()

        # calculate pixel loss in holes
        predictholes = predict * labels_basic.float() * weights.float()
        adcholes = adc * labels_basic.float() * weights.float()
        L1loss=torch.nn.L1Loss(self.size_average)
        holepixelloss = L1loss(predictholes, adcholes)
        holepixellosstotal = (holepixelloss.sum())/((weights.float()*labels_basic.float()).sum())
        #weighting the hole loss even more: very little of the image is dead but it's what we care about
        holeweight = labels_basic.float().sum()/(512.0 * 832.0)
        # print (1.0/holeweight.item())
        holepixellosstotal = holepixellosstotal * (1.0/holeweight)

        # calculate pixel loss in valid (seperate to allow for weighting)
        predictvalid = predict * (1-labels_basic.float()) * weights.float()
        adcvalid = adc * (1-labels_basic.float()) * weights.float()
        validpixelloss = L1loss(predictvalid, adcvalid)
        validpixellosstotal = (validpixelloss.sum())/((weights.float()*(1-labels_basic.float())).sum())

        # perception loss
        # comp is prediction, but non-dead regions are filled with truth
        # comp = adcvalid + predictholes
        # # mm: matrix multiplication
        # print "psi: ", psi1.shape
        # print "predict: ",predict.shape
        # actout= torch.matmul(psi1,predict)
        # print "actout: ", actout.shape
        # actcomp = torch.mm(psi1,comp)
        # acttruth = torch.mm(psi1,adc)
        # #sum of 2 L1 losses
        # perceploss = L1loss(actout,acttruth)+L1loss(actcomp,acttruth)
        # print "precption loss: ", perceploss.sum()

        # get total loss
        # still need:
        # style loss
        # total variation loss

        loss = holepixellosstotal + validpixellosstotal
        # loss = (holepixelloss.sum()+validpixelloss.sum())/weights.float().sum()

        return loss, holepixellosstotal, validpixellosstotal
