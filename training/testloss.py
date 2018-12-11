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

def main():
    predict = torch.tensor([[[5.0,8.0],[1.0,10.0]]])
    print "Predict"
    print predict
    adc = torch.tensor([[[3.0,7.0],[2.0,8.0]]])
    print "adc"
    print adc
    labels_basic = torch.tensor([[0.0,1.0],[1.0,0.0]])
    print "labels_basic"
    print labels_basic

    # calculate pixel loss in holes
    predictholes = predict * labels_basic.float()
    print "predictholes"
    print predictholes
    adcholes = adc * labels_basic.float()
    print "adcholes"
    print adcholes
    L1loss=torch.nn.L1Loss(False)
    holepixelloss = L1loss(predictholes, adcholes)
    print "holepixelloss"
    print holepixelloss

    # calculate pixel loss in valid (seperate to allow for weighting)
    predictvalid = predict * (1-labels_basic.float())
    print "predictvalid"
    print predictvalid
    adcvalid = adc * (1-labels_basic.float())
    print "adcvalid"
    print adcvalid
    validpixelloss = L1loss(predictvalid, adcvalid)
    print "validpixelloss"
    print validpixelloss

    loss = holepixelloss.sum() + validpixelloss.sum()
    print "loss"
    print loss.item()

if __name__ == '__main__':
    main()
