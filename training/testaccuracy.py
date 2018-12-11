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
    predict = torch.tensor([[[6.0,1.0],[4.0,2.0]],[[26.0,4.0],[6.0,2.0]]])
    print "Predict"
    print predict
    target = torch.tensor([[[25.0,6.0],[10.0,8.0]],[[25.0,6.0],[10.0,8.0]]])
    print "adc"
    print target
    labels = torch.tensor([[[1.0,0.0],[1.0,0.0]],[[1.0,0.0],[1.0,0.0]]])
    print "labels_basic"
    print labels

    accvals = (2,5,10,20)

    pred_dead = predict.float()*labels.float()
    target_dead = target.float()*labels.float()
    print "pred_dead: ", pred_dead
    print "target_dead: ", target_dead

    # pred_deadcharge = pred_dead * (target > 0.0).float()
    # target_deadcharge = target_dead * (target > 0.0).float()
    # print "pred_deadcharge: ", pred_deadcharge
    # print "target_deadcharge: ", target_deadcharge

    err = (pred_dead - target_dead).abs()
    # errcharge = (pred_deadcharge - target_deadcharge).abs()
    totaldeadpix = labels.float().sum().item()
    # totaldeadchargepix = (labels.float() * (target > 0.0).float()).sum().item()
    print  "err: ",err
    print "totaldeadpix: ", totaldeadpix
    # print  "errcharge: ",errcharge
    # print "totaldeadchargepix: ", totaldeadchargepix

    for level in accvals:
        # print (errcharge.lt(level).float()*labels.float())
        # print level, "acc charge: " , (errcharge.lt(level).float()*labels.float()*(target>0.0).float()).sum().item()/totaldeadchargepix
        print level, "acc: " , (err.lt(level).float()*labels.float()).sum().item()/totaldeadpix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()
