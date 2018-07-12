#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
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

# tensorboardX
from tensorboardX import SummaryWriter

# dataset interface
from larcvdataset import LArCVDataset

# Our model definitions
if "UBRESNET_MODELDIR" in os.environ:
    sys.path.append( os.environ["UBRESNET_MODELDIR"] )
else:
    sys.path.append( "../models" )
from ub_resnet import UResNet # copy of old ssnet

# Loss Functions
from pixelwise_nllloss import PixelWiseNLLLoss # pixel-weighted loss


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=True
RUNPROFILER=False
CHECKPOINT_FILE=""
start_iter  =  14500
# on meitner
#TRAIN_LARCV_CONFIG="flowloader_train.cfg"
#VALID_LARCV_CONFIG="flowloader_valid.cfg"
# on tufts grid
TRAIN_LARCV_CONFIG="ubresnet_train.cfg"
VALID_LARCV_CONFIG="ubresnet_valid.cfg"
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
ADC_THRESH=10.0
VISI_WEIGHT=0.1
USE_VISI=False
DEVICE_IDS=[3,4,5]
GPUID=DEVICE_IDS[0]
# map multi-training weights 
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:2",
                          "cuda:1":"cuda:3",
                          "cuda:2":"cuda:4"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=True
DEVICE="cuda:0" 
#DEVICE="cpu"
# ===================================================


# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer

    # create model, mark it to run on the GPU
    if GPUMODE:
        model = UResNet(inplanes=32,input_channels=1,num_classes=3,showsizes=False, use_visi=USE_VISI)
        model.to(device=torch.device(DEVICE)) # put onto gpuid
    else:
        model = UResNet(inplanes=32,input_channels=1,num_classes=3)

    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])

    if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids

    # uncomment to dump model
    print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda

    # define loss function (criterion) and optimizer
    if GPUMODE:
        criterion = PixelWiseNLLLoss()
        criterion.to(device=torch.device(DEVICE))
    else:
        criterion = PixelWiseNLLLoss()

    # training parameters
    lr = 1.0e-5
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = 4*len(DEVICE_IDS)
    batchsize_valid = 2*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    num_iters   = 30000
    iter_per_epoch = None # determined later
    iter_per_valid = 10
    iter_per_checkpoint = 500

    nbatches_per_itertrain = 20
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = 100
    
    nbatches_per_itervalid = 40
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = 100

    # SETUP OPTIMIZER

    # SGD w/ momentum
    #optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)
    
    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr, 
                                 weight_decay=weight_decay)
    
    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

    # LOAD THE DATASET
    
    iotrain = LArCVDataset(TRAIN_LARCV_CONFIG,"ThreadProcessorTrain")
    iovalid = LArCVDataset(VALID_LARCV_CONFIG,"ThreadProcessorValid")
    iotrain.start( batchsize_train )
    iovalid.start( batchsize_valid )
    iosample = {"valid":iovalid,
                "train":iotrain}

    NENTRIES = len(iotrain)
    print "Number of entries in training set: ",NENTRIES

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print "Number of epochs: ",epochs
    print "Iter per epoch: ",iter_per_epoch

    
    if False:
        # for debugging/testing data
        sample = "valid"
        print "TEST BATCH: sample=",sample
        adc_t,label_t,weight_t = prep_data( iosample[sample], sample, batchsize_train, 
                                            IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        print "adc shape: ",adc_t.shape()
        print "label shape: ",label_t.shape()
        print "weight shape: ",weight_t.shape()

        # load opencv, to dump png of image
        import cv2 as cv
        cv.imwrite( "testout_adc.png",    adc_t.numpy()[0,0,:,:] )
        cv.imwrite( "testout_label.png",  label_t.numpy()[0,:,:] )
        cv.imwrite( "testout_weight.png", weight_t.numpy()[0,0,:,:] )
        
        print "STOP FOR DEBUGGING"
        iotrain.stop()
        iovalid.stop()
        sys.exit(-1)

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # Resume training option
        #if RESUME_FROM_CHECKPOINT:
        #    print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        #    checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS )
        #    best_prec1 = checkpoint["best_prec1"]
        #    model.load_state_dict(checkpoint["state_dict"])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #if GPUMODE:
        #    optimizer.cuda(GPUID)

        for ii in range(start_iter, num_iters):

            adjust_learning_rate(optimizer, ii, lr)
            print "MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one iteration
            try:
                train_ave_loss, train_ave_acc = train(iotrain, batchsize_train, model,
                                                      criterion, optimizer,
                                                      nbatches_per_itertrain, ii, trainbatches_per_print)
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Train Iter:%d Epoch:%d.%d train aveloss=%.3f aveacc=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,train_ave_loss,train_ave_acc)

            # evaluate on validation set
            if ii%iter_per_valid==0:
                try:
                    prec1 = validate(iovalid, batchsize_valid, model, criterion, nbatches_per_itervalid, validbatches_per_print, ii)
                except Exception,e:
                    print "Error in validation routine!"            
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                # check point for best model
                if is_best:
                    print "Saving best model"
                    save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%iter_per_checkpoint==0:
                print "saving periodic checkpoint"
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
            # flush the print buffer after iteration
            sys.stdout.flush()
                
        # end of profiler context
        print "saving last state"
        save_checkpoint({
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)


    print "FIN"
    print "PROFILER"
    print prof
    writer.close()


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, iiter, nclasses, print_freq):

    global writer

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters
    losses = AverageMeter()
    acc_list = [] 
    for i in range(nclasses+1): # last accuracy is for total
        acc_list.append( AverageMeter() )

    # switch to train mode
    model.train()

    nnone = 0
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()        
        adc_t, label_t, weight_t = prep_data( train_loader, "train", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )        
        data_time.update( time.time()-end )

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        pred_t = model.forward(adc_t)
        loss = criterion.calc_loss(pred_t,label_t,weight_t)
        if RUNPROFILER:
            torch.cuda.synchronize()                
        forward_time.update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()                
        end = time.time()        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        backward_time.update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # measure accuracy and record loss
        acc_values = accuracy(pred_t,label_t)
        if acc_values is not None:
            losses.update(loss.data[0])
            for iacc,acc in enumerate(acc_list):
                acc.update( acc_values[iacc] )
        else:
            nnone += 1
        
        acc_time.update(time.time()-end)

        # measure elapsed time for batch
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (iiter,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      forward_time.val,forward_time.avg,
                      backward_time.val,backward_time.avg,
                      acc_time.val,acc_time.avg,                      
                      losses.val,losses.avg,
                      acc_list[-1].val,acc_list[1].avg)
            print "Train Iter: [%d][%d/%d]  Batch %.3f (%.3f)  Data %.3f (%.3f)  Forw %.3f (%.3f)  Back %.3f (%.3f) Acc %.3f (%.3f)\t || \tLoss %.3f (%.3f)\tAcc[total] %.3f (%.3f)"%status


    status = (iiter,
              batch_time.avg,
              data_time.avg,
              forward_time.avg,
              backward_time.avg,
              acc_time.avg,                      
              losses.avg,
              acc_list[-1].avg,
              nnone)
    print "Train Iter [%d] Ave: Batch %.3f  Data %.3f  Forw %.3f  Back %.3f  Acc %.3f ||  Loss %.3f Acc[Total] || NumNone=%d"%status

    writer.add_scalar( 'data/train_loss', losses.avg, iiter )        
    writer.add_scalars('data/train_accuracy', {'bg':     acc_list[0].avg,
                                               'shower': acc_list[1].avg,
                                               'track':  acc_list[2].avg,
                                               'total':  acc_list[3].avg} iiter )
    
    return losses.avg,acc_list[1].avg


def validate(val_loader, batchsize, model, criterion, nbatches, print_freq, iiter):
    """
    inputs
    ------
    val_loader: instance of LArCVDataSet for loading data
    batchsize (int): image (sets) per batch
    model (pytorch model): network
    criterion (pytorch module): loss function
    nbatches (int): number of batches to process
    print_freq (int): number of batches before printing output
    iiter (int): current iteration number of main loop
    
    outputs
    -------
    average percent of predictions within 5 pixels of truth
    """


    global writer
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    vis_acc = AverageMeter()
    load_data = AverageMeter()
    acc_list = []
    for i in range(nclasses+1):
        acc_list.append( AverageMeter() )
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()
        
        tdata_start = time.time()
        adc_t,label_t,weight_t = prep_data( val_loader, "valid", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        load_data.update( time.time()-tdata_start )
        
        # compute output
        pred_t = model.forward(adc_t)
        loss_t = criterion.calc_loss(pred_t,label_t,weight_t)

        # measure accuracy and record loss
        acc_values = accuracy(pred_t,label_t)
        if acc_values is not None:
            losses.update(loss.data[0])
            for iacc,acc in enumerate(acc_list):
                acc.update( acc_values[iacc] )
        else:
            nnone += 1
                
        # measure elapsed time
        batch_time.update(time.time() - batchstart)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,acc_list[-1].val,acc_list[-1].avg)
            print "Valid: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tAcc[Total] %.3f (%.3f)"%status

    status = (iiter,batch_time.avg,load_data.avg,losses.avg,acc_list[11].avg, nnone)
    print "Valid Iter %d sum: Batch %.3f\tData %.3f || Loss %.3f\tAcc[Total] %.3f\tNone=%d"%status    

    writer.add_scalar( 'data/valid_loss', losses.avg, iiter )
    writer.add_scalars('data/valid_accuracy', {'bg':     acc_list[0].avg,
                                               'shower': acc_list[1].avg,
                                               'track':  acc_list[2].avg,
                                               'total':  acc_list[-1].avg},iiter)

    print "Test:Result* Acc[Total] %.3f\tLoss %.3f"%(acc_list[-1].avg,losses.avg)

    return float(acc_list[-1].avg)


def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    profile = False
    # needs to be as gpu as possible!
    maxk = 1
    batch_size = target.size(0)
    if profile:
        torch.cuda.synchronize()
        start = time.time()    
    #_, pred = output.topk(maxk, 1, True, False) # on gpu. try never to use. it's slow AF
    _, pred = output.max( 1, keepdim=False) # max index along the channel dimension
    if profile:
        torch.cuda.synchronize()
        print "time for topk: ",time.time()-start," secs"

    if profile:
        start = time.time()
    #print "pred ",pred.size()," iscuda=",pred.is_cuda
    #print "target ",target.size(), "iscuda=",target.is_cuda
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy
    correct  = pred.eq( targetex ) # on gpu
    #print "correct ",correct.size(), " iscuda=",correct.is_cuda    
    if profile:
        torch.cuda.synchronize()
        print "time to calc correction matrix: ",time.time()-start," secs"

    # we want counts for elements wise
    num_per_class = {}
    corr_per_class = {}
    total_corr = 0
    total_pix  = 0
    if profile:
        torch.cuda.synchronize()            
        start = time.time()
    for c in range(output.size(1)):
        # loop over classes
        classmat = targetex.eq(int(c))        # pixels where class 'c' is correct answer
        #print "classmat: ",classmat.size()," iscuda=",classmat.is_cuda
        num_per_class[c]  = classmat.sum()     # number of correct answers
        corr_per_class[c] = (correct*classmat).sum() # mask by class matrix, then sum
        total_corr += corr_per_class[c]
        total_pix  += num_per_class[c]
    if profile:
        torch.cuda.synchronize()                
        print "time to reduce: ",time.time()-start," secs"
        
    # make result vector
    res = []
    for c in range(output.size(1)):
        if num_per_class[c]>0:
            res.append( corr_per_class[c]/float(num_per_class[c])*100.0 )
        else:
            res.append( 0.0 )

    # totals
    res.append( 100.0*float(total_corr)/total_pix )
        
    return res

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def prep_data( larcvloader, train_or_valid, batchsize, width, height, src_adc_threshold ):
    """
    This example is for larflow
    
    inputs
    ------
    larcvloader: instance of LArCVDataloader
    train_or_valid (str): "train" or "valid"
    batchsize (int)
    width (int)
    height(int)
    src_adc_threshold (float)

    outputs
    -------
    source_t (Pytorch Tensor): source ADC
    label_t  (Pytorch Variable): labels image
    flow_var (Pytorch Variable): flow from source to target
    visi_var (Pytorch Variable): visibility of source (long)
    fvisi_var(Pytorch Variable): visibility of target (float)
    """

    # get data
    data = larcvloader[0]

    # make torch tensors from numpy arrays
    source_t = torch.from_numpy( data["source_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) # source image ADC
    weight_t = torch.from_numpy( data["weight_%s"%(train_or_valid)].reshape(  (batchsize,1,width,height) ) ) # target image ADC    
    label_t  = torch.from_numpy( data["label_%s"%(train_or_valid)].reshape(  (batchsize,width,height) ) ) # target image ADC

    # apply threshold to source ADC values. returns a byte mask
    #source_t[ source_t<src_adc_threshold ] = 0.0
    #label_t[  source_t<src_adc_threshold ] = 0
    
    source_t.to(device=torch.device(DEVICE))
    label_t.to( device=torch.device(DEVICE))
    weight_t.to(device=torch.device(DEVICE))

    return source_t,label_t,weight_t

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
