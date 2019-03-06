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
from ub_uresnet_infill import UResNetInfill # copy of old ssnet

# Loss Functions
#from pixelwise_nllloss import PixelWiseNLLLoss # pixel-weighted loss
from infill_loss import InfillLoss # pixel loss in holes
# still need:
# valid pixel loss
# perception loss
# style loss
# total variation loss

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT= True
RUNPROFILER=False
CHECKPOINT_FILE="uplanefromv_26500.tar"
start_iter  = 26500
# on meitner
#TRAIN_LARCV_CONFIG="flowloader_train.cfg"
#VALID_LARCV_CONFIG="flowloader_valid.cfg"
# on tufts grid
TRAIN_LARCV_CONFIG="ubresnet_infill_train.cfg"
VALID_LARCV_CONFIG="ubresnet_infill_valid.cfg"
IMAGE_WIDTH=512
IMAGE_HEIGHT=832
ADC_THRESH=0.0
DEVICE_IDS=[0,1]
GPUID=DEVICE_IDS[0]
# map multi-training weights
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS="cuda:0"
CHECKPOINT_FROM_DATA_PARALLEL=True
DEVICE="cuda:0"
# ===================================================


# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer
    global outputs

    # create model, mark it to run on the GPU
    if GPUMODE:
        model = UResNetInfill(inplanes=32,input_channels=1,num_classes=1,showsizes=False )
        model.to(device=torch.device(DEVICE)) # put onto gpuid
    else:
        model = UResNetInfill(inplanes=32,input_channels=1,num_classes=1)

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
        print "IN DATA PARALLEL"

    # uncomment to dump model
    #print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda

    # define loss function (criterion) and optimizer
    if GPUMODE:
        criterion = InfillLoss()
        criterion.to(device=torch.device(DEVICE))
    else:
        criterion = InfillLoss()

    # training parameters
    lr = 1e-4
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    if "cuda" in DEVICE:
        batchsize_train = 1*len(DEVICE_IDS)
        batchsize_valid = 1*len(DEVICE_IDS)
    else:
        batchsize_train = 4
        batchsize_valid = 2

    start_epoch = 0
    epochs      = 40#10
    num_iters   = 60000#30000
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
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=lr,
    #                              weight_decay=weight_decay)
    # RMSProp
    optimizer = torch.optim.RMSprop(model.parameters(),
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

    if NENTRIES>itersize_train:
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
        sample = "train"
        print "TEST BATCH: sample=",sample
        adc_t,weights_t,adcmasked_t,labelbasic_t = prep_data( iosample[sample], sample, batchsize_train,
                                            IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        print "adc shape: ",adc_t.shape
        print "weights shape: ",weights_t.shape
        print "adcmasked shape: ",adcmasked_t.shape
        print "labelbasic shape: ",labelbasic_t.shape
        # load opencv, to dump png of image
        import cv2 as cv

        cv.imwrite( "testout_adc.png",    adc_t.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_weights.png",  weights_t.cpu().numpy()[0,:,:] )
        cv.imwrite( "testout_adcmasked.png",    adcmasked_t.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_labelbasic.png",  labelbasic_t.cpu().numpy()[0,:,:] )


        print "STOP FOR DEBUGGING"
        iotrain.stop()
        iovalid.stop()
        sys.exit(-1)

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # Resume training option
        if RESUME_FROM_CHECKPOINT:
           print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
           checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS )
           best_prec1 = checkpoint["best_prec1"]
           model.load_state_dict(checkpoint["state_dict"])
           optimizer.load_state_dict(checkpoint['optimizer'])
        #if GPUMODE:
          # optimizer.cuda(GPUID)

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


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, iiter,  print_freq):

    global writer
    global outputs

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters
    losses = AverageMeter()
    holelosses = AverageMeter()
    validlosses = AverageMeter()

    accnames = ("infilldead2",
                "infilldead5",
                "infilldead10",
                "infilldead20",
                "infilldeadcharge2",
                "infilldeadcharge5",
                "infilldeadcharge10",
                "infilldeadcharge20")
    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

    # switch to train mode
    model.train()
    # model.pool1.register_forward_hook(hook)

    # pool1_tensor = torch.tensor(glb_feature, requires_grad=True
    #                 , device=torch.device("cuda:0"))

    nnone = 0

    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
        adc_t, weights_t, adcmasked_t, labelbasic_t = prep_data( train_loader, "train", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        data_time.update( time.time()-end )

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        pred_t = model.forward(adcmasked_t)
        # print "pool1act_shape: ", outputs.shape
        # print "pool1act_sum: ", outputs.sum().item()
        # print "location of pred b/f loss (train): ",pred_t.get_device()
        # print "location of labels b/f loss (train): ",labelbasic_t.get_device()
        # print "location of adc b/f loss (train): ",adc_t.get_device()
        # print "location of weights b/f loss (train): ",weights_t.get_device()
        loss,holeloss,validloss = criterion.forward(pred_t ,labelbasic_t, adc_t, weights_t)
        # print "validpixelloss: ", validloss
        # print "holepixelloss: ", holeloss
        # print "totalloss: ", loss
        # print "--------------"

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
        acc_values = accuracy(pred_t.detach(),adc_t.detach(),labelbasic_t.detach(),acc_meters)

        if acc_values is not None:
            losses.update(loss.item())
            holelosses.update(holeloss.item())
            validlosses.update(validloss.item())

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
                      losses.val,losses.avg)
            print "Train Iter: [%d][%d/%d]  Batch %.3f (%.3f)  Data %.3f (%.3f)  Forw %.3f (%.3f)  Back %.3f (%.3f) Acc %.3f (%.3f)\t || \tLoss %.3f (%.3f)\t "%status

    print "Train Iter: ", iiter
    print "Loss(training): ", losses.avg
    print "Accuracycharge(training): @2[%.1f] @5[%.1f] @10[%.1f] @20[%.1f]"%(acc_meters["infilldeadcharge2"].avg,acc_meters["infilldeadcharge5"].avg,acc_meters["infilldeadcharge10"].avg,acc_meters["infilldeadcharge20"].avg)


    writer.add_scalars( 'data/train_loss', {'totalloss': losses.avg,
                                        'holeloss': holelosses.avg,
                                        'validloss': validlosses.avg}, iiter )

    writer.add_scalars('data/train_accuracy', {'deadcharge - 2 ADC': acc_meters['infilldeadcharge2'].avg,
                                               'deadcharge - 5 ADC': acc_meters['infilldeadcharge5'].avg,
                                               'deadcharge - 10 ADC': acc_meters['infilldeadcharge10'].avg,
                                               'deadcharge - 20 ADC': acc_meters['infilldeadcharge20'].avg}, iiter )


    return losses.avg, acc_meters['infilldead5'].avg


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
    global outputs

    batch_time = AverageMeter()
    losses = AverageMeter()
    holelosses = AverageMeter()
    validlosses = AverageMeter()
    vis_acc = AverageMeter()
    load_data = AverageMeter()
    accnames = ("infilldead2",
                "infilldead5",
                "infilldead10",
                "infilldead20",
                "infilldeadcharge2",
                "infilldeadcharge5",
                "infilldeadcharge10",
                "infilldeadcharge20")
    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model.pool1.register_forward_hook(hook)

    end = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()

        tdata_start = time.time()
        adc_t,weights_t,adcmasked_t,labelbasic_t = prep_data( val_loader, "valid", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        load_data.update( time.time()-tdata_start )

        # compute output
        pred_t = model.forward(adc_t)
        # print "location of pred b/f loss (valid): ",pred_t.get_device()
        # print "location of labels b/f loss (valid): ",labelbasic_t.get_device()
        # print "location of adc b/f loss (valid): ",adc_t.get_device()
        # print "location of weights b/f loss (valid): ",weights_t.get_device()
        loss_t, holeloss, validloss = criterion.forward(pred_t ,labelbasic_t, adc_t, weights_t)

        # measure accuracy and record loss
        acc_values = accuracy(pred_t.detach(),adc_t.detach(),labelbasic_t.detach(),acc_meters)
        if acc_values is not None:
            losses.update(loss_t.item())
            holelosses.update(holeloss.item())
            validlosses.update(validloss.item())


        else:
            nnone += 1

        # measure elapsed time
        batch_time.update(time.time() - batchstart)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg)
            print "Valid: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)"%status

    # status = (iiter,batch_time.avg,load_data.avg,losses.avg, nnone)
    # print "Valid Iter %d sum: Batch %.3f\tData %.3f || Loss %.3f\tNone=%d"%status
    print "Iter: ", iiter
    print "Loss(valid): ", losses.avg
    print "Accuracydeadcharge(valid): @2[%.1f] @5[%.1f] @10[%.1f] @20[%.1f]"%(acc_meters["infilldeadcharge2"].avg,acc_meters["infilldeadcharge5"].avg,acc_meters["infilldeadcharge10"].avg,acc_meters["infilldeadcharge20"].avg)

    writer.add_scalars( 'data/valid_loss', {'totalloss': losses.avg,
                                            'holeloss': holelosses.avg,
                                            'validloss': validlosses.avg}, iiter )

    writer.add_scalars('data/valid_accuracy', {'deadcharge - 2 ADC': acc_meters['infilldeadcharge2'].avg,
                                               'deadcharge- 5 ADC': acc_meters['infilldeadcharge5'].avg,
                                               'deadcharge - 10 ADC': acc_meters['infilldeadcharge10'].avg,
                                               'deadcharge - 20 ADC': acc_meters['infilldeadcharge20'].avg}, iiter )


    # print "Test:Result* Acc[Total] %.3f\tLoss %.3f"%(acc_meters['infilldead5'].avg,losses.avg)

    return float(acc_meters['infilldead5'].avg)


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

def accuracy(output, target, labels, acc_meters):
    """
    Computes the accuracy for infill network.
    Calculate the percentage of pixels that match the true adc within various thresholds
    """
    profile = False
    # needs to be as gpu as possible!
    maxk = 1
    # batch_size = target.size(0)
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

    accvals = (2.0,5.0,10.0,20.0)

    # print "output: ", output.size()
    # print "target: ", target.size()
    # print "labels: ", labels.size()

    labels = labels.reshape(labels.size(0),1,labels.size(1),labels.size(2))

    pred_dead = output.float()*labels.float()
    target_dead = target.float()*labels.float()


    pred_deadcharge = pred_dead * (target > 0.0).float()
    target_deadcharge = target_dead * (target > 0.0).float()

    err = (pred_dead - target_dead).abs()
    errcharge = (pred_deadcharge - target_deadcharge).abs()
    totaldeadpix = labels.float().sum().item()
    totaldeadchargepix = (labels.float() * (target > 0.0).float()).sum().item()
    # print "dead: ", totaldeadpix
    # print "deadcharge: ", totaldeadchargepix

    for level in accvals:
        name = "infilldeadcharge%d"%(level)
        acc_meters[name].update((errcharge.lt(level).float()*labels.float()*(target>0.0).float()).sum().item()/totaldeadchargepix)
        name = "infilldead%d"%(level)
        acc_meters[name].update((err.lt(level).float()*labels.float()).sum().item()/totaldeadpix)


    if profile:
        torch.cuda.synchronize()
        print "time to calc correction matrix: ",time.time()-start," secs"

    # print "2: ",acc_meters["infilldead2"].avg
    # print "5: ",acc_meters["infilldead5"].avg
    # print "10: ",acc_meters["infilldead10"].avg
    # print "20: ",acc_meters["infilldead20"].avg
    #
    # print "2charge: ",acc_meters["infilldeadcharge2"].avg
    # print "5charge: ",acc_meters["infilldeadcharge5"].avg
    # print "10charge: ",acc_meters["infilldeadcharge10"].avg
    # print "20charge: ",acc_meters["infilldeadcharge20"].avg
    #
    # print "============================================="

    return acc_meters["infilldead2"],acc_meters["infilldead5"],acc_meters["infilldead10"],acc_meters["infilldead20"],acc_meters["infilldeadcharge2"],acc_meters["infilldeadcharge5"],acc_meters["infilldeadcharge10"],acc_meters["infilldeadcharge20"]

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
    adc_t (Pytorch Tensor):   true ADC
    label_t  (Pytorch Variable): labels image
    adcmasked_t (Pytorch Tensor):    ADC with dead regions
    labelbasic_t  (Pytorch Variable): labels of dead regions image
    weight_t (Pytorch Variable): weights image

    """

    # get data
    data = larcvloader[0]

    # make torch tensors from numpy arrays
    adcmasked_t = torch.from_numpy( data["adcmasked_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) # source image ADC
    adc_t = torch.from_numpy( data["adc_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) )   # target image ADC
    weights_t = torch.from_numpy( data["weights_%s"%(train_or_valid)].reshape( (batchsize,width,height) ) )
    labelbasic_t = torch.from_numpy( data["labelsbasic_%s"%(train_or_valid)].reshape( (batchsize,width,height) ).astype(np.int) )

    adc_t=adc_t.to(device=torch.device(DEVICE))
    weights_t=weights_t.to( device=torch.device(DEVICE))
    adcmasked_t=adcmasked_t.to(device=torch.device(DEVICE))
    labelbasic_t=labelbasic_t.to( device=torch.device(DEVICE))

    #print "after to cuda: ",weight_t.sum()
    return adc_t,weights_t,adcmasked_t,labelbasic_t

# -----------------Hooking functions--------------------------------

def hook(module, input, output):
    global outputs
    outputs = torch.tensor(torch.zeros(1, 32, 256, 412)
                        , requires_grad=True, device=torch.device("cuda:0"))
    outputs = output.data
    # print(output.data.sum().item(), outputs.data.sum().item())

def printnorm(self, input, output):
   # input is a tuple of packed inputs
   # output is a Tensor. output.data is the Tensor we are interested
   print('Inside ' + self.__class__.__name__ + ' forward')
   print('')
   print('input: ', type(input))
   print('input[0]: ', type(input[0]))
   print('output: ', type(output))
   print('')
   print('input size:', input[0].size())
   print('output size:', output.data.size())
   print('output norm:', output.data.norm())
# -------------------------------------------------------------------

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
