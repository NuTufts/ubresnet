#!/bin/env python

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

# Our model definitions
if "UBRESNET_BASEDIR" in os.environ:
    print "APPENDING REPO PATH TO PYTHONPATH: ",os.environ["UBRESNET_BASEDIR"]
    sys.path.append( os.environ["UBRESNET_BASEDIR"] )
else:
    raise RuntimeError("Did not find UBRESNET_BASEDIR environment variable. Was repo. setup properly")
    
from models.ASPP_ResNet1 import ASPP_ResNet as Model # copy of old ssnet

# Loss Functions
from training.pixelwise_nllloss import PixelWiseNLLLoss # pixel-weighted loss

# LArCV1 Data interface
#from larcvdataset import LArCV1Dataset

GPUMODE=True
GPUID=0
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="plane2_caffe/run1/checkpoint.20000th.tar"
    
# Data augmentation/manipulation functions
def padandcrop(npimg2d,nplabelid,npweightid):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]

def padandcropandflip(npimg2d):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 0 )
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 1 )
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]    


# SegData: class to hold batch data
# we expect LArCV1Dataset to fill this object
class SegData:
    def __init__(self):
        self.dim = None
        self.images = None # adc image
        self.labels = None # labels
        self.weights = None # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim
        
# Data interface: eventually will move to larcvdataset
class LArCV1Dataset:
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration. see test.py.ipynb for example of configuration
        self.name = name
        self.cfgfile = cfgfile
        return

    def init(self):
        # create instance of data file interface
        self.io = larcv.ThreadDatumFiller(self.name)
        self.io.configure(self.cfgfile)
        self.nentries = self.io.get_n_entries()
        self.io.set_next_index(0)
        print "[LArCV1Data] able to create ThreadDatumFiller"
        return

    def getbatch(self, batchsize):
        self.io.batch_process(batchsize)
        #time.sleep(0.1)
        itry = 0
        while self.io.thread_running() and itry<100:
            time.sleep(0.01)
            itry += 1
        if itry>=100:
            raise RuntimeError("Batch Loader timed out")

        # fill SegData object
        data = SegData()
        dimv = self.io.dim() # c++ std vector through ROOT bindings
        self.dim     = (dimv[0], dimv[1], dimv[2], dimv[3] )
        self.dim3    = (dimv[0], dimv[2], dimv[3] )

        # numpy arrays
        data.np_images  = np.zeros( self.dim,  dtype=np.float32 )
        data.np_labels  = np.zeros( self.dim3, dtype=np.int )
        data.np_weights = np.zeros( self.dim3, dtype=np.float32 )
        data.np_images[:]  = larcv.as_ndarray(self.io.data()).reshape(    self.dim  )[:]
        data.np_labels[:]  = larcv.as_ndarray(self.io.labels()).reshape(  self.dim3 )[:]
        data.np_weights[:] = larcv.as_ndarray(self.io.weights()).reshape( self.dim3 )[:]
        data.np_labels[:] += -1

        # pytorch tensors
        data.images = torch.from_numpy(data.np_images)
        data.labels = torch.from_numpy(data.np_labels)
        data.weight = torch.from_numpy(data.np_weights)
        #if GPUMODE:
        #    data.images.cuda()
        #    data.labels.cuda(async=False)
        #    data.weight.cuda(async=False)


        # debug values
        #print "max label: ",np.max(data.labels)
        #print "min label: ",np.min(data.labels)

        return data


torch.cuda.device( 1 )

# global variables
best_prec1 = 0.0         # best accuracy, use to decide when to save network weights
writer = SummaryWriter() # interface to Tensorboard

def main():

    global best_prec1
    global writer

    # create model, mark it to run on the GPU
    if GPUMODE:
        model = Model(inplanes=16,input_channels=1,num_classes=3,showsizes=False)
        model.cuda(GPUID)
    else:
        model = Model(inplanes=16,input_channels=1,num_classes=3)

    # uncomment to dump model
    print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda

    # define loss function (criterion) and optimizer
    if GPUMODE:
        criterion = PixelWiseNLLLoss().cuda(GPUID)
    else:
        criterion = PixelWiseNLLLoss()

    # training parameters
    base_lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-3

    # training length
    batchsize_train = 10
    batchsize_valid = 2
    start_epoch = 0
    epochs      = 1
    start_iter  = 0
    num_iters   = 10000
    #num_iters    = None # if None
    iter_per_epoch = None # determined later
    iter_per_valid = 10
    iter_per_checkpoint = 500

    nbatches_per_itertrain = 5
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = 1
    
    nbatches_per_itervalid = 25
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = 5

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    # LOAD THE DATASET

    # define configurations
    traincfg = """ThreadDatumFillerTrain: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    true
  #InputFiles:   ["/cluster/kappa/wongjiradlab/larbys/dllee_ssnet_trainingdata/train00.root","/cluster/kappa/wongjiradlab/larbys/dllee_ssnet_trainingdata/train01.root","/cluster/kappa/wongjiradlab/larbys/dllee_ssnet_trainingdata/train02.root","/cluster/kappa/wongjiradlab/larbys/dllee_ssnet_trainingdata/train03.root"]
  InputFiles:   ["/tmp/train00.root","/tmp/train01.root","/tmp/train02.root","/tmp/train03.root"]
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: true
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    validcfg = """ThreadDatumFillerValid: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    true
  #InputFiles:   ["/cluster/kappa/wongjiradlab/larbys/dllee_ssnet_trainingdata/val.root"]  
  InputFiles:   ["/tmp/val.root"]  
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: true
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    with open("segfiller_train.cfg",'w') as ftrain:
        print >> ftrain,traincfg
    with open("segfiller_valid.cfg",'w') as fvalid:
        print >> fvalid,validcfg
    
    iotrain = LArCV1Dataset("ThreadDatumFillerTrain","segfiller_train.cfg" )
    iovalid = LArCV1Dataset("ThreadDatumFillerValid","segfiller_valid.cfg" )
    print "initialize datasets ... "
    iotrain.init()
    iovalid.init()
    print "get first batch ... "
    iotrain.getbatch(batchsize_train)

    NENTRIES = iotrain.io.get_n_entries()
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

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # set the initial base learning rate
        lr = base_lr

        # Resume training option
        if RESUME_FROM_CHECKPOINT:
            checkpoint = torch.load( CHECKPOINT_FILE )
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
        

        for ii in range(start_iter, num_iters):
            iepoch = float(ii)/float(iter_per_epoch)
            lr = adjust_learning_rate(optimizer, iepoch, ii, lr, base_lr)
            print "Iter:%d Epoch:%.2f "%(ii,iepoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one epoch
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
            print "Iter:%d Epoch:%d.%d train aveloss=%.3f aveacc=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,train_ave_loss,train_ave_acc)

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


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, epoch, print_freq):

    global writer
    
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    format_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    acc_list = []
    for i in range(5):
        acc_list.append( AverageMeter() )

    # switch to train mode
    model.train()
    model.cuda(GPUID)

    for i in range(0,nbatches):
        #print "epoch ",epoch," batch ",i," of ",nbatches
        batchstart = time.time()

        # data loading time        
        end = time.time()        
        data = train_loader.getbatch(batchsize)
        data_time.update(time.time() - end)


        # convert to pytorch Variable (with automatic gradient calc.)
        end = time.time()        
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda(GPUID))
            labels_var = torch.autograd.Variable(data.labels.cuda(GPUID),requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(GPUID),requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)
        format_time.update( time.time()-end )

        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        output = model(images_var)
        loss = criterion(output, labels_var, weight_var)
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
        prec1 = accuracy(output.data, labels_var.data, images_var.data)
        acc_time.update(time.time()-end)

        # updates
        losses.update(loss.data[0], data.images.size(0))
        top1.update(prec1[-1], data.images.size(0))
        for i,acc in enumerate(prec1):
            acc_list[i].update( acc )

        # measure elapsed time for batch
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (epoch,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      format_time.val,format_time.avg,
                      forward_time.val,forward_time.avg,
                      backward_time.val,backward_time.avg,
                      acc_time.val,acc_time.avg,                      
                      losses.val,losses.avg,
                      top1.val,top1.avg)
            print "Iter: [%d][%d/%d]\tBatch %.3f (%.3f)\tData %.3f (%.3f)\tFormat %.3f (%.3f)\tForw %.3f (%.3f)\tBack %.3f (%.3f)\tAcc %.3f (%.3f)\t || \tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status

    writer.add_scalar('data/train_loss', losses.avg, epoch )        
    writer.add_scalars('data/train_accuracy', {'background': acc_list[0].avg,
                                               'track':  acc_list[1].avg,
                                               'shower': acc_list[2].avg,
                                               'total':  acc_list[3].avg,
                                               'nonzero':acc_list[4].avg}, epoch )        
    
    return losses.avg,top1.avg


def validate(val_loader, batchsize, model, criterion, nbatches, print_freq, iiter):

    global writer
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    acc_list = []
    for i in range(5):
        acc_list.append( AverageMeter() )
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i in range(0,nbatches):
        data = val_loader.getbatch(batchsize)

        # convert to pytorch Variable (with automatic gradient calc.)
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda(GPUID))
            labels_var = torch.autograd.Variable(data.labels.cuda(GPUID),requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(GPUID),requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)

        # compute output
        output = model(images_var)
        loss = criterion(output, labels_var, weight_var)
        #loss = criterion(output, labels_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, labels_var.data, images_var.data)
        losses.update(loss.data[0], data.images.size(0))
        top1.update(prec1[-1], data.images.size(0))
        for i,acc in enumerate(prec1):
            acc_list[i].update( acc )
                
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,top1.val,top1.avg)
            print "Valid: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status
            #print('Test: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        i, len(val_loader), batch_time=batch_time, loss=losses,
            #        top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'
    #      .format(top1=top1))

    writer.add_scalar( 'data/valid_loss', losses.avg, iiter )
    writer.add_scalars('data/valid_accuracy', {'background': acc_list[0].avg,
                                               'track':   acc_list[1].avg,
                                               'shower':  acc_list[2].avg,
                                               'total':   acc_list[3].avg,
                                               'nonzero': acc_list[4].avg}, iiter )

    print "Test:Result* Prec@1 %.3f\tLoss %.3f"%(top1.avg,losses.avg)

    return float(top1.avg)


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


def adjust_learning_rate(optimizer, epoch, iteration, current_lr, base_lr):
    """Sets the learning rate. Many different variables in order to provide customized rate"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = base_lr * ( 0.1**(iteration//10000) )
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def accuracy(output, target, imgdata):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    profile = False
    # needs to be as gpu as possible!
    maxk = 1
    batch_size = target.size(0)
    if profile:
        torch.cuda.synchronize()
        start = time.time()    
    #_, pred = output.topk(maxk, 1, True, False) # on gpu. slow AF
    _, pred = output.max( 1, keepdim=False) # on gpu
    if profile:
        torch.cuda.synchronize()
        print "time for topk: ",time.time()-start," secs"

    if profile:
        start = time.time()
    #print "pred ",pred.size()," iscuda=",pred.is_cuda
    #print "target ",target.size(), "iscuda=",target.is_cuda
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy
    correct = pred.eq( targetex ) # on gpu
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
        classmat = targetex.eq(int(c)) # elements where class is labeled
        #print "classmat: ",classmat.size()," iscuda=",classmat.is_cuda
        num_per_class[c] = classmat.sum()
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
    res.append( 100.0*float(corr_per_class[1]+corr_per_class[2])/(num_per_class[1]+num_per_class[2]) ) # track/shower acc
        
    return res

def dump_lr_schedule( startlr, numiters ):
    for iters in range(0,numiters):
        lr = adjust_learning_rate( None, 0, iters, lr, startlr )
    print "Iteration [%d] lr=%.3e"%(iters,lr)
    return

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
