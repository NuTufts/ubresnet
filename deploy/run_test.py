# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from ROOT import TH1F,TTree,TFile,TH2F,TCanvas,TLine,TAttFill,TPad
from larcv import larcv

# pytorch
import torch

# # numba
from numba import jit

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from infill_funcs import load_model

import time

# larflow
if "LARCVDATASET_BASEDIR" in os.environ:
    sys.path.append(os.environ["LARCVDATASET_BASEDIR"])
else:
    sys.path.append("../pytorch-larflow/larcvdataset") # default location
from larcvdataset import LArCVDataset

# @jit
def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    larcvdataset_config="""ThreadProcessor: {
        Verbosity:3
        NumThreads: 2
        NumBatchStorage: 2
        RandomAccess: false
        InputFiles: ["../../notebooks/cropinfill.root"]
        ProcessName: ["ADCmasked_valid"]
        ProcessType: ["BatchFillerImage2D"]
        ProcessList: {
            ADCmasked_valid: {
                Verbosity:3
                ImageProducer: "ADC"
                Channels: [1]
                EnableMirror: false
            }
        }
    }

    """

    with open("larcv_dataloader.cfg",'w') as f:
        print >> f,larcvdataset_config
    iotest = LArCVDataset( "larcv_dataloader.cfg","ThreadProcessor") #, store_eventids=True

    return iotest
def pixelloop(ADC_numpy,
    ADCvalue_numpy,
    diff_numpy):

    #loop through all pixels
    # calculate accuracies
    # create acc image
    for rows in range(512):
        for cols in range(832):
            truepix = ADC_numpy[ib,0,rows,cols].item()
            outpix = ADCvalue_numpy[ib,0,rows,cols].item()
            if outpix < 0:
                ADCvalue_numpy[ib,0,rows,cols] = 0
                outpix = 0
            diff = abs(truepix - outpix)
            diff_numpy[ib,0,rows,cols] = diff

    return diff_numpy, ADCvalue_numpy

if __name__=="__main__":
    # ARGUMENTS DEFINTION/PARSER
    start = time.time()
    if len(sys.argv)>1:
        crop_view_parser = argparse.ArgumentParser(description='Process cropped-image views through LArFlow.')
        crop_view_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
        crop_view_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
        crop_view_parser.add_argument( "-c", "--checkpoint",   required=True, type=str, help="location of model checkpoint file")
        crop_view_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on")
        crop_view_parser.add_argument( "-p", "--chkpt-gpuid",  default=0,     type=int, help="GPUID used in checkpoint")
        crop_view_parser.add_argument( "-b", "--batchsize",    default=2,     type=int, help="batch size" )
        crop_view_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
        crop_view_parser.add_argument( "-v", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)")

        args = crop_view_parser.parse_args(sys.argv)
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        gpuid                 = args.gpuid
        checkpoint_gpuid      = args.chkpt_gpuid
        batch_size            = args.batchsize
        verbose               = args.verbose
        nprocess_events       = args.nevents
    else:
        # for testing
        # for checkpoint files see: /mnt/disk1/nutufts/kmason/ubresnet/training/ on nudot
        input_larcv_filename = "../../notebooks/cropinfill.root" # test cropped image file
        output_larcv_filename = "output_infill.root"
        checkpoint_data = "/mnt/disk1/nutufts/kmason/ubresnet/training/vplane_40000.tar"
        batch_size = 1
        gpuid = 1
        checkpoint_gpuid = 0
        verbose = True
        nprocess_events = 110

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=832
    height=512

    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
    model.eval()

    # output IOManager
    outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    inputdata.start(batch_size)

    nevts = len(inputdata)
    if nprocess_events>=0:
        nevts = nprocess_events
    nbatches = nevts/batch_size
    if nevts%batch_size!=0:
        nbatches += 1


    ientry = 0
    for ibatch in range(nbatches):

        if verbose:
            print "=== [BATCH %d] ==="%(ibatch)
        #
        data = inputdata[0]

        # get input ADC(masked) images
        ADCmasked_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) ) # source image ADC
        ADCmasked_t = ADCmasked_t.to(device=torch.device("cuda:%d"%(gpuid)))

        ev_out_ADCmasked = outputdata.get_data("image2d", "ADCMasked")
        ADCmaskedout_t = ADCmasked_t.detach().cpu().numpy()

        # run model
        pred_ADCvalue = model.forward( ADCmasked_t) #ADC_t
        # get predictions from gpu
        ADCvalue_np = pred_ADCvalue.detach().cpu().numpy().astype(np.float32)

        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch
                break

            # get meta
            inputmeta.read_entry(ientry)
            ev_meta   = inputmeta.get_data("image2d","ADC")
            outmeta   = ev_meta.image2d_array()[2].meta()

            # save inputs to network for reference
            # and save original images

            ADCmasked_slice=ADCmaskedout_t[ib,0,:,:]
            ADCmasked_out = larcv.as_image2d_meta(ADCmasked_slice,outmeta)
            ev_out_ADCmasked.append( ADCmasked_out )


            # save output of network as images
            img_slice0 = ADCvalue_np[ib,0,:,:]

            ev_out_diff = outputdata.get_data("image2d","diff")

            ADCMasked_numpy = ADCmasked_t.detach().cpu().numpy().astype(np.float32)

            diff_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) )
            diff_numpy = diff_t.numpy()


            diff_numpy,ADCvalue_np = pixelloop(ADCMasked_numpy, ADCvalue_np, diff_numpy)

            diff_t = torch.from_numpy(diff_numpy)
            diff_out = larcv.as_image2d_meta(diff_t[ib,0,:,:], outmeta)
            ev_out_diff.append(diff_out)

            img_slice0 = torch.from_numpy(ADCvalue_np)
            out_lcv  = larcv.as_image2d_meta( img_slice0[ib,0,:,:], outmeta )
            ev_out_out  = outputdata.get_data("image2d","out")
            ev_out_out.append( out_lcv )

            beforeloop = time.time()

            outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            outputdata.save_entry()
            ientry += 1
            afterloop = time.time()


    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()

    print "DONE."
    end = time.time()
    print "time to loop start: ", (beforeloop - start)
    print "time to do loop: ", (afterloop - beforeloop)
    print "time to finalize: ", (end - afterloop)
    print "total time: ", (end - start)
