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

# larcv
if "LARCVDATASET_BASEDIR" in os.environ:
    sys.path.append(os.environ["LARCVDATASET_BASEDIR"])
else:
    sys.path.append("../pytorch-larflow/larcvdataset") # default location
from larcvdataset import LArCVDataset


def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    # set ImageProducer to branch names in input root file
    # channel = plane
    larcvdataset_config="""ThreadProcessor: {
        Verbosity:3
        NumThreads: 2
        NumBatchStorage: 2
        RandomAccess: false
        InputFiles: ["crop_sample.root"]
        ProcessName: ["First","Second"]
        ProcessType: ["BatchFillerImage2D","BatchFillerImage2D"]
        ProcessList: {
            First: {
                Verbosity:3
                ImageProducer: "ADC"
                Channels: [2]
                EnableMirror: false
            }
            Second: {
                Verbosity:3
                ImageProducer: "LabelsBasic"
                Channels: [2]
                EnableMirror: false
            }
        }
    }

    """

    with open("larcv_dataloader.cfg",'w') as f:
        print >> f,larcvdataset_config
    iotest = LArCVDataset( "larcv_dataloader.cfg","ThreadProcessor") #, store_eventids=True

    return iotest


if __name__=="__main__":

    # ARGUMENTS DEFINTION/PARSER
    # for testing
    input_larcv_filename = "crop_sample.root" # test cropped image file
    output_larcv_filename = "output_diff.root"
    batch_size = 1
    gpuid = 1
    verbose = True
    nprocess_events = 100

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=832
    height=512

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

        data = inputdata[0]

        # get first of two images
        first_t = torch.from_numpy( data["First"].reshape( (batch_size,1,height,width) ) ) # source image ADC
        first_t = first_t.to(device=torch.device("cuda:%d"%(gpuid)))

        # get second of two images
        second_t = torch.from_numpy( data["Second"].reshape( (batch_size,1,height,width) ) )
        second_t = second_t.to(device=torch.device("cuda:%d"%(gpuid)))

        #save copys of input images
        # Names can be changed to fit file
        ev_out_first = outputdata.get_data("image2d","First")
        first_t = first_t.detach().cpu().numpy()
        ev_out_second = outputdata.get_data("image2d","Second")
        second_t = second_t.detach().cpu().numpy()

        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch
                break

            # get meta
            inputmeta.read_entry(ientry)

            # NEED TO CHANGE THIS DEPENDING ON INPUT ROOT FILE.
            ev_meta = inputmeta.get_data("image2d","ADC")
            outmeta = ev_meta.image2d_array()[2].meta()

            # save inputs to output file
            first_slice = first_t[ib,0,:,:]
            first_out = larcv.as_image2d_meta(first_slice,outmeta)
            ev_out_first.append( first_out )

            second_slice = second_t[ib,0,:,:]
            second_out = larcv.as_image2d_meta(second_slice,outmeta)
            ev_out_second.append( second_out )


            # create diff image
            ev_out_diff = outputdata.get_data("image2d","Diff")


            #save a copy of labels for use in creating diff image
            copy_t = torch.from_numpy( data["First"].reshape( (batch_size,1,height,width) ) )
            copy_numpy = copy_t.numpy()

            # loop over pixels and set values in copy
            #loop through all pixels
            for rows in range(height):
                for cols in range(width):
                    firstpix = first_t[ib,0,rows,cols].item()
                    secondpix = second_t[ib,0,rows,cols].item()
                    diff = abs(firstpix - secondpix)
                    copy_numpy[ib,0,rows,cols] = diff

            copy_t = torch.from_numpy(copy_numpy)

            # save created outputs
            diff_out = larcv.as_image2d_meta(copy_t[ib,0,:,:], outmeta)
            ev_out_diff.append(diff_out)


            outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            outputdata.save_entry()
            ientry += 1
            afterloop = time.time()


    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()
