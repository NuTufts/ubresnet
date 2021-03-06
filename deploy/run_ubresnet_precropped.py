# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# -----------------------------------------------------------------------------------------
# run_ubresnet_precropped.py
# ---------------------------
#  process images through UBResNet model. Assumes that input images have been precropped.
#  if you need to process entire plane-views, use run_ubresnet_wholeview.py
#  right now supports LArCV1 inputs only. LArCV2 support to come.
# -----------------------------------------------------------------------------------------
# ARGUMENTS DEFINTION/PARSER
# We have to do it here, first, before ROOT gets loaded.
# Otherwise ROOT process --help command and quits
if len(sys.argv)>1 or True:
    crop_view_parser = argparse.ArgumentParser(description='Process cropped-image views through Ubresnet.')
    crop_view_parser.add_argument( "-i", "--input",        required=True,    type=str, help="location of input larcv file" )
    crop_view_parser.add_argument( "-o", "--output",       required=True,    type=str, help="location of output larcv file" )
    crop_view_parser.add_argument( "-c", "--checkpoint",   required=True,    type=str, help="location of model checkpoint file")
    crop_view_parser.add_argument( "-p", "--plane",        required=True,    type=int, help="MicroBooNE Plane ID (0=U,1=V,2=Y)")
    crop_view_parser.add_argument( "-t", "--treename",     required=True,    type=str, help="Name of tree in ROOT file containing images. e.g. 'wire' for 'image2d_wire_tree' in file.")   
    crop_view_parser.add_argument( "-d", "--device",       default="cuda:0", type=str, help="device to use. e.g. \"cpu\" or \"cuda:0\" for gpuid=0")
    crop_view_parser.add_argument( "-g", "--chkpt-gpuid",  default=0,        type=int, help="GPUID used in checkpoint")
    crop_view_parser.add_argument( "-b", "--batchsize",    default=2,        type=int, help="batch size" )
    crop_view_parser.add_argument( "-n", "--nevents",      default=-1,       type=int, help="process number of events (-1=all)")
    crop_view_parser.add_argument( "-v", "--verbose",      action="store_true",        help="verbose output")        
    
    args = crop_view_parser.parse_args(sys.argv[1:])
    input_larcv_filename  = args.input
    output_larcv_filename = args.output
    checkpoint_data       = args.checkpoint
    device                = args.device
    checkpoint_gpuid      = args.chkpt_gpuid
    batch_size            = args.batchsize
    verbose               = args.verbose
    nprocess_events       = args.nevents
    plane                 = args.plane
    treename              = args.treename
else:

    # quick for testing: change 'or True' to 'or False' to use this block
    input_larcv_filename = "ssnet_retrain_cocktail_p03.root" # test cropped image file
    output_larcv_filename = "output_ubresnet.root"
    checkpoint_data = "../weights/checkpoint.58000th.tar"
    batch_size = 1
    gpuid = 0
    checkpoint_gpuid = 0
    verbose = False
    nprocess_events = 10
    plane = 0
    treename = "adc"
# ---------------------------------------------------------------------------------------


# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv

# pytorch
import torch

# util functions
# also, implicitly loads dependencies, pytorch ubresnet model definition
from ubresnet_funcs import load_cosmic_retrain_model

# ubresnet
if "UBRESNET_MODELDIR" in os.environ:
    sys.path.append(os.environ["UBRESNET_MODELDIR"])
else:
    sys.path.append("../models") # default location

from larcvdataset import LArCV1Dataset

    

if __name__=="__main__":

    # load data
    products = [(larcv.kProductImage2D,treename)]
    inputdata = LArCV1Dataset( input_larcv_filename, products, randomize=False )
    
    # load model
    model = load_cosmic_retrain_model( checkpoint_data, device=device, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device(device))
    model.eval()

    # output IOManager
    # we only save flow and visi prediction results
    outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    timing = OrderedDict()
    timing["total"]              = 0.0
    timing["+batch"]             = 0.0
    timing["++load_larcv_data"]  = 0.0
    timing["++alloc_arrays"]     = 0.0
    timing["++run_model"]        = 0.0
    timing["++save_output"]      = 0.0

    ttotal = time.time()

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
        
        tbatch = time.time()
        
        tdata = time.time()
        data = inputdata.getbatch(batch_size)
        
        nimgs = batch_size
        tdata = time.time()-tdata
        timing["++load_larcv_data"] += tdata
        if verbose:
            print "time to get images: ",tdata," secs"
        
        if verbose:
            print "number of images in whole-view split: ",nimgs

        # get input adc images
        talloc = time.time()
        adc_np = data[(larcv.kProductImage2D,treename)][:,plane,:]
        adc_np = adc_np.reshape( (1,1,adc_np.shape[1],adc_np.shape[2]) )
        adc_t  = torch.from_numpy( adc_np )
        adc_t.to(device=torch.device(device))        
        talloc = time.time()-talloc
        timing["++alloc_arrays"] += talloc
        if verbose:
            print "time to allocate memory (and copy) for numpy arrays: ",talloc,"secs"

        # run model
        trun = time.time()
        pred_labels = model.forward( adc_t )
        trun = time.time()-trun
        timing["++run_model"] += trun
        if verbose:
            print "time to run model: ",trun," secs"            

        # turn pred_flow back into larcv
        tsave = time.time()

        # get predictions from gpu
        flow_np = pred_labels.detach().cpu().numpy().astype(np.float32)

        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch 
                break
            evtinfo   = data["_rse_"][ib,:]
            meta_v    = inputdata.getmeta(treename)
            ev_out    = outputdata.get_data(larcv.kProductImage2D,"uburn_plane%d"%(plane))
            nclasses = flow_np.shape[1]
            for c in range(nclasses):
                img_slice = data[(larcv.kProductImage2D,treename)][:,plane,:,:]
                flow_lcv  = larcv.as_image2d_meta( flow_np[ib,c,:,:], meta_v[plane] )
                ev_out.Append( flow_lcv )
            outputdata.set_id( evtinfo[0], evtinfo[1], evtinfo[2] )
            outputdata.save_entry()
            ientry += 1
            
        tsave = time.time()-tsave
        timing["++save_output"] += tsave

        # end of batch
        tbatch = time.time()-tbatch
        if verbose:
            print "time for batch: ",tbatch,"secs"
        timing["+batch"] += tbatch

    # save results
    outputdata.finalize()

    print "DONE."
    
    ttotal = time.time()-ttotal
    timing["total"] = ttotal

    print "------ TIMING ---------"
    for k,v in timing.items():
        print k,": ",v," (per event: ",v/float(nevts)," secs)"

    


