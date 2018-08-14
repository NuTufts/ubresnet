# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv

# pytorch
import torch

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from infill_funcs import load_model

# larflow
if "LARCVDATASET_BASEDIR" in os.environ:
    sys.path.append(os.environ["LARCVDATASET_BASEDIR"])
else:
    sys.path.append("../pytorch-larflow/larcvdataset") # default location
from larcvdataset import LArCVDataset


def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    """ we can just use the normal larcvdataset"""

    larcvdataset_config="""ThreadProcessor: {
      Verbosity:3
      NumThreads: 2
      NumBatchStorage: 2
      RandomAccess: false
      InputFiles: ["test_crops.root"]
      ProcessName: ["target_valid","wire_valid","weights_valid"]
      ProcessType: ["BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D"]
      ProcessList: {
        target_valid: {
          Verbosity:3
          ImageProducer: "Target"
          Channels: [2]
          EnableMirror: false
        }
        wire_valid: {
          Verbosity:3
          ImageProducer: "wire"
          Channels: [2]
          EnableMirror: false
        }
        weights_valid: {
          Verbosity:3
          ImageProducer: "Weights"
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
        input_larcv_filename = "test_crops.root" # test cropped image file
        output_larcv_filename = "output_infill.root"
        #checkpoint_data = "checkpoint_fullres_bigsample_11000th_gpu3.tar"
        checkpoint_data = "/mnt/disk1/nutufts/kmason/ubresnet/training/infill_080618_best.tar"
        batch_size = 1
        gpuid = 0
        checkpoint_gpuid = 0
        verbose = True
        nprocess_events = 50

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=512
    height=832

    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
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

        tbatch = time.time()

        tdata = time.time()
        data = inputdata[0]

        nimgs = batch_size
        tdata = time.time()-tdata
        timing["++load_larcv_data"] += tdata
        if verbose:
            print "time to get images: ",tdata," secs"

        if verbose:
            print "number of images in whole-view split: ",nimgs

        # get input adc images
        talloc = time.time()
        source_t = torch.from_numpy( data["wire_valid"].reshape( (batch_size,1,width,height) ) ) # source image ADC
        source_t = source_t.to(device=torch.device("cuda:%d"%(gpuid)))
        
        # save a copy of input
        ev_out_wire = outputdata.get_data("image2d", "wire")
        wire_t = source_t.detach().cpu().numpy()
        
        #get weights to save them
        weights_t = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,width,height) ) )
        weights_t = weights_t.to(device=torch.device("cuda:%d"%(gpuid)))
        ev_out_weights = outputdata.get_data("image2d","weights")
        weight_t = weights_t.detach().cpu().numpy()
 
        talloc = time.time()-talloc
        timing["++alloc_arrays"] += talloc
        if verbose:
            print "time to allocate memory (and copy) for numpy arrays: ",talloc,"secs"

        # run model
        trun = time.time()
        pred_labels = model.forward( source_t) #target_t
        trun = time.time()-trun
        timing["++run_model"] += trun
        if verbose:
            print "time to run model: ",trun," secs"

        # turn pred_labels back into larcv
        tsave = time.time()

        # get predictions from gpu
        labels_np = pred_labels.detach().cpu().numpy().astype(np.float32)
        labels_np = 10**labels_np
        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch
                break
            #evtinfo   = data["event_base"][ib,:] #Change event_ids to just id?
            #outmeta   = data["source_test"][ib].meta()
            inputmeta.read_entry(ientry)
            ev_meta   = inputmeta.get_data("image2d","wire")
            # if ev_meta.run()!=evtinfo.run() or ev_meta.subrun()!=evtinfo.subrun() or ev_meta.event()!=evtinfo.event():
                # raise RuntimeError("(run,subrun,event) for evtinfo and ev_meta do not match!")
            outmeta   = ev_meta.image2d_array()[2].meta()
            
            img_slice0 = labels_np[ib,0,:,:]
            nofill_lcv  = larcv.as_image2d_meta( img_slice0, outmeta )
            ev_out    = outputdata.get_data("image2d","nofill")
            ev_out.append( nofill_lcv )
            
            img_slice1 = labels_np[ib,1,:,:]
            fill_lcv = larcv.as_image2d_meta( img_slice1, outmeta )
            ev_out = outputdata.get_data("image2d", "fill")
            ev_out.append( fill_lcv)
           
            wire_slice=wire_t[ib,0,:,:]
            wire_out = larcv.as_image2d_meta(wire_slice,outmeta)
            ev_out_wire.append( wire_out )
             
            weight_slice=weight_t[ib,0,:,:]
            weights_out = larcv.as_image2d_meta(weight_slice,outmeta)
            ev_out_weights.append( weights_out )
            
            outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            #outputdata.set_id(1,1,ibatch*batch_size+ib)
            outputdata.save_entry()
            ientry += 1

        tsave = time.time()-tsave
        timing["++save_output"] += tsave

        tbatch = time.time()-tbatch
        if verbose:
            print "time for batch: ",tbatch,"secs"
        timing["+batch"] += tbatch

    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()

    print "DONE."

    ttotal = time.time()-ttotal
    timing["total"] = ttotal

    print "------ TIMING ---------"
    for k,v in timing.items():
        print k,": ",v," (per event: ",v/float(nevts)," secs)"
