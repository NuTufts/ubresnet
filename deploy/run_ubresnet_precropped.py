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
# also, implicitly loads dependencies, pytorch ubresnet model definition
from ubresnet_funcs import load_model

# ubresnet
if "UBRESNET_MODELDIR" in os.environ:
    sys.path.append(os.environ["UBRESNET_MODELDIR"])
else:
    sys.path.append("../models") # default location
from larcvdataset import LArCVDataset


def load_pre_cropped_data( inputfile, configfilename"larcv_dataloader.cfg"=, batchsize=1 ):
    """ since we assume that the input file already has precropped images, we will just use the normal larcvdataset

    inputs
    ------
    inputfile ( str or list of str): Paths to files we want to process.
    larcvdataset_configfile (str): name of the configuraiton file we will create
    batchsize (int, optional): size of a batch. default is 1

    output
    ------
    (written to disk) configuration file named with the value of 'larcvdataset_configfile'
    """

    # first we create a configuration file
    # we substitute in the list of inputfiles

    infile_str = "["
    if type(inputfile) is str:
        infile_str += "\"%s\""%(inputfile)
    elif type(inputfile) is list:
        for n,l in enumerate(inputfile):
            infile_str += "\"%s\""%(inputfile)
            if n+1!=len(inputfile):
                infile_str += ","
    infile_str += "]"            
    
    larcvdataset_config="""ThreadProcessorTest: {
    Verbosity:3
    NumThreads: 2
    NumBatchStorage: 2
    RandomAccess: false
    InputFiles: %s
    ProcessName: ["source_test"]
    ProcessType: ["BatchFillerImage2D"]
    ProcessList: {
      source_test: {
        Verbosity:3
        ImageProducer: "adc"
        Channels: [0]
        EnableMirror: false
      }
     }
    }
    """
    with open(configfilename,'w') as f:
        print >> f,larcvdataset_config%(infile_str)
    iotest = LArCVDataset( configfilename,"ThreadProcessorTest", store_eventids=True)

    return iotest
    

if __name__=="__main__":

    # ARGUMENTS DEFINTION/PARSER
    if len(sys.argv)>1:
        crop_view_parser = argparse.ArgumentParser(description='Process cropped-image views through Ubresnet.')
        crop_view_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
        crop_view_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
        crop_view_parser.add_argument( "-c", "--checkpoint",   required=True, type=str, help="location of model checkpoint file")
        crop_view_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on")
        crop_view_parser.add_argument( "-p", "--chkpt-gpuid",  default=0,     type=int, help="GPUID used in checkpoint")
        crop_view_parser.add_argument( "-b", "--batchsize",    default=2,     type=int, help="batch size" )
        crop_view_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
        crop_view_parser.add_argument( "-n", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)") 

        args = crop_view_parser.parse_args(sys.argv[1:])
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        gpuid                 = args.gpuid
        checkpoint_gpuid      = args.chkpt_gpuid
        batch_size            = args.batchsize
        verbose               = args.verbose
        nprocess_events       = args.nevents
    else:

        # quick for testing
        input_larcv_filename = "ssnet_retrain_cocktail_p03.root" # test cropped image file
        output_larcv_filename = "output_ubresnet.root"
        checkpoint_data = "../weights/checkpoint.58000th.tar"
        batch_size = 1
        gpuid = 0
        checkpoint_gpuid = 0
        verbose = False
        nprocess_events = 10

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=512
    height=512
    
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
        source_t = torch.from_numpy( data["source_test"].reshape( (batch_size,1,width,height) ) ) # source image ADC
        target_t = torch.from_numpy( data["target_test"].reshape( (batch_size,1,width,height) ) ) # target image ADC
        source_t = source_t.to(device=torch.device("cuda:%d"%(gpuid)))
        target_t = target_t.to(device=torch.device("cuda:%d"%(gpuid)))
        
        talloc = time.time()-talloc
        timing["++alloc_arrays"] += talloc
        if verbose:
            print "time to allocate memory (and copy) for numpy arrays: ",talloc,"secs"

        # run model
        trun = time.time()
        pred_flow, pred_visi = model.forward( source_t, target_t )
        trun = time.time()-trun
        timing["++run_model"] += trun
        if verbose:
            print "time to run model: ",trun," secs"            

        # turn pred_flow back into larcv
        tsave = time.time()

        # get predictions from gpu
        flow_np = pred_flow.detach().cpu().numpy().astype(np.float32)

        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch 
                break
            evtinfo   = data["event_ids"][ib]
            #outmeta   = data["source_test"][ib].meta()
            inputmeta.read_entry(ientry)
            ev_meta   = inputmeta.get_data("image2d","adc")
            if ev_meta.run()!=evtinfo.run() or ev_meta.subrun()!=evtinfo.subrun() or ev_meta.event()!=evtinfo.event():
                raise RuntimeError("(run,subrun,event) for evtinfo and ev_meta do not match!")
            outmeta   = ev_meta.image2d_array()[2].meta()
            img_slice = flow_np[ib,0,:,:]
            flow_lcv  = larcv.as_image2d_meta( img_slice, outmeta )
            ev_out    = outputdata.get_data("image2d","ubresnet_y2u")
            ev_out.append( flow_lcv )
            outputdata.set_id( evtinfo.run(), evtinfo.subrun(), evtinfo.event() )
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

    


