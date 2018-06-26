#!/usr/bin/env python
import os,sys,time
import argparse
from collections import OrderedDict
os.environ["GLOG_minloglevel"] = "1"
import caffe
import numpy as np

from larcv import larcv

# DEFINE ARGUMENTS DEFINTION/PARSER
arg_parser = argparse.ArgumentParser(description='Process precropped views through UB ResNet.')
arg_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
arg_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
arg_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on")
arg_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
arg_parser.add_argument( "-n", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)")
arg_parser.add_argument( "-s", "--start",        default=0,     type=int, help="entry number to start on")
arg_parser.add_argument( "-d", "--debug",        action="store_true",    help="run debug configuration (hard-coded in script)")
arg_parser.add_argument( "-w0", "--weights-p0",   default="__not__", type=str, help="set weight file (usually .caffemodel) for plane 0")
arg_parser.add_argument( "-w1", "--weights-p1",   default="__not__", type=str, help="set weight file (usually .caffemodel) for plane 1")
arg_parser.add_argument( "-w2", "--weights-p2",   default="__not__", type=str, help="set weight file (usually .caffemodel) for plane 2")

if __name__ == "__main__":

    # SSNET MODEL USED IN 2018 MICROBOONE PAPER
    MODEL_PROTOTXT = "../models/dllee_ssnet2018.prototxt"
    WEIGHTS = [ "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane0_iter_75500.caffemodel",
                "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane1_iter_65500.caffemodel",
                "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane2_iter_68000.caffemodel" ]
    
    # Number of Planes (should be three at most)
    NPLANES = 3

    # Tensor dimension sizes
    BATCHSIZE=1
    NCLASSES = 3
    WIDTH=None  # set later
    HEIGHT=None # set later
    
    
    # parse the arguments
    args = arg_parser.parse_args(sys.argv[1:])
    if not args.debug:
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        gpuid                 = args.gpuid
        verbose               = args.verbose
        nprocess_events       = args.nevents
        start_entry           = args.start
        if args.weights_p0 != "":
            WEIGHTS[0] = args.weights_p0
        if args.weights_p1 != "":
            WEIGHTS[1] = args.weights_p1
        if args.weights_p2 != "":
            WEIGHTS[2] = args.weights_p2
    else:

        # for testing (use -d flag) to avoid having to set arguments
        input_larcv_filename  = "/media/hdd1/larbys/ssnet_dllee_trainingdata/test_1e1p_lowE_00.root" # on meitner
        output_larcv_filename = "output_caffe_precropped.root"
        gpuid = 0
        verbose = False
        nprocess_events = 100
        start_entry = 0


    # SET THE GPUID
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)


    # we need to load all three planes
    nets = [ caffe.Net( MODEL_PROTOTXT, WEIGHTS[x], caffe.TEST ) for x in range(0,NPLANES) ]

    # Input IO
    # A larcv 1 file
    io = larcv.IOManager( larcv.IOManager.kREAD )
    io.add_in_file( input_larcv_filename )
    io.initialize()

    # Output IO
    out = larcv.IOManager( larcv.IOManager.kWRITE )
    out.set_out_file( output_larcv_filename )
    out.initialize()

    nentries = io.get_n_entries()

    # place holders for numpy arrays
    input_np = None
    output_np = None

    # timers
    timer = OrderedDict()
    timer["total"]     = 0.0
    timer["totentry"]  = 0.0
    timer["readentry"] = 0.0
    timer["loadblob"]  = 0.0
    timer["forward"]   = 0.0
    timer["copyout"]   = 0.0
    timer["writeout"]  = 0.0
    
    timer["total"] = time.time()


    if start_entry+nprocess_events>nentries:
        nprocess_events = nentries-start_entry

    print "Num entires in the file: ",nentries
    print "Num entries to process: ",nprocess_events
    print "Starting entry: ",start_entry        

    for ientry in range(start_entry,start_entry+nprocess_events):

        if ientry<10 or ientry%100==0:
            print "entry ",ientry
    
        t_entry = time.time()
    
        # Get entry data
        t_read = time.time()
        io.read_entry(ientry)
        event_image_container = io.get_data(larcv.kProductImage2D, "wire")
        img_v = event_image_container.Image2DArray()
        timer["readentry"] += time.time()-t_read

        # need to figure out dimension, usually on the first event processed    
        if WIDTH is None or HEIGHT is None:
            # note, image is transposed
            HEIGHT = img_v[0].meta().cols()
            WIDTH  = img_v[0].meta().rows()

            input_np  = [np.zeros( (BATCHSIZE,1,HEIGHT,WIDTH), dtype=np.float32 ) for x in range(0,NPLANES)]
            output_np = [np.zeros( (BATCHSIZE,NCLASSES,HEIGHT,WIDTH), dtype=np.float32 ) for x in range(0,NPLANES)]

        # copy image data into numpy array
        t_blob = time.time()
        for p in range(0,NPLANES):
            input_np[p][0,0,:] = larcv.as_ndarray( img_v[p] )[:]
            nets[p].blobs['data'].reshape( *input_np[p].shape )
            nets[p].blobs['data'].data[...] = input_np[p]
        timer["loadblob"] = time.time()-t_blob

        # run the net
        t_forward = time.time()
        for p in range(0,NPLANES):
            nets[p].forward()
        timer["forward"] += time.time()-t_forward

        # retrive the data
        t_out = time.time()
        for p in range(0,NPLANES):
            output_np[p][0,:,:,:] = nets[p].blobs['softmax'].data[:]
        timer["copyout"]  = time.time()-t_out

        # write to disk
        t_disk = time.time()
        event_ssnet_containers = [ out.get_data( larcv.kProductImage2D, "ssnet_plane%d"%(p) ) for p in range(NPLANES) ]
        for p in range(NPLANES):
            for c in range(NCLASSES):
                event_ssnet_containers[p].Append( larcv.as_image2d_meta( output_np[p][0,c,:], img_v[p].meta() ) )
        out.set_id( event_image_container.run(), event_image_container.subrun(), event_image_container.event() )
        out.save_entry()
        timer["writeout"] += time.time()-t_disk
    
        timer["totentry"] += time.time()-t_entry
        
    print "End of entry loop"
    print "Finalize output"
    out.finalize()
    timer["total"] = time.time()-timer["total"]

    print "Timing for different steps"
    print "--------------------------"
    for k,i in timer.items():
        print k,": ",i," secs (%.2f sec/event)"%(i/float(nprocess_events))

    if args.debug:
        print "[ENTER to EXIT]"
        raw_input()
