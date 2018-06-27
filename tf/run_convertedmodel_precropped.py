import os,sys,time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

from larcv import larcv

if __name__ == "__main__":

    # PARAMS
    NPLANES = 1
    
    # Load the model
    tload = time.time()
    predict_fn = predictor.from_saved_model("converted_models/SaveModelSSNet")
    tload = time.time()-tload
    print "Time to load TF predictor: ",tload,"secs"

    # Load the data (LArCV1)
    # for testing (use -d flag) to avoid having to set arguments
    input_larcv_filename  = "/media/hdd1/larbys/ssnet_dllee_trainingdata/test_1e1p_lowE_00.root" # on meitner
    output_larcv_filename = "output_tfconverted_precropped.root"
    gpuid = 0
    verbose = False
    nprocess_events = 10
    start_entry = 0
    BATCHSIZE=1
    NCLASSES=3

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

    # place holders for numpy arrays and shape
    input_np = None
    output_np = None
    WIDTH = None
    HEIGHT = None

    # timers
    timer = OrderedDict()
    timer["total"]     = 0.0
    timer["totentry"]  = 0.0
    timer["readentry"] = 0.0
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

            input_np  = [np.zeros( (BATCHSIZE,HEIGHT,WIDTH,1), dtype=np.float32 ) for x in range(0,NPLANES)]

        # copy image data into numpy array
        t_blob = time.time()
        for p in range(0,NPLANES):
            input_np[p][0,:,:,0] = larcv.as_ndarray( img_v[p] )[:]
            
            
        timer["loadblob"] = time.time()-t_blob

        # run the net
        t_forward = time.time()
        output_np = [ predict_fn( {"uplane":input_np[p] } )["pred"] for p in range(NPLANES) ]
        timer["forward"] += time.time()-t_forward

        # write to disk
        t_disk = time.time()
        event_ssnet_containers = [ out.get_data( larcv.kProductImage2D, "ssnet_plane%d"%(p) ) for p in range(NPLANES) ]
        for p in range(NPLANES):
            for c in range(NCLASSES):
                event_ssnet_containers[p].Append( larcv.as_image2d_meta( output_np[p][0,:,:,c], img_v[p].meta() ) )
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

    #if args.debug:
    #    print "[ENTER to EXIT]"
    #    raw_input()




