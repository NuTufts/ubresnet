import os,sys
import numpy as np

# Import LArCV1
from larcv import larcv

# opencv
import cv2 as cv

# ------------------------------------------------------------------------
# This script is meant to quickly check the output differences between
# the 2018 MicroBooNE SSNet model implemented in Caffe1 and Tensorflow
# ------------------------------------------------------------------------

NPLANES  = 1
NCLASSES = 3
ADC_THRESHOLD = 10.0

# Load Caffe scores
io_caffe = larcv.IOManager( larcv.IOManager.kREAD )
io_caffe.add_in_file( "../caffe/output_caffe_precropped.root" ) # ssnet scores
io_caffe.add_in_file( "/media/hdd1/larbys/ssnet_dllee_trainingdata/test_1e1p_lowE_00.root" ) # input ADC values
io_caffe.initialize()

# Load TF scores
io_tf = larcv.IOManager( larcv.IOManager.kREAD )
io_tf.add_in_file( "output_tfconverted_precropped.root" )
io_tf.initialize()


n_caffe = io_caffe.get_n_entries()
n_tf    = io_tf.get_n_entries()

nevents = np.minimum( n_caffe, n_tf )

HEIGHT = None
WIDTH  = None
input_np = None
caffe_np = None
tfout_np = None

nevents = 5

for ientry in range(nevents):

    # load entry
    io_caffe.read_entry( ientry )
    io_tf.read_entry( ientry )

    # get event container for image data and labels
    ev_adc_caffe   = io_caffe.get_data( larcv.kProductImage2D, "wire" )
    ev_ssnet_caffe = [ io_caffe.get_data( larcv.kProductImage2D, "ssnet_plane%d"%(p) ) for p in range(NPLANES) ]
    ev_ssnet_tf    = [ io_tf.get_data(    larcv.kProductImage2D, "ssnet_plane%d"%(p) ) for p in range(NPLANES) ]

    # get c++ vector<larcv::Image2D>, some for each plane
    adc_v = ev_adc_caffe.Image2DArray()
    caffe_v = [ ev_ssnet_caffe[p].Image2DArray() for p in range(NPLANES) ]
    tfout_v = [ ev_ssnet_tf[p].Image2DArray() for p in range(NPLANES) ]

    # set image dimensions
    if HEIGHT is None or WIDTH is None:
        meta = ev_adc_caffe.Image2DArray()[0].meta()
        # conversion to numpy transposes image
        WIDTH  = meta.rows()
        HEIGHT = meta.cols()

        input_np = [ np.zeros( (HEIGHT,WIDTH) ) for p in range(NPLANES) ]
        caffe_np = [ np.zeros( (NCLASSES,HEIGHT,WIDTH) ) for p in range(NPLANES) ]
        tfout_np = [ np.zeros( (NCLASSES,HEIGHT,WIDTH) ) for p in range(NPLANES) ]

    # xfer image data to np for analysis
    for p in range(NPLANES):
        # turn ADC image into numpy
        input_np[p][:] = np.transpose( larcv.as_ndarray( adc_v[p] ), (1,0) )

        # convert outputs to numpy
        for c in range(NCLASSES):
            caffe_np[p][c,:,:] = larcv.as_ndarray( caffe_v[p][c] )
            tfout_np[p][c,:,:] = larcv.as_ndarray( tfout_v[p][c] )

            # we blank out pixels with sub-threshold values
            #caffe_np[p][c,:][ input_np[p]<=ADC_THRESHOLD ] = 0.0
            #tfout_np[p][c,:][ input_np[p]<=ADC_THRESHOLD ] = 0.0
        #print "TF OUT p=%d"%(p)
        #print tfout_np[p][2,:][ input_np[p]>ADC_THRESHOLD ][100:110]
        #print "CAFFE OUT p=%d"%(p)        
        #print caffe_np[p][2,:][ input_np[p]>ADC_THRESHOLD ][100:110]

        nonzero = (input_np[p]>ADC_THRESHOLD).sum()
        
        # compare diff
        diff_np = np.fabs( tfout_np[p] - caffe_np[p] )

        ave_diff_per_class = [ diff_np[c,:].sum()/float(nonzero) for c in range(NCLASSES) ]

        # average difference for each type
        print "plane 0, ave. diff: ",ave_diff_per_class

        print caffe_np[p][0,:].sum()

        # dump cv image
        input_np[p][ input_np[p]<0 ] = 0
        input_np[p][ input_np[p]>100 ] = 100.0
        input_np[p] *= 254.0/100.0
        input_tmp = cv.applyColorMap( input_np[p].astype(np.int32).astype(np.uint8), cv.COLORMAP_WINTER )
        cv.imwrite( "adc_entry%d_p%d.png"%(ientry,p), input_tmp )
        
        caffe_tmp = np.zeros( (HEIGHT,WIDTH,3), dtype=np.uint8 )
        for c in range(NCLASSES):
            caffe_tmp[:,:,c] = (255.0*caffe_np[p][c,:,:]).astype( dtype=np.uint8 )
        cv.imwrite( "caffe_entry%d_p%d_c1.png"%(ientry,p), caffe_tmp )

        tfout_tmp = np.zeros( (HEIGHT,WIDTH,3), dtype=np.uint8 )
        for c in range(NCLASSES):
            tfout_tmp[:,:,c] = (255.0*tfout_np[p][c,:,:]).astype( dtype=np.uint8 )
        cv.imwrite( "tfout_entry%d_p%d_c1.png"%(ientry,p), tfout_tmp )

        diff_tmp = np.zeros( (HEIGHT,WIDTH,3), dtype=np.uint8 )
        for c in range(NCLASSES):
            diff_tmp[:,:,c] = (255.0*diff_np[c,:,:]).astype( dtype=np.uint8 )
        cv.imwrite( "diff_entry%d_p%d_c1.png"%(ientry,p), diff_tmp )

    #break

    

    
        
    
