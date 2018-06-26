import os,sys,time
from collections import OrderedDict
os.environ["GLOG_minloglevel"] = "1"
import caffe
import numpy as np

from larcv import larcv

# Number of Planes (should be three at most)
NPLANES = 3

# GPUID
GPUID = 0
caffe.set_mode_gpu()
caffe.set_device(GPUID)

# SSNET MODEL USED IN 2018 MICROBOONE PAPER
MODEL_PROTOTXT = "../models/dllee_ssnet2018.prototxt"
WEIGHTS = [ "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane0_iter_75500.caffemodel",
            "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane1_iter_65500.caffemodel",
            "../weights/ssnet2018caffe/segmentation_pixelwise_ikey_plane2_iter_68000.caffemodel" ]


# we need to load all three planes
nets = [ caffe.Net( MODEL_PROTOTXT, WEIGHTS[x], caffe.TEST ) for x in range(0,NPLANES) ]

# Input IO
# A larcv 1 file
#INPUT_FILE="/media/hdd1/larbys/ssnet_dllee_trainingdata/test_1e1p_lowE_00.root"
INPUT_FILE="/media/hdd1/larbys/ssnet_dllee_trainingdata/test_1e1p_lowE_00.root"

io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( INPUT_FILE )
io.initialize()

nentries = io.get_n_entries()

nentries = 3 # for debug

BATCHSIZE=1
NCLASSES = 3
WIDTH=None
HEIGHT=None

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

print "Num entries to process: ",nentries

for ientry in range(nentries):

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

    timer["totentry"] += time.time()-t_entry
        

timer["total"] = time.time()-timer["total"]


for k,i in timer.items():
    print k,": ",i," secs (%.2f sec/event)"%(i/float(nentries))

print "[ENTER to EXIT]"
raw_input()
