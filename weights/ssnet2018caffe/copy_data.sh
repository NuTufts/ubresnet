#!/bin/bash

# Copy data from remote computer.
# Note, it is recommented to use the make_link.sh script if at all possible.

# set username of remote computer
username=`whoami`
#username="override"

# On Meitner
SSNET_WEIGHT_DIR="/media/hdd1/larbys/ssnet_model_weights/"
scp  ${username}@130.64.84.151:${SSNET_WEIGHT_DIR}/segmentation_pixelwise_ikey_plane*.caffemodel .

# On Tufts Cluster

# On UBOONE machines
