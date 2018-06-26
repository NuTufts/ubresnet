#!/bin/bash

export UBRESNET_BASEDIR=$PWD
export UBRESNET_MODELDIR=$(UBRESNET_BASEDIR)/models

# OPENCV
export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include
export USE_OPENCV=1

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup larcv environment variabls
cd ../larcv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../larcvdataset
source setenv.sh

# setup post-processor
export UBRESNET_POST_LIBDIR=${UBRESNET_BASEDIR}/postprocessor/lib
[[ ":$LD_LIBRARY_PATH:" != *":${UBRESNET_POST_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${UBRESNET_POST_LIBDIR}:${LD_LIBRARY_PATH}"

# return to top-level directory
cd ../
