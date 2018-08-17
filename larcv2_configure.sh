#!/bin/bash

export UBRESNET_BASEDIR=$PWD
export UBRESNET_MODELDIR=${UBRESNET_BASEDIR}/models
export LARCV_VERSION=2

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

# return to top-level directory
cd ../

# add model dir to python path

