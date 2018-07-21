#!/bin/bash

repodir_in_container=$1
# example: /cluster/kappa/wongjiradlab/twongj01/ubresnet

startdir=$PWD

# setup CUDA
#export PATH=/usr/local/nvidia:${PATH}
#export LD_LIBRARY_PATH=/usr/local/nvidia:${LD_LIBRARY_PATH}

# setup ROOT
source /usr/local/root/release/bin/thisroot.sh

# go to repo dir
cd ${repodir_in_container}

export UBRESNET_BASEDIR=$PWD
export UBRESNET_MODELDIR=${UBRESNET_BASEDIR}/models
export LARCV_VERSION=1

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup larcv environment variabls
cd ../caffe/larcv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../../larcvdataset
source setenv.sh

# go back to startdir
cd $startdir