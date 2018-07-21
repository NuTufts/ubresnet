#!/bin/bash

repodir_in_container=$1
# example: /cluster/kappa/wongjiradlab/twongj01/ubresnet

startdir=$PWD

# setup CUDA
export PATH=/usr/local/nvidia:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia:${LD_LIBRARY_PATH}

# setup ROOT
source /usr/local/root/release/bin/thisroot.sh

# go to repo dir
cd $repodir

source larcv1_configure.sh

# go back to startdir
cd $startdir