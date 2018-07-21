#!/bin/bash

## Script to build components of ubresnet repo inside the container on the grid
## you only need to do this the first time you clone the ubresnet container
## onto the grid or if you change any source code in larlite,larcv

## NOTE: it is advisable to run this on an interactive batch node
## AND NOT the login node
## To start an interactive node, run
## srun --pty -p batch bash

UBRESNET_DIR_INCONTAINER=$1
# example: /cluster/kappa/wongjiradlab/twongj01/ubresnet/

# pytorch 0.3, larcv1 container
CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-larbys-pytorch/singularity-pytorch-0.3-larcv2-nvidia384.66.img

echo "UBRESNET DIR: ${UBRESNET_DIR_INCONTAINER}"

module load singularity
singularity exec ${CONTAINER} bash -c "cd ${UBRESNET_DIR_INCONTAINER}/training/grid_scripts && source /usr/local/root/release/bin/thisroot.sh && source larcv1_build.sh ${UBRESNET_DIR_INCONTAINER}"