#!/bin/bash

# location of container
CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-larbys-pytorch/singularity-pytorch-0.3-larcv2-nvidia384.66.img

# start singularity
module load singularity

# start container
singularity shell --nv $CONTAINER
