#!/bin/bash

# load the larcv1 submodule
git submodule init

# setup the environment variables for it
source setenv_caffe_meitner.sh

# build larcv1
source build.sh
