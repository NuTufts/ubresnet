#!/bin/bash

source setenv_caffe_meitner.sh

cd ../larlite
make
cd ../caffe

cd larcv
make -j4
cd ..
