#!/bin/bash

source setenv_caffe_meitner.sh

cd larcv
make -j4
cd ..
