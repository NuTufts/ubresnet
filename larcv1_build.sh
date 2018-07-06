#!/bin/bash

cd larlite
make
cd UserDev/BasicTool
make
cd ../..

cd ../caffe/larcv
make

cd ../..

