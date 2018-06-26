#!/bin/bash

export CAFFE_ROOT=/home/twongj01/software/caffe/build-gpu/install
export CAFFE_BINDIR=${CAFFE_ROOT}/bin
export CAFFE_LIBDIR=${CAFFE_ROOT}/lib
export CAFFE_PYTHONDIR=${CAFFE_ROOT}/python

[[ ":$PATH:" != *":${CAFFE_BINDIR}:"* ]] && PATH="${CAFFE_BINDIR}:${PATH}"
[[ ":$LD_LIBRARY_PATH:" != *":${CAFFE_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${CAFFE_LIBDIR}:${LD_LIBRARY_PATH}"
if [ -z ${PYTHONPATH+x} ]; then
    PYTHONPATH=.:${CAFFE_PYTHONDIR};
else
    [[ ":$PYTHONPATH:" != *":${CAFFE_PYTHONDIR}:"* ]] && PYTHONPATH="${CAFFE_PYTHONDIR}:${PYTHONPATH}";
fi

cd larcv
source configure.sh
cd ..
