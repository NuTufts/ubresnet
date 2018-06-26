# Caffe Tools/Comparisons

MicroBooNE currently uses an SSNet model built using Caffe1 and LArCV1.

*NOTE:* This uses LArCV1. If you want to use the code in these modules, you cannot setup larcv2, which is setup in the head of the repo.
A submodule for LArCV1 is provided in this folder.  

All development will be benchmarked against this model, which is (as of June 2018) being used by the DL LEE group in their analysis.
This model was also used to write the SSNet paper released by MicroBooNE in 2018.

This folder contains scripts to run the model in Caffe1. This is to be used to compare against model translations into pytorch or tensorflow.
Also, any improvements being developed, must show itself to be superior to this version.

## Caffe

We, of course, need a copy of Caffe in order to run the model natively in Caffe.

To setup the enviroment variables

* on Meitner: `source setenv_caffe_meitner.sh`