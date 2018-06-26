# Caffe Tools/Comparisons

MicroBooNE currently uses an SSNet model built using Caffe1 and LArCV1.

*NOTE:* This uses LArCV1. If you want to use the code in these modules, you should not setup the environment variables for larcv2, which is setup in the head of the repo.  If you have already setup the environment variables for larcv2, you should start a new shell and only load up the environmen variables for larcv1.

A submodule for LArCV1 is provided in this folder.  When first checking out the code, run `first_setup.sh`.  This will build larcv1.  When re-setting up the code, you only need to set the environment variables with `setenv_caffe_meitner.sh`.

All development will be benchmarked against this model, which is (as of June 2018) being used by the DL LEE group in their analysis.
This model was also used to write the SSNet paper released by MicroBooNE in 2018.

This folder contains scripts to run the model in Caffe1. This is to be used to compare against model translations into pytorch or tensorflow.
Also, any improvements being developed, must show itself to be superior to this version.

## Setting up Caffe

We, of course, need a copy of Caffe in order to run the model natively in Caffe. We do not cover the instructions for building Caffe.

To setup the enviroment variables

* on Meitner: `source setenv_caffe_meitner.sh`

## Running

### On precropped image sets

To run on precropped image sets use `run_caffe_precropped.py`. Output:

  ```
  Attaching file output_caffe_precropped.root as _file0...
  (TFile *) 0x35844a0
  root [1] .ls
  TFile**         output_caffe_precropped.root
   TFile*         output_caffe_precropped.root
    KEY: TTree    image2d_ssnet_plane0_tree;9     ssnet_plane0 tree
    KEY: TTree    image2d_ssnet_plane1_tree;9     ssnet_plane1 tree
    KEY: TTree    image2d_ssnet_plane2_tree;8     ssnet_plane2 tree
  ```

For each event in the trees, `ssnet_plane[0-2]_tree`, an image with each classes' scores per pixel is saved.

* Index 0: background
* Index 1: track
* Index 2: shower

The scores per pixel should add to 1. Note that we have not applied a charge threshold on the output.
