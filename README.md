# MicroBooNE U-RESNET: develpoment files for network for semantic segmentation

This repository contains the code for developing the uresnet semantic segmentation model for MicroBooNE data analysis

## Dependencies

### Not included in repo

* ROOT (6.12/04 known to work)
* opencv (3.2.0 known to work)
* pytorch (0.4)
* numpy (1.14.03 known to work)
* tensorboardX (from [here](https://github.com/lanpa/tensorboard-pytorch))
* tensorboard (1.3.0c)
* cuda (9.1 known to work)
* (to do: add missing)

### Included as submodules

* LArCV2 (tufts_ub branch): library for representing LArTPC data as images along with meta-data. Also, provides IO.
* larlite: classes for meta-data. Also provides access to constants for the UB detector geometry and LAr physics
* larcvdataset: wrapper class providing interface to images stored in the larcv format. converts data into numpy arrays for use in pytorch

## Setup

### First-time setup

* clone this repository: `git clone https://github.com/NuTufts/ubresnet ubresnet`
* setup the submodules, configure environment variables, and build: `source first_setup.sh`
* if you plan to modify any of the submodules, you will need go to the head branch for each submodule. use: `source goto_head_of_submodules.sh`

### Each time you start a new shell and want to use the code
* setup environment variables via `source configure.sh`
* if you made a modification to the submodules and want to build, you can use the build script: `source build.sh`. (of course you can porbably just type make in the top directory of the submodule as well.

### Pushing back changes

If you made changes to a submodule, you need to check in that code and then check in the new commit hash of the submodule to this repo.

Say you made a change to larcv. (Same instructions basically for all submodules).

* First make sure you are not in a DEATCHED_HEAD state)

      git branch
        develop
	  * tufts_ub
	
* If it says detached head, go back to head of this repo and run `source goto_head_of_submodules.sh` and come back
* stage your commits and then push

      git add [[some file you edited]]
      git commit -m "[[short description of change]]"
      git push
* go back to head of this repo and commit the updated submodule (in this example `larcv`) to this repo

      cd ..
      git add larcv
      git commit -m "[[which submodule you updated]]"
      git push


## Contents

(not including submodules) (many of the modules are not implemented yet)

* models: different version of models
* dataprep: scripts to make input and truth images from larsoft files and then prepare crops for training
* training: training scripts
* deploy: take trained models and process files
* ana: analysis scripts for processed test files
* caffe: tools for running old caffe model -- for performance comparisons. requires caffe1.
