# UB ResNet Deploy scripts

Scripts to deploy and process images go here.

* run_ubresnet_precropped.py: Process larcv files containeing pre-cropped images. Useful for per-subimage analysis.
* run_ubresnet_wholeview.py: (not created yet)
* run_dllee_wholeview.py: (not created yet) Process larcv files. Images only evaluated inside CROI. Details of splitting and merging are meant to reproduce the DLLEE SSNet code.


### run_ubresnet_precropped.py

process images through UBResNet model. Assumes that input images have been precropped.
if you need to process entire plane-views, use run_ubresnet_wholeview.py
right now supports LArCV1 inputs only. LArCV2 support to come.

```
usage: run_ubresnet_precropped.py [-h] -i INPUT -o OUTPUT -c CHECKPOINT -p
                                  PLANE -t TREENAME [-d DEVICE]
                                  [-g CHKPT_GPUID] [-b BATCHSIZE] [-n NEVENTS]
                                  [-v]

Process cropped-image views through Ubresnet.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        location of input larcv file
  -o OUTPUT, --output OUTPUT
                        location of output larcv file
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        location of model checkpoint file
  -p PLANE, --plane PLANE
                        MicroBooNE Plane ID (0=U,1=V,2=Y)
  -t TREENAME, --treename TREENAME
                        Name of tree in ROOT file containing images. e.g.
                        'wire' for 'image2d_wire_tree' in file.
  -d DEVICE, --device DEVICE
                        device to use. e.g. "cpu" or "cuda:0" for gpuid=0
  -g CHKPT_GPUID, --chkpt-gpuid CHKPT_GPUID
                        GPUID used in checkpoint
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -n NEVENTS, --nevents NEVENTS
                        process number of events (-1=all)
  -v, --verbose         verbose output

```

