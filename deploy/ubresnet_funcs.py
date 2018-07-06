# larflow_funcs: utility functions used by the various analysis scripts

# python builtins
import os,sys,time
from collections import OrderedDict

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv

# pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# larflow
UBRESNET_MODELDIR=None
if "UBRESNET_MODELDIR" in os.environ: # should have been placed there by configure.sh script
    UBRESNET_MODELDIR=os.environ["UBRESNET_MODELDIR"]
    sys.path.append(UBRESNET_MODELDIR)    
else:
    sys.path.append("../models")

from ub_uresnet import UResNet


# load model with weights from checkpoints

def load_cosmic_retrain_model( checkpointfile, device="cuda:0", checkpointgpu=0 ):

    model = UResNet(inplanes=16,input_channels=1,num_classes=4,showsizes=False)

    # stored parameters know what gpu ID they were on
    # if we change gpuids, we need to force the map here
    map_location=None
    if device!="cuda:%d"%(checkpointgpu):
        map_location={"cuda:%d"%(checkpointgpu):device}
    
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    # check for data_parallel checkpoint which has "module" prefix
    from_data_parallel = False
    for k,v in checkpoint["state_dict"].items():
        if "module." in k:
            from_data_parallel = True
            break

    if from_data_parallel:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        checkpoint["state_dict"] = new_state_dict

    model.load_state_dict(checkpoint["state_dict"])

    return model
