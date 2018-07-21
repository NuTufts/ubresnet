#!/bin/bash

#SBATCH --jobname=stage_dllee_data
#SBATCH --output=log_stage_dllee_data.log
#SBATCH --time=1:00:00
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03

#rsync -av --progress /cluster/kappa/90-days-archive/wongjiradlab/larbys/dllee_ssnet_trainingdata/train*.root /tmp/
rsync -av --progress /cluster/kappa/90-days-archive/wongjiradlab/larbys/dllee_ssnet_trainingdata/val.root /tmp/