#!/bin/bash
#
#SBATCH --job-name=training_ubresnet
#SBATCH --output=log_training_ubresnet.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=2
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03

CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-larbys-pytorch/singularity-larbys-pytorch-0.3-larcv1-nvidia384.66.img
TRAININGSCRIPT_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet/training/grid_scripts/train_ubresnet_wlarcv1_tuftsgrid.py
WORKDIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet/training/workdir
REPODIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet

module load singularity
singularity exec --nv ${CONTAINER} bash -c "mkdir -p ${WORKDIR_IN_CONTAINER} && cd ${WORKDIR_IN_CONTAINER} && source larcv1_run_training.sh ${REPODIR_IN_CONTAINER} ${WORKDIR_IN_CONTAINER} ${TRAININGSCRIPT_IN_CONTAINER}"