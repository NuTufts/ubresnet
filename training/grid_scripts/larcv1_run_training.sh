#!/bin/bash

# note that this script assumes it is inside the container

# repository directory inside container
# example: /cluster/kappa/wongjiradlab/twongj01/ubresnet
repodir=$1

# working directory where job files and tensorboard files will be saved
# this can be anywhere inside your personal directory in the wongjiradlab space
# example: /cluster/kappa/wongjirad/twongjirad/training/workdir (not made automatically)
workdir=$2

# training python script
# example: /cluster/kappa/wongjirad/twongjirad/training/train_ubresnet2018_wlarcv1.py
training_pyscript=$3

# gpuid
GPUID=${SLURM_ARRAY_TASK_ID}

echo "REPODIR (in container): ${repodir}"
echo "WORKDIR (in container): ${workdir}"
echo "TRAINING SCRIPT (in container): ${training_pyscript}"
echo "GPUID: ${GPUID}"

# data loader config files

# setup the container
source ${repodir}/training/grid_scripts/larcv1_setup_container.sh ${repodir}

# go to the working directory
cd ${workdir}

# make a folder for this job
jobdir=`printf larcv1_training_job%d_gpuid%d ${SLURM_JOB_ID} ${GPUID}`
mkdir -p ${jobdir}

# copy training script 
# note, if you need to copy more python files that the training script uses, do that here
trainscript=`printf run_training_job%d_gpuid%d.py ${SLURM_JOB_ID} ${GPUID}`
cp ${training_pyscript} ${jobdir}/${trainscript}

# go into jobdir
cd ${jobdir}

# define log file name
logfile=`printf log_larcv1_training_job%d_gpuid%d.txt ${SLURM_JOB_ID} ${GPUID}` 

export PYTHONPATH=${repodir}:${PYTHONPATH}
# run the job
python ${trainscript} ${GPUID} >& ${logfile}
