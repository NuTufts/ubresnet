# UBResNet grid scripts

Scripts for running and training on the Tufts cluster.

On the cluster, we have to run our code inside a singualarity container which contains `pytorch` and `ROOT`. 
The container provides the dependencies we need to compile our versions of `larlite` and `larcv` that we have included as submodules inside the repository.

## Setup

To set things up we need to build `larlite` and `larcv` while inside the container so it uses the version of ROOT inside the container.

To do this:

* go to your folder in the `wongjiradlab` network storage area:

      cd /cluster/tufts/wongjiradlab/[your-user-name]

* clone the `ubresnet` repository

      git clone https://github.com/NuTufts/ubresnet.git

* note the *repository directory*. following the above it's now

      /cluster/tufts/wongjiradlab/[user-name]/ubresnet

  You are free to have changed this location to something else. 

* go to the `[repo. dir]/training/grid_scripts` folder and launch the build script

     cd /cluster/tufts/wongjiradlab/[user-name]/ubresnet/training/grid_scripts/
     source 

      
## Running


### stage your data

First thing to do is to stage your data on the node you are going to use.  
Staging means transferring the data from the network storage area, `/cluster/tufts/wongjiradlab`, to the local harddrive of the GPU node.
This means that the data will be read locally on the machine, instead of transferred via network (admittedly a fast Infiniband network -- 2.5 Gigabits/s).
But loading data is often the slowest part of training, so we want to reduce this as much as possible.

An example of a slurm job script is provided in `training/grid_scripts/stage_dllee_ssnet_training_data.sh`.
This script requests that a job be run on a certain node, here `PGPU03` which is the `wongjiradlaba` GPU machine.
It copies data from the `wongjiradlab` area to `/tmp`.
All jobs created by slurm can access the `/tmp/` directory of the nodes.
Also, the singularity container automatically binds the `/tmp` directory into the container. 
This makes `/tmp` a good place to park data.

This script is a `slurm` batch submission script.  
You'll notice the `#SBATCH` lines in the begining. 
These are comment lines to the bash shell.
But they are also lines read in by the slurm batch submission program to configure the job request.

You can (if needed) make a copy and modify this script to

  * transfer the data you want. this means modifying the `rsync` lines.
  * make sure it goes to the right node. this means modifying the `#SBATCH --nodelist=` argument.

Launch the script

  sbatch stage_dllee_ssnet_training_data.sh


Note, you only have to do this once the first time you start jobs. 
Or after awhile since you last use these files as `/tmp` gets cleaned out periodically if a file is not used.

### edit the training python script

The python script `training/grid_scripts/train_ubresnet_wlarcv_tuftsgrid.py` is a template for your to configure your training job.

Note the following:

In the area where the input data is setup (where the ThreadDatumFiller configs are being defined/written) you should point to the data.
By default in this script, it points to `/tmp` assuming you staged your data.
But it can point to the network as well, and examples of doing that are in the config, but commented out.
Note that inside the container the path to the `wongjiradlab` network drives are different than when you are on the login or worker nodes.

On the login node

    /cluster/tufts/wongjiradlab

On the worker nodes

    /cluster/kappa/90-days-archive/wongjiradlab

Inside the container

    /cluster/kappa/wongjiradlab


This has to do with how the cluster is setup and beyond the control of our group.


Second, if you are going to start from a checkpoint, set

     RESUME_FROM_CHECKPOINT=True

and point to your checkpoint file

     CHECKPOINT_FILE=[path to checkpoint file]

Usually, you can go ahead and point to the file on the `wongjiradlab` network storage area. 
You only read this file once at the beginning, so it is OK if it is read through the network.
Note that this path has to be from *inside the container*.

### edit the script that runs the job on the worker node

You should copy and modify `training/grid_scripts/larcv1_run_training.sh` to meet your needs.

The job of this script is to 

* setup a working directory where your checkpoint files, log file, and tensorboard files get written
* mostly you will need to make sure all the python files your training script needs is in the job directory.  append commands after

      cp ${training_pyscript} ${jobdir}/${trainscript}

  to copy additional files, not just your training python script.


### launch the job

Finally, you can modify, if needed, the script, `sbatch_submit_larcv1_training.sh`, which we use to have slurm configure the job. 
It will setup a worker job on a node we specify and call `larcv1_run_training.sh`.

Things to note are

* what node you want to run on: `--nodelist=pgu03`
* adjust the max time you need: `--time=3-00:00:00`. Note that this is in the form `--time=[days]-[hours]:[minutes]:[secs]`. Allocate enough time for the job to run comfortably. But the more time you ask for, the lower your priority.
* setup the arguments that will go to `larcv1_run_training.sh`: `TRAININGSCRIPT_IN_CONTAINER`, `WORKDIR_IN_CONTAINER`, `REPODIR_IN_CONTAINER`.

To launch it:

    sbatch sbatch_submit_larcv1_training.sh

