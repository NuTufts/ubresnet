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
* setup the number of copies you want to run. Each copy runs on a different gpu based on its array number: `--array=0-5'.  For the example command, this creates 6 array jobs, numbered from 0-5, inclusive.  They will launch on gpuids 0-5, inclusive.  To set just one, use something like `--array=2`, where you would run on gpuid=2. Note, there seems to be a high infance mortality rate for the jobs. So you should check if they launched and rerun launch the jobs that failed.

To launch the submission script:

    sbatch sbatch_submit_larcv1_training.sh


### Checking on your jobs

To check on your jobs:

    squeue -u [username]

You should see something like

```
[twongj01@login001 grid_scripts]$ squeue -u twongj01
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
        32592204_4       gpu training twongj01  R       4:55      1 pgpu03
        32592201_3       gpu training twongj01  R       6:55      1 pgpu03
        32592201_5       gpu training twongj01  R       6:55      1 pgpu03
        32592200_2       gpu training twongj01  R       9:08      1 pgpu03
        32592196_0       gpu training twongj01  R      12:52      1 pgpu03
        32592196_1       gpu training twongj01  R      12:52      1 pgpu03
```

The `R` state means the job is running. You can check the standard out of the job in the logfiles in the work directorys. 
For example use

     tail -n 100 larcv1_training_job32592197_gpuid0/log_larcv1_training_job32592197_gpuid0.txt

to see the last 100 lines of the log file.

Or use

     tail -f larcv1_training_job32592197_gpuid0/log_larcv1_training_job32592197_gpuid0.txt

to have the info written to the log file continuously be sent to your terminal so you can watch it.

You can also go into the machine you ran on to check on it.  For example, above the jobs were sent to `pgpu03`.
If there are jobs running on it that you own, you may ssh into it via

    ssh pgpu03

You can check how much memory the jobs are using:

```
[twongj01@login001 grid_scripts]$ ssh pgpu03
Last login: Sat Jul 21 17:17:27 2018 from login001.lux.tufts.edu
[twongj01@pgpu03 ~]$ nvidia-smi 
Sat Jul 21 17:24:26 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.66                 Driver Version: 384.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:04:00.0 Off |                    0 |
| N/A   42C    P0   184W / 250W |  15092MiB / 16276MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-PCIE...  Off  | 00000000:05:00.0 Off |                    0 |
| N/A   41C    P0   118W / 250W |  12747MiB / 16276MiB |     38%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla P100-PCIE...  Off  | 00000000:08:00.0 Off |                    0 |
| N/A   36C    P0    39W / 250W |  12767MiB / 16276MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla P100-PCIE...  Off  | 00000000:09:00.0 Off |                    0 |
| N/A   38C    P0   156W / 250W |  12767MiB / 16276MiB |     20%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla P100-PCIE...  Off  | 00000000:87:00.0 Off |                    0 |
| N/A   39C    P0    37W / 250W |  12747MiB / 16276MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla P100-PCIE...  Off  | 00000000:88:00.0 Off |                    0 |
| N/A   41C    P0    36W / 250W |  12747MiB / 16276MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     53773    C   python                                       12757MiB |
|    0     53778    C   python                                         465MiB |
|    0     54909    C   python                                         465MiB |
|    0     55944    C   python                                         465MiB |
|    0     55945    C   python                                         465MiB |
|    0     57038    C   python                                         465MiB |
|    1     53778    C   python                                       12737MiB |
|    2     54909    C   python                                       12757MiB |
|    3     55945    C   python                                       12757MiB |
|    4     57038    C   python                                       12737MiB |
|    5     55944    C   python                                       12737MiB |
+-----------------------------------------------------------------------------+
```

(some of the GPU-Util is 0%!! This is because the threadfiller isn't very efficient. LArcV2 has a better data loader that is much fast and keeps the GPU occupied. We can try to fix up the loader for LArCV1 -- which is what the example uses.)

You can also check the CPU usage:

```
[twongj01@login001 grid_scripts]$ ssh pgpu03
[twongj01@pgpu03 ~]$ top
top - 17:27:07 up 143 days,  5:00,  1 user,  load average: 5.92, 5.57, 3.88
Tasks: 1847 total,   3 running, 1844 sleeping,   0 stopped,   0 zombie
Cpu(s):  8.1%us,  0.3%sy,  0.0%ni, 91.6%id,  0.0%wa,  0.0%hi,  0.0%si,  0.0%st
Mem:  264404360k total, 83616884k used, 180787476k free,  2075196k buffers
Swap: 33554428k total,   133268k used, 33421160k free, 46584472k cached
Which user (blank for all): 
  PID USER      PR  NI  VIRT  RES  SHR S %CPU %MEM    TIME+  COMMAND
54909 twongj01  20   0 76.8g 3.4g 386m S 100.3  1.4  15:02.14 python
53773 twongj01  20   0 71.4g 2.4g 303m S 100.3  0.9  18:41.79 python
55945 twongj01  20   0 76.7g 3.4g 386m S 100.3  1.3  12:19.94 python
53778 twongj01  20   0 76.7g 3.4g 386m R 100.2  1.3  18:44.55 python
55944 twongj01  20   0 76.8g 3.4g 386m S 100.0  1.4  12:01.92 python
57038 twongj01  20   0 76.7g 3.3g 386m R 100.0  1.3  10:10.53 python
```


