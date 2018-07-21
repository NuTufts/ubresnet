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

      
