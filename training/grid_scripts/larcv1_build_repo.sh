
REPODIR=$1

cd $REPODIR

git submodule init
git submodule update

export UBRESNET_BASEDIR=$PWD
export UBRESNET_MODELDIR=${UBRESNET_BASEDIR}/models
export LARCV_VERSION=1

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup larcv environment variabls
cd ../caffe/larcv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../../larcvdataset
source setenv.sh

# return to top-level directory
cd ../

source larcv1_build.sh
