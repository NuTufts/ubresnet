#Segmentation Network Use Instructions:
Author: Joshua Mills, Tufts University
github address:
https://github.com/NuTufts/ubresnet/tree/jmills_segment/

#NUDOT
#Network Location on NUDOT:
/mnt/disk0/jmills/ubresnet/

#DEPLOY:
#the deploy script is currently configured to run on the y plane (see config section of file)
#It also saves as output a ROOT file with trees:
Information       Tree
ADC Values:       wire
MCTruth Labels:   labels
Weights           weights
Background Prob:  background
Track Prob:       track
Track End Prob:   track_end
Shower Prob:      shower

As well as an image with labels given to each pixel according to it's highest probable classes
called: full_image, that can be compared with the MCtruth labels


/mnt/disk0/ubresnet/deploy/
                           run_segment_precropped.py
                           segment_funcs.py
                           Test_Validation.ipynb <------ A jupyter notebook with commands to examine the output of the deploy script
                                                         you will need root and I source ubresnet/larcv2_configure.sh

###Checkpoint Files Location on NUDOT
/mnt/disk0/jmills/
                  checkpoints_uplane/
                  checkpoints_vplane/
                  checkpoints_yplane/

#note that 3 ideal tar files copied to a directory that might work best for DEPLOY
mnt/disk0/jmills/selected_tar_files/
                   model_best_uplane.tar
                   model_best_vplane.tar
                   checkpoint.52500th_yplane.tar

      where the u and v plane best model is determined by the highest average accuracy across track, shower, trackend classes (not background)
      and the yplane checkpoint was the last in the run (it's model best parameters are just highest total accuracy, likely a bad choice)

#Run Information for Tensorboard on NUDOT:
/mnt/disk0/jmills/RUNS/
                       runs_u/
                       runs_v/
                       runs_y/

#Cropped ROOT files on NUDOT:
#These files are large 52, 72, and 31 GB respectively
/mnt/disk0/jmills/croppedfiles/
                                crop_train1.root
                                crop_train2.root
                                crop_valid.root

#Cropped files on tufts cluster:
#Each folder contains numerous individually cropped files, and an HADD version of them all together

/cluster/tufts/wongjiradlab/jmills09/uboonecode/files/crop_train/first/
/cluster/tufts/wongjiradlab/jmills09/uboonecode/files/crop_train/second/
/cluster/tufts/wongjiradlab/jmills09/uboonecode/files/crop_valid/
