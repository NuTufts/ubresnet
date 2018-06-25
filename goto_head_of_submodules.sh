#!/bin/bash

# Assumes we are in larflow repo
# goes to the dev branches

cd larlite
git checkout trunk

cd ../larcv
git checkout tufts_ub

cd ../larcvdataset
git checkout master

cd ..
