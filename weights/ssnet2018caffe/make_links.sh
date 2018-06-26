#!/bin/sh

# Make symbolic links to weight files to avoid copying data.

# On Meitner
SSNET_WEIGHT_DIR="/media/hdd1/larbys/ssnet_model_weights/"
ln -s ${SSNET_WEIGHT_DIR}/segmentation_pixelwise_ikey_plane0_iter_75500.caffemodel segmentation_pixelwise_ikey_plane0_iter_75500.caffemodel 
ln -s ${SSNET_WEIGHT_DIR}/segmentation_pixelwise_ikey_plane1_iter_65500.caffemodel segmentation_pixelwise_ikey_plane1_iter_65500.caffemodel 
ln -s ${SSNET_WEIGHT_DIR}/segmentation_pixelwise_ikey_plane2_iter_68000.caffemodel segmentation_pixelwise_ikey_plane2_iter_68000.caffemodel
