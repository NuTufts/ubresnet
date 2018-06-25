# Models

Folder for models in use or underdevelopment. Most code in pytorch. In the future, also will support TensorFlow versions.

Ask someone for trained model weights.  Will post somewhere public at some point.

## In Use

* `UResNet` in `ub_uresnet.py`. UNet with ResNet blocks. Upsampling done using Conv2d transpose layers. Simiar to model used by MicroBooNE SSNet 2018 paper.
* `dllee_ssnet2018.prototxt`. Prototxt describing network used for MicroBooNE SSNet 2018 paper and current SSNet used in MicroBooNE DL LEE analysis. To use this requires the package caffe2pytorch which is dependent on caffe1 being installed as well. (maybe will look for a pure conversion tool.)

## In Development

* (your model goes here)


