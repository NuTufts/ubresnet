# UB ResNet Deploy scripts

Scripts to deploy and process images go here.

* run_larflow_wholeview.py: Process larcv files containing whole images. Uses larcv processes, UBSplitDetector, and, UBLArFlowStitcher,
  to first divide image into subimages, runs SSNet, and then remerges them.
* run_larflow_precropped.py: Process larcv files containeing pre-cropped images. Useful for per-subimage analysis.
* run_dllee_wholeview.py: Process larcv files. Details of splitting and merging are meant to reproduce the DLLEE SSNet code.

