# Simple Spike Detectors with Detection De-Duplication
This repository contains routines to run threshold spike detectors on HDF5 files containing raw voltage traces. Thresholds are based on estimates of noise in the voltage signals, and can be estimated either with the Root-Mean-Square (RMS) or Median of Absolute Deviations (MED or MAD) of the voltage signal. There are routines for single channel, multi-channel, or whole file detections.


### Citing
This repository accompanies the paper "A fast and simple algorithm for accurate spike detection in HD-MEA
recordings" by Zegers-Delgado et al. Please cite the paper if you find this code useful in your work.
