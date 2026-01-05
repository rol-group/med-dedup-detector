# Simple Spike Detectors with Detection De-Duplication
This repository contains routines to run threshold spike detectors on HDF5 files containing raw voltage traces. Thresholds are based on estimates of noise in the voltage signals, and can be estimated either with the Root-Mean-Square (RMS) or Median of Absolute Deviations (MED or MAD) of the voltage signal. There are routines for single channel, multi-channel, or whole file detections.


### Setup
To install the package on your system, run the following steps:
```bash
git clone https://github.com/rol-group/med-dedup-detector.git
cd med-dedup-detector
pip install .
```

### Parameters
All spike detection routines have the following signature:

```python
detectspikes(sig,            # Input data. [Chans, Samples]
             filtered=False, # If data is pre-filtered
             order=3,        # Order of High-Pass Butterworth Filter
             cutoff=300,     # Cutoff of HPF
             thresh=7,       # Spike detection threshold
             win=0.5,        # Duration for noise estimate
             fs=20000)       # Sampling rate
```
### Example Usage
The routines for loading ephys files and running spike detection are de-coupled. If you have a format other than the MaxWell Biosystems HD-MEA format, please consult your instrumentation's documentation on loading its output files. Once your traces are loaded into a numpy array, you can do the following:

```python
from spikedetection import detectspikesMat

X = load_binary(path_to_binary) # [Nchans x Nsamples] replace with your loading code
fs = 20000
batch_size = 10 * 20000 # 10s in samples
Nbatches = X.shape[1] // batch_size + 1 * (X.shape[1]%batch_size != 0)

allspks = []
for i in range(Nbatches):
   idx0,idx1 = i*batch_size,(i+1)*batch_size
   spks, _, _ = detectspikesMat(X[:,idx0:idx1],filtered=False)
   spks = [(fr + idx0,ch,amp) for ch,fr,amp in spks]
   allspks = allspks + spks

spks = allspks
```


### Citing
This repository accompanies the paper "A fast and simple algorithm for accurate spike detection in HD-MEA
recordings" by Zegers-Delgado et al. Please cite the paper if you find this code useful in your work.
