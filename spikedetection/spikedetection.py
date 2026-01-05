import numpy as np
import scipy.signal as signal
from multiprocessing import Pool
from functools import reduce
import logging
import h5py

import time

logger = logging.getLogger(__name__)

# "detectors" can take whole files... "detect" routines just takes a data
# matrix and detects the spikes in it.

# This is kinda slow... How to speed it up?
def detectspikes(sig,filtered=False,order=3,cutoff=300,thresh=7,win=0.5,fs=20000):

    t0  = time.time()
    if filtered is False:
        b, a = signal.butter(order,cutoff,btype='high',fs=fs)
        sigf = signal.filtfilt(b,a,sig)
    else:
        sigf = sig
    t1 = time.time()

    Ns = len(sigf)
    dtsamps = int(win * fs)
    nwins = (Ns // dtsamps) + 1
    medvals = np.zeros(nwins)
    rmsvals = np.zeros(nwins)
    idxs = [min(i*dtsamps,Ns) for i in range(nwins)]
    dettrace = np.zeros_like(sigf)
    for i in range(nwins-1):
        clip = sigf[idxs[i]:idxs[i+1]]
        medvals[i] = np.median(np.abs(clip / 0.6745)) # TODO: Why is the factor of 0.6745 included in Quiroga et al. 2004?
        rmsvals[i] = np.sqrt(((clip-clip.mean(axis=-1))**2).mean())
        dettrace[idxs[i]:idxs[i+1]] = clip <= (medvals[i] * -1 * thresh)

    medthresh = medvals * -1 * thresh

    t2 = time.time()
    crossings = np.argwhere(np.diff(dettrace) == 1) # find positive crossings
    if len(crossings) > 0:
        intervals = np.diff(crossings) * 50e-6
        removespks = np.argwhere(intervals <= 1e-3) # find detections with ISI <= 1ms
        crossings[removespks+1] = -1 # get rid of detections with ISI < 1 ms. +1 to account for diff indices
        crossings = crossings[crossings != -1]
        crossings += 1

        # For each crossing, find the maximum value in the next few samples
        for i,cidx in enumerate(crossings):
            crossings[i] = np.argmin(sigf[cidx:cidx+20]).squeeze() + cidx

    t3 = time.time()
    print(f'filt: {t1 - t0:.4f}s; thr: {t2 - t1:.4f}s; post {t3 - t2:.4f}s')
    return crossings, idxs, medthresh

''' Assumes sig is [Nchans x Nsamps]
'''
def detectspikesMat(sig,filtered=False,order=3,cutoff=300,thresh=7,win=0.5,fs=20000):
    t0  = time.time()
    if filtered is False:
        b, a = signal.butter(order,cutoff,btype='high',fs=fs)
        sigf = signal.filtfilt(b,a,sig,axis=-1)
    else:
        sigf = sig - sig.mean(axis=-1)[:,None] # Center each channel
    t1 = time.time()

    Ns = sigf.shape[1]
    Nchans = sigf.shape[0]
    dtsamps = int(win * fs)
    nwins = (Ns // dtsamps) + 1
    medvals = np.zeros((Nchans,nwins))
    idxs = [min(i*dtsamps,Ns) for i in range(nwins)]
    dettrace = np.zeros_like(sigf)
    for i in range(nwins-1):
        clip = sigf[:,idxs[i]:idxs[i+1]]
        medvals[:,i] = np.median(np.abs(clip / 0.6745),axis=-1) # 0.6745 ~= median of abs(X), X ~ N(0,1)
        dettrace[:,idxs[i]:idxs[i+1]] = clip <= (medvals[:,i,None] * -1 * thresh)

    medthresh = medvals * -1 * thresh

    t2 = time.time()
    crossings = np.argwhere(np.diff(dettrace,axis=-1) == 1) # find positive crossings
    if len(crossings) > 0:
        # Sort crossings by channel
        chansort = np.argsort(crossings[:,0],kind='stable')
        crossings = crossings[chansort]

        # Find places where the channel switches
        chswitch = np.argwhere(np.diff(crossings[:,0]) != 0)

        # Compute ISIs. Mark channel switch ISIs as infinite
        intervals = np.diff(crossings[:,1],axis=-1) * 50e-6
        intervals[chswitch] = np.inf

        # Remove ISIs that are less than 1 ms
        removespks = np.argwhere(intervals < 1e-3) # find detections with ISI < 1ms
        crossings[removespks+1,0] = -1 # set channel of 1ms violations to -1
        crossings = crossings[crossings[:,0] != -1]
        crossings[:,1] += 1

        # For each crossing, find the maximum value in the next few samples
        amps = np.zeros(len(crossings))
        for i,(ch,t) in enumerate(crossings):
            crossings[i,1] = np.argmin(sigf[ch,t:t+20]).squeeze() + t
            amps[i] = sigf[ch,crossings[i,1]]

        crossings = np.column_stack([crossings[:,0],crossings[:,1],amps])

    t3 = time.time()
    print(f'filt: {t1 - t0:.4f}s; thr: {t2 - t1:.4f}s; post {t3 - t2:.4f}s')
    return crossings, idxs, medthresh

''' Assumes sig is [Nchans x Nsamps]
'''
def rmsdetectspikesMat(sig,filtered=False,order=3,cutoff=300,thresh=7,win=0.5,fs=20000):
    t0  = time.time()
    if filtered is False:
        b, a = signal.butter(order,cutoff,btype='high',fs=fs)
        sigf = signal.filtfilt(b,a,sig,axis=-1)
    else:
        sigf = sig
    t1 = time.time()

    Ns = sigf.shape[1]
    Nchans = sigf.shape[0]
    dtsamps = int(win * fs)
    nwins = (Ns // dtsamps) + 1
    rmsvals = np.zeros((Nchans,nwins))
    idxs = [min(i*dtsamps,Ns) for i in range(nwins)]
    dettrace = np.zeros_like(sigf)
    for i in range(nwins-1):
        clip = sigf[:,idxs[i]:idxs[i+1]]
        rmsvals[:,i] = np.sqrt(((clip - clip.mean(axis=-1)[:,None])**2).mean(axis=-1))
        dettrace[:,idxs[i]:idxs[i+1]] = clip <= (rmsvals[:,i,None] * -1 * thresh)

    rmsthresh = rmsvals * -1 * thresh

    t2 = time.time()
    crossings = np.argwhere(np.diff(dettrace,axis=-1) == 1) # find positive crossings
    if len(crossings) > 0:
        # Sort crossings by channel
        chansort = np.argsort(crossings[:,0],kind='stable')
        crossings = crossings[chansort]

        # Find places where the channel switches
        chswitch = np.argwhere(np.diff(crossings[:,0]) != 0)

        # Compute ISIs. Mark channel switch ISIs as infinite
        intervals = np.diff(crossings[:,1],axis=-1) * 50e-6
        intervals[chswitch] = np.inf

        # Remove ISIs that are less than 1 ms
        removespks = np.argwhere(intervals < 1e-3) # find detections with ISI < 1ms
        crossings[removespks+1,0] = -1 # set channel of 1ms violations to -1
        crossings = crossings[crossings[:,0] != -1]
        crossings[:,1] += 1

        # For each crossing, find the maximum value in the next few samples
        amps = np.zeros(len(crossings))
        for i,(ch,t) in enumerate(crossings):
            crossings[i,1] = np.argmin(sigf[ch,t:t+20]).squeeze() + t
            amps[i] = sigf[ch,crossings[i,1]]

        crossings = np.column_stack([crossings[:,0],crossings[:,1],amps])

    t3 = time.time()
    print(f'filt: {t1 - t0:.4f}s; thr: {t2 - t1:.4f}s; post {t3 - t2:.4f}s')
    return crossings, idxs, rmsthresh

def mediandetector(rec,filtered=False,batchsize=10,thresh=7,order=3,cutoff=300):
    """Run median based spike detection on a recording and return a list of
    spike times

    rec -- MaxWellH5Recording object

    Keyword Arguments:
    batchsize -- duration (in seconds) of batches
    thresh -- multiple of noise estimate
    order -- order of the high pass butterworth filter
    cutoff -- critical frequency of the butterworth filter
    """

    assert rec.raws is not None
    raw = rec.raws[0]
    nchans, dursamps = raw.shape
    dur = dursamps * 50e-6
    bsamps = batchsize * 20000
    nbatches = (dursamps // bsamps) + 1
    idxs = [[bsamps*i,min(bsamps*(i+1),dursamps)] for i in range(nbatches)]
    spklist = []

    # Sometimes the last batch is exceptionally small, about 10 samples. If the
    # last batch is smaller than some threshold, then we should merge it to the
    # next to last batch
    if (idxs[-1][1]-idxs[-1][0]) / bsamps < 0.25:
        print(f'Last batch is {idxs[-1][1]-idxs[-1][0]} samples. Merging to previous batch.')
        idxs[-2][1] = dursamps
        idxs = idxs[:-1]
        nbatches -= 1

    for i in range(nbatches):
        print(f'Batch {i+1} of {nbatches}')
        bX = raw[:,idxs[i][0]:idxs[i][1]]
        for chidx in range(nchans):
            print(f'Channel {chidx+1} of {nchans}',end='\r')
            spks, _, _ = detectspikes(bX[chidx,:],thresh=thresh,filtered=filtered)
            spks = [(fr + idxs[i][0],chidx) for fr in spks]
            spklist += spks
        print("\r\033[A\033[A\r")

    spklist = np.array(spklist)
    spklist = spklist[np.argsort(spklist[:,0])]

#    chmap = rec.getMapping()['channel']
#    chs = [chmap[i] for i in spklist[:,1]]
#    spklist[:,1] = chs

    # indices are indexing frames, and are not frames yet
    frs = rec.getFrames()
    frs -= frs[0]
    spklist[:,0] = frs[spklist[:,0]]

    return spklist

def mediandetectorMat(rec,filtered=False,batchsize=10,thresh=7,order=3,cutoff=300,win=0.5):
    """Run median based spike detection on a recording and return a list of
    spike times

    rec -- MaxWellH5Recording object

    Keyword Arguments:
    batchsize -- duration (in seconds) of batches
    thresh -- multiple of noise estimate
    order -- order of the high pass butterworth filter
    cutoff -- critical frequency of the butterworth filter
    """

    assert rec.raws is not None
    raw = rec.raws[0]
    nchans, dursamps = raw.shape
    dur = dursamps * 50e-6
    bsamps = batchsize * 20000
    nbatches = (dursamps // bsamps) + 1
    idxs = [[bsamps*i,min(bsamps*(i+1),dursamps)] for i in range(nbatches)]
    spklist = []

    if dursamps == 0:
        return []

    # Sometimes the last batch is exceptionally small, about 10 samples. If the
    # last batch is smaller than some threshold, then we should merge it to the
    # next to last batch
    if (idxs[-1][1]-idxs[-1][0]) / bsamps < 0.25 and nbatches > 1:
        print(f'Last batch is {idxs[-1][1]-idxs[-1][0]} samples. Merging to previous batch.')
        idxs[-2][1] = dursamps
        idxs = idxs[:-1]
        nbatches -= 1

    for i in range(nbatches):
        print(f'Batch {i+1} of {nbatches}: ', end='')
        bX = raw[:,idxs[i][0]:idxs[i][1]]
        spks, _, _ = detectspikesMat(bX,thresh=thresh,win=win,filtered=filtered)
        spks = [(fr + idxs[i][0],ch,amp) for ch,fr,amp in spks]
        spklist += spks

    spklist = np.array(spklist)
    spklist = spklist[np.argsort(spklist[:,0])]


#    chmap = rec.getMapping()['channel']
#    chs = [chmap[i] for i in spklist[:,1]]
#    spklist[:,1] = chs

    # indices are indexing frames, and are not frames yet
    frs = rec.getFrames()
    frs -= frs[0]
    spklist[:,0] = frs[spklist[:,0].astype(np.int64)]

    return spklist

def rmsdetectorMat(rec,filtered=False,batchsize=10,thresh=7,order=3,cutoff=300,win=0.5):
    """Run rms based spike detection on a recording and return a list of
    spike times

    rec -- MaxWellH5Recording object

    Keyword Arguments:
    batchsize -- duration (in seconds) of batches
    thresh -- multiple of noise estimate
    order -- order of the high pass butterworth filter
    cutoff -- critical frequency of the butterworth filter
    """

    assert rec.raws is not None
    raw = rec.raws[0]
    nchans, dursamps = raw.shape
    dur = dursamps * 50e-6
    bsamps = batchsize * 20000
    nbatches = (dursamps // bsamps) + 1
    idxs = [[bsamps*i,min(bsamps*(i+1),dursamps)] for i in range(nbatches)]
    spklist = []

    if dursamps == 0:
        return []

    # Sometimes the last batch is exceptionally small, about 10 samples. If the
    # last batch is smaller than some threshold, then we should merge it to the
    # next to last batch
    if (idxs[-1][1]-idxs[-1][0]) / bsamps < 0.25 and nbatches > 1:
        print(f'Last batch is {idxs[-1][1]-idxs[-1][0]} samples. Merging to previous batch.')
        idxs[-2][1] = dursamps
        idxs = idxs[:-1]
        nbatches -= 1

    for i in range(nbatches):
        print(f'Batch {i+1} of {nbatches}')
        bX = raw[:,idxs[i][0]:idxs[i][1]]
        spks, _, _ = rmsdetectspikesMat(bX,thresh=thresh,win=win,filtered=filtered)
        spks = [(fr + idxs[i][0],ch) for ch,fr in spks]
        spklist += spks
        print("\r\033[A\033[A\r")

    spklist = np.array(spklist)
    spklist = spklist[np.argsort(spklist[:,0])]

    chmap = rec.getMapping()['channel']
    chs = [chmap[i] for i in spklist[:,1]]
    spklist[:,1] = chs

    # indices are indexing frames, and are not frames yet
    frs = rec.getFrames()
    frs -= frs[0]
    spklist[:,0] = frs[spklist[:,0]]

    return spklist

def prochelpermedian(fname,idx0,idx1,thresh,win,filtered):
    f = h5py.File(fname)
    bX = f['data_store/data0000/groups/routed/raw'][:,idx0:idx1]
    spks, _, _ = detectspikesMat(bX,thresh=thresh,win=win,filtered=filtered)
    spks = [(fr + idx0,ch,amp) for ch,fr,amp in spks]
    f.close()
    return spks

def prochelperrms(fname,idx0,idx1,thresh,win,filtered):
    f = h5py.File(fname)
    bX = f['data_store/data0000/groups/routed/raw'][:,idx0:idx1]
    spks, _, _ = rmsdetectspikesMat(bX,thresh=thresh,win=win,filtered=filtered)
    spks = [(fr + idx0,ch,amp) for ch,fr,amp in spks]
    f.close()
    return spks

def mediandetectorMultiMat(rec,filtered=False,numprocs=4,batchsize=10,thresh=7,order=3,cutoff=300,win=0.5):
    """Run median based spike detection on a recording and return a list of
    spike times on multiple processes.

    rec -- MaxWellH5Recording object

    Keyword Arguments:
    batchsize -- duration (in seconds) of batches
    thresh -- multiple of noise estimate
    order -- order of the high pass butterworth filter
    cutoff -- critical frequency of the butterworth filter
    numprocs -- number of processes
    """

    assert rec.raws is not None
    raw = rec.raws[0]
    nchans, dursamps = raw.shape
    dur = dursamps * 50e-6
    bsamps = batchsize * 20000
    nbatches = (dursamps // bsamps) + 1
    idxs = [[bsamps*i,min(bsamps*(i+1),dursamps)] for i in range(nbatches)]
    spklist = []

    if dursamps == 0:
        return []

    # Sometimes the last batch is exceptionally small, about 10 samples. If the
    # last batch is smaller than some threshold, then we should merge it to the
    # next to last batch
    if (idxs[-1][1]-idxs[-1][0]) / bsamps < 0.25 and nbatches > 1:
        print(f'Last batch is {idxs[-1][1]-idxs[-1][0]} samples. Merging to previous batch.')
        idxs[-2][1] = dursamps
        idxs = idxs[:-1]
        nbatches -= 1

    fname = rec.fname
    args = [(fname,idxs[i][0],idxs[i][1],thresh,win,filtered) for i in range(len(idxs))]
    print("Launching processes")
    with Pool(numprocs) as p:
        print("starmapping")
        spksall = p.starmap(prochelpermedian,args)
        print("done")
        p.close()
        p.join()

    spklist = reduce(lambda x,y: x + y, spksall, [])
    spklist = np.array(spklist)
    spklist = spklist[np.argsort(spklist[:,0])]

    # Convert to channel nos. rather than indices in the data matrix
    chmap = rec.getMapping()['channel']
    xmap = rec.getMapping()['x']
    ymap = rec.getMapping()['y']

    if len(chmap) > raw.shape[0]:
        print("WARNING: More channels in mapping than in raw matrix")
        chmap = rec.fhandle['data_store/data0000/groups/routed/channels']
    chs = [chmap[int(i)] for i in spklist[:,1]]
    xs = [xmap[int(i)] for i in spklist[:,1]]
    ys = [ymap[int(i)] for i in spklist[:,1]]
    spklist[:,1] = chs
    spklist = np.column_stack([spklist[:,0], spklist[:,1], xs, ys, spklist[:,2]])

    # indices are indexing frames, and are not frames yet
    frs = rec.getFrames()
    frs -= frs[0]
    spklist[:,0] = frs[spklist[:,0].astype(np.int64)]

    print(spklist)
    return spklist

def rmsdetectorMultiMat(rec,filtered=False,numprocs=4,batchsize=10,thresh=7,order=3,cutoff=300,win=0.5):
    """Run RMS based spike detection on a recording and return a list of
    spike times on multiple processes.

    rec -- MaxWellH5Recording object

    Keyword Arguments:
    batchsize -- duration (in seconds) of batches
    thresh -- multiple of noise estimate
    order -- order of the high pass butterworth filter
    cutoff -- critical frequency of the butterworth filter
    numprocs -- number of processes
    """

    assert rec.raws is not None
    raw = rec.raws[0]
    nchans, dursamps = raw.shape
    dur = dursamps * 50e-6
    bsamps = batchsize * 20000
    nbatches = (dursamps // bsamps) + 1
    idxs = [[bsamps*i,min(bsamps*(i+1),dursamps)] for i in range(nbatches)]
    spklist = []

    if dursamps == 0:
        return []

    # Sometimes the last batch is exceptionally small, about 10 samples. If the
    # last batch is smaller than some threshold, then we should merge it to the
    # next to last batch
    if (idxs[-1][1]-idxs[-1][0]) / bsamps < 0.25 and nbatches > 1:
        print(f'Last batch is {idxs[-1][1]-idxs[-1][0]} samples. Merging to previous batch.')
        idxs[-2][1] = dursamps
        idxs = idxs[:-1]
        nbatches -= 1

    fname = rec.fname
    args = [(fname,idxs[i][0],idxs[i][1],thresh,win,filtered) for i in range(len(idxs))]
    print("Launching processes")
    with Pool(numprocs) as p:
        print("starmapping")
        spksall = p.starmap(prochelperrms,args)
        print("done")
        p.close()
        p.join()

    spklist = reduce(lambda x,y: x + y, spksall, [])
    spklist = np.array(spklist)
    spklist = spklist[np.argsort(spklist[:,0])]

    # Convert to channel nos. rather than indices in the data matrix
    chmap = rec.getMapping()['channel']
    xmap = rec.getMapping()['x']
    ymap = rec.getMapping()['y']

    chs = [chmap[int(i)] for i in spklist[:,1]]
    xs = [xmap[int(i)] for i in spklist[:,1]]
    ys = [ymap[int(i)] for i in spklist[:,1]]
    spklist[:,1] = chs
    spklist = np.column_stack([spklist[:,0], spklist[:,1], xs, ys, spklist[:,2]])

    # indices are indexing frames, and are not frames yet
    frs = rec.getFrames()
    frs -= frs[0]
    spklist[:,0] = frs[spklist[:,0].astype(np.int64)]

    print(spklist)
    return spklist

def createSpikeListIndex(spks,mode='abs'):
    '''
        This function takes in a spikelist [Nx2] and returns a [Tx2] list of
        indices. T is the length in samples of the spikelist, NOT the length in
        spikes of the spikelist.

        Entry out[i,0] represents the first spike to occur at or after time i
        Entry out[i,1] represents the last spike to occur at or before time i
        The spikes contained in an interval [i,j] can be obtained by:
            spikes[out[i,0]:out[j,0]+1]

        mode: 'abs' or 'rel'. Whether to treat the first spike as time 0 or recording start as time 0.
    '''

    assert mode in ['abs', 'rel'], "mode must be 'abs' or 'rel'"
    assert spks.shape[1] >= 2     # at least 2 columns
    assert len(spks.shape) == 2   # 2 dimensions

    T0 = int(spks[0,0]) if mode == 'rel' else 0
    Tlen = int((spks[-1,0] - T0) + 1)
    Nspks = len(spks)
    idxpairs = np.zeros((Tlen,2),dtype=np.int64)

    spkidx = 0
    spkidx2 = len(spks) - 1
    for i in range(Tlen):
        # this gives the first index greater than the current sample i
        # If we are already greater than i, we don't need to do anything
        while spkidx < Nspks and spks[spkidx,0] < i + T0:
            spkidx += 1
        idxpairs[i,0] = spkidx

        # Moving backwards gives the last index less than the current sample j
        j = Tlen - i - 1
        while spkidx2 >= 0 and spks[spkidx2,0] > j + T0:
            spkidx2 -= 1
        idxpairs[j,1] = spkidx2

    # Properties if idxpairs for i < j:
    # Anywhere idxpairs[i,1] < idxpairs[j,0] means 0 spikes in the interval [i,j]
    # Anywhere idxpairs[i,0] == idxpairs[j,1] means 1 spike in the interval [i,j]
    # Anywehre idxpairs[i,1] == -1 means no spikes at or before i
    # Anywhere idxpairs[i,0] == Nspks + 1 means no spikes at or after i
    return idxpairs

def getSpikesInInterval(t1,t2,spks,idxpairs,fs=20000):
    '''
        Takes in two times t1 and t2 (in seconds), a spikelist, and a pre-computed index.
        Returns a subset of spks that fall between t1 and t2, inclusive.
    '''
    pairidx0 = max(0, min(len(idxpairs)-1,int(t1*fs)))
    pairidx1 = max(0, min(len(idxpairs)-1,int(t2*fs)))
    idx0, idx1 = idxpairs[pairidx0, 0], idxpairs[pairidx1, 1]
    out = np.zeros((0,spks.shape[1]),dtype=np.int32)
    if idx1 != -1 and idx0 != len(spks) + 1 and idx1 >= idx0:
        out = spks[idx0:idx1+1].copy()

    return out

def getSpikeArgsInInterval(t1,t2,spks,idxpairs,fs=20000):
    '''
        Takes in two times t1 and t2 (in seconds), a spikelist, and a pre-computed index.
        Returns the indices that correspond to a subset of spks that fall
        between t1 and t2, inclusive.
    '''
    pairidx0 = max(0, min(len(idxpairs)-1,int(t1*fs)))
    pairidx1 = max(0, min(len(idxpairs)-1,int(t2*fs)))
    idx0, idx1 = idxpairs[pairidx0, 0], idxpairs[pairidx1, 1]
    out = np.zeros((0,2),dtype=np.int32)
    if idx1 != -1 and idx0 != len(spks) + 1 and idx1 >= idx0:
        out = np.arange(idx0,idx1+1,dtype=np.int32)

    return out


'''
    This function takes in a spikelist [Nx5] and removes duplicate spikes.
    Spikes that occurr within a couple of samples of one another and are within
    a certain distance are treated as one spike, with the maximum amplitude
    spike being retained.

    The spikelist here assumes that we have columns for the position and amplitude of the spikes.
'''
def deduplicateSpikes(spks,return_duplicates=False,sampwin=3,debug=False):
    index = createSpikeListIndex(spks)
    range = 25 # [um]. this defines a 3x3 electrode square
    keep = np.ones(len(spks),dtype=np.int32) # start by keeping all spikes
    for i,(samp,ch,x,y,amp) in enumerate(spks):
        # Skip spikes we've said are duplicate
        # We actually don't want to do this.
        #if keep[i] == 0:
        #    continue

        t0 = (samp-sampwin)*50e-6
        t1 = (samp+sampwin)*50e-6

        tspks = getSpikesInInterval(t0,t1,spks,index)
        if len(tspks) > 0:
            idxspks = getSpikeArgsInInterval(t0,t1,spks,index)
            assert len(tspks) == len(idxspks), print(f'Length mismatch {tspks.shape}, {idxspks.shape}')
            ds = np.atleast_1d(np.linalg.norm((np.array([x,y]) - tspks[:,2:4]),axis=1))
            args = np.atleast_1d(np.squeeze(np.argwhere(ds < range)))
            nearby = tspks[args]
            if len(nearby) > 0:
                nearbyidxs = idxspks[args]
                if debug:
                    print(nearby[:,4])
                argkeep = np.argmax(np.abs(nearby[:,4])) # Highest amplitude
                if debug:
                    print(nearbyidxs,argkeep)
                discard = np.delete(nearbyidxs, argkeep)
                if debug:
                    print(discard)
                keep[discard] = 0

    if return_duplicates:
        return spks[np.argwhere(keep==1).squeeze()].copy(), spks[np.argwhere(keep==0).squeeze()].copy()
    else:
        return spks[np.argwhere(keep==1).squeeze()].copy()
