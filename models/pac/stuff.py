import numpy as np
import scipy as sp
import mne
from functools import reduce


def get_mask(raw, evts, tmin, tmax, by_trial=False):
    T = len(raw.times)
    mask = [(np.arange(T) >= max(int(e+tmin*raw.info['sfreq']), 0)) &
            (np.arange(T) <= min(int(e+tmax*raw.info['sfreq']), T)) for e in evts]
    if not by_trial:
        mask = reduce(lambda x,y: x & y, mask)
    return mask


def phase_fn(x):
    hil = sp.signal.hilbert(x)
    res = np.angle(hil)+np.pi
    return res


def amplitude_fn(x):
    hil = sp.signal.hilbert(x)
    res = np.abs(hil)
    return res


def get_pac(raw, mask, lofrq, hifrq, lo_func=phase_fn, hi_func=amplitude_fn, pac_func=lambda x: x):
    channels = mne.pick_types(raw.info, eeg=True)
    pacs = {raw.ch_names[ch]:[] for ch in channels}
    for ch in channels:
        print("begin filtering")
        hi = [hi_func(mne.filter.filter_data(raw.get_data(raw.ch_names[ch]), raw.info['sfreq'], hi[0], hi[1])) for hi in hifrq]
        print("filtered hi")
        lo = np.array([lo_func(mne.filter.filter_data(raw.get_data(raw.ch_names[ch]), raw.info['sfreq'], lo[0], lo[1])) for lo in lofrq])
        print("filtered lo")
        hi = [np.array([h[0,m] for m in mask]) for h in hi]
        lo = [np.array([l[0,m] for m in mask]) for l in lo]
        print("masked")
        pacs[raw.ch_names[ch]] = {str(lofrq[i][0])+'-'+str(lofrq[i][1]): {str(hifrq[j][0])+'-'+str(hifrq[j][1]): pac_func(l, h) for j,h in enumerate(hi)} for i,l in enumerate(lo)}
    return pacs


def pac_tort(lo, hi, nbin=20):
    bins = [np.logical_and(lo >= b[0], lo <= b[1]) for b in
            zip(np.linspace(0, 2 * np.pi, nbin + 1)[0:-1], np.linspace(0, 2 * np.pi, nbin + 1)[1:])]
    amps = np.fmax(np.array([[np.mean(h[a]) for h, a in zip(hi, b)] for b in bins]), np.finfo(float).eps)
    amps = amps / np.sum(amps, axis=0)
    hs = np.sum(amps * np.log(amps), axis=0)
    return (np.log(nbin) + hs) / np.log(nbin)