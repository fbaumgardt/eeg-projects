import numpy as np
import mne
from functools import reduce

def get_time_slice(tfrs,tmin=None,tmax=0):
    tf = tfrs[0].copy()
    tf.data = np.vstack([np.squeeze(t.data)[:,:,(t.times>=tmin) & (t.times<=tmax)] for t in tfrs])
    return tf

def time_sliced_stats(tfrs,baseline=(-.1,-.001),intervals=[(.001,.1)],p_threshold=.05,condition=None):
    # baseline is tuple with 2 floats
    # condition is tuple with two strings

    k=len(intervals); T = list(range(k)); c = list(range(k)); p = list(range(k));
    for i,t in enumerate(intervals):
        bsl_mean = np.nanmean((get_time_slice(tfrs,baseline[0],baseline[1]) if isinstance(baseline,(list,tuple)) else baseline).data,-1) if condition is None else \
            np.nanmean(get_time_slice([tf[condition[0]] for tf in tfrs],t[0],t[1]).data,-1)
        act_mean = np.nanmean(get_time_slice(tfrs,t[0],t[1]).data,-1) if condition is None else \
            np.nanmean(get_time_slice([tf[condition[1]] for tf in tfrs],t[0],t[1]).data,-1)
        T[i],c[i],p[i],_=mne.stats.permutation_cluster_test([bsl_mean,act_mean])

    m = [[[f in tfrs[0].freqs[x] for f in tfrs[0].freqs] for x,y in zip(a,b) if y<p_threshold] for a,b in zip(c,p)]
    mask = [reduce(lambda x,y: np.logical_or(x,y),[np.ones_like(tfrs[0].freqs)>1,*n]) for n in m]
    return (np.vstack(T).T,np.array(mask).T)