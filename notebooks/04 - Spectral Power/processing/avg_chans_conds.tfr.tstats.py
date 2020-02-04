import warnings  # Hide all warnings here
warnings.filterwarnings("ignore")

import mne
mne.utils.set_log_level('error')

import numpy as np

from os import listdir
from os.path import isfile, join
from functools import reduce

#%%
d = "../../../data/reinhartlab/multimodal/cg/"
ext = ".coll_chans.2to50-tfr.h5"
files = [join(d, f) for f in listdir(d) if isfile(join(d, f)) and ext in f]
tfrs = [mne.time_frequency.read_tfrs(f) for f in files]

#%%
def get_time_slice(tfrs,tmin=None,tmax=0):
    tf = tfrs[0].copy()
    tf.data = np.vstack([np.squeeze(t.data)[:,:,(t.times>=tmin) & (t.times<=tmax)] for t in tfrs])
    return tf

#%%
bsl = get_time_slice(tfrs,-.1,-.001)
bsl_mean = np.mean(np.mean(bsl.data,-1),1)

#%%
k=20; T = list(range(k)); c = list(range(k)); p = list(range(k));
for i in list(range(k)):
    act_mean = np.mean(np.mean(get_time_slice(tfrs,.001+(i-1)*.1,.1+(i-1)*.1).data,-1),1)
    T[i],c[i],p[i],_=mne.stats.permutation_cluster_test([bsl_mean,act_mean],threshold=6.)

#%%
m = [[[b in bsl.freqs[x] for b in bsl.freqs] for x,y in zip(a,b) if y<.05] for a,b in zip(c,p)]

#%%
n = [reduce(lambda x,y: np.logical_or(x,y),[np.ones_like(bsl.freqs)>1,*n]) for n in m]
#m = np.vstack(m)
mask = np.transpose(np.repeat(n,100,axis=0))