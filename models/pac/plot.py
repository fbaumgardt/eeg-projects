import matplotlib.pyplot as plt

import warnings  # Hide all warnings here
warnings.filterwarnings("ignore")

import mne
mne.utils.set_log_level('error')

import numpy as np

def plot_fxf_topomaps(pacs,info,f_phase,f_amp,figsize=(12,12),pltkws={'sharex':True,'sharey':True},topokws={'cmap':'Spectral_r'}):
    fig,axes = plt.subplots(len(f_phase),len(f_amp),figsize=figsize,**pltkws)
    if np.ndim(axes)>1:
        axes = [a for ax in axes for a in ax]
    for i,a in enumerate(axes):
        mne.viz.topomap.plot_topomap(pacs[:,i],pos=info,show=False,axes=a,**topokws);
        if i>=len(f_amp)*(len(f_phase)-1):
            fa=i-len(f_amp)*(len(f_phase)-1)
            a.set_xlabel("$f_P="+str(f_amp[fa][0])+"-"+str(f_amp[fa][1])+"Hz$")
        if i%len(f_amp)==0:
            fp=i//len(f_amp)
            a.set_ylabel("$f_P="+str(f_phase[fp][0])+"-"+str(f_phase[fp][1])+"Hz$")