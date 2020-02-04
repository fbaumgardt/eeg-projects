import local
import warnings  # Hide all warnings here
warnings.filterwarnings("ignore")
import mne
mne.utils.set_log_level('error')
import numpy as np
from lib.mne_sandbox.connectivity.cfc import _phase_amplitude_coupling
from joblib import Parallel,delayed

def get_single_pac_sensor_space(epochs,f_phase=None,f_amp=None,picks=None,pac_func='mi_tort',return_data=False):
    ##
    # Returns: pacs) # chans, freqxfreq // phs) # chans, freq, time // ams) # chans, freq, time
    ##
    if f_phase is None:
        f_phase = [[5,7],[7,9]]
    if f_amp is None:
        f_amp = [[40,94]]
    if picks is None:
        picks = mne.pick_types(epochs.info,meg=False,eeg=True)
    else:
        picks = mne.pick_channels(epochs.info,include=picks)
    chans = [[p,p] for p in picks]
    if return_data:
        pacs,freq,phs,ams = _phase_amplitude_coupling(np.hstack(epochs),epochs.info['sfreq'],f_phase,f_amp,chans,pac_func=pac_func,return_data=return_data)
        return {'file':epochs.filename,'info':epochs.info,'channels':[epochs.ch_names[p] for p in picks],'frequencies':np.squeeze(freq),'times':epochs.times,'pac':np.squeeze(pacs),'phases':np.squeeze(phs),'amplitudes':np.squeeze(ams),'metadata':epochs.metadata}
    else:
        pacs,freq = _phase_amplitude_coupling(np.hstack(epochs),epochs.info['sfreq'],f_phase,f_amp,chans,pac_func=pac_func,return_data=return_data)
        return {'file':epochs.filename,'info':epochs.info,'channels':[epochs.ch_names[p] for p in picks],'frequencies':np.squeeze(freq),'times':epochs.times,'pac':np.squeeze(pacs),'phases':np.array([]),'amplitudes':np.array([]),'metadata':epochs.metadata}

def get_single_pac_source_space(source_estimates,epochs,f_phase=None,f_amp=None,pac_func='mi_tort'):
    ##
    # Returns: pacs) # chans, freqxfreq // phs) # chans, freq, time // ams) # chans, freq, time
    ##
    if f_phase is None:
        f_phase = [[5,7],[7,9]]
    if f_amp is None:
        f_amp = [[40,94]]
    data = np.hstack([s.data for s in source_estimates])
    pacs = [None]*len(data)
    for i  in range(len(data)):
        pacs[i],freq = _phase_amplitude_coupling(data[np.newaxis,i,:],epochs.info['sfreq'],f_phase,f_amp,[[0,0]],pac_func=pac_func,return_data=False)
    return {'file':epochs.filename,'info':epochs.info,'frequencies':np.squeeze(freq),'times':epochs.times,'pac':np.squeeze(pacs),'metadata':epochs.metadata}

def get_mis(pc,chan,freq):
    c = np.where([p==chan for p in pc['channels']])[0]
    f = np.where([np.all(p==freq) for p in pc['frequencies']])[0]
    return pc['pac'][c,f]

def get_cross_pac(single_pacs_condition,single_pacs_control):
    # filter pacs? [p for p in pacs if chans in p['channels']]
    # ALSO, CHANGE TO CLUSTER PERMUTATION!!!
    channels = set([c for s in single_pacs_condition for c in s['channels']])
    cont = np.concatenate([get_mis(pac,chan,freq) for pac in single_pacs_condition])
    cond = np.concatenate([get_mis(pac,chan,freq) for pac in single_pacs_condition])
    t,p = stats.ttest_rel(cont,cond,nan_policy='omit')
    return (t,p)

def get_pacs(f,condition=None,f_phase=[[4,7]],f_amp=[[40,94]],tmin=0.,tmax=None,baseline=(-.5,-.3),ch_order=None,by_trial=False):
    ch_order = ch_order if ch_order is not None else ['AF3','AF4','AF7','AF8','AFz','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','FCz','FT7','FT8','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP10','TP7','TP8','TP9']
    epochs = f.copy() if type(f) is mne.epochs.EpochsFIF else mne.read_epochs(f)
    epochs = epochs.reorder_channels(ch_order).apply_baseline(baseline).crop(tmin,tmax)
    if condition is not None:
        epochs = epochs[condition]
    if by_trial:
        return [get_single_pac_sensor_space(epochs[i],f_phase,f_amp) for i in range(len(epochs))]
    else:
        return get_single_pac_sensor_space(epochs,f_phase,f_amp)