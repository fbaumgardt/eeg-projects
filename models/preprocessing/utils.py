import os
import numpy as np
import pandas as pd
import mne
from functools import reduce
from ipywidgets import Dropdown, Checkbox, HBox, Layout

def find_complete_intervals(starts,ends):
    """
    Find closed intervals from list of starts and endings

    :param starts: List of samples with segment start codes
    :param ends: List of samples with segment end codes
    """
    if len(starts) and len(ends):
        idx = np.argsort(np.concatenate((starts,ends)))
        pairs = [all([idx[s]<len(starts),idx[s+1]>=len(starts)]) for s in range(len(idx)-1)]
        return np.vstack((starts[idx[pairs+[False]]],ends[idx[[False]+pairs]-len(starts)])).T
    else:
        return np.array([[0,0]])
    
def get_samples(e,c):
    """
    Get sample locations from event list e for types in c.

    :param e: MNE-style event list (3 columns)
    :param c: A list of event codes
    :returns: List of samples at which events of specified types occur
    """
    return np.array([f[0] for f in e if f[2] in c])

def get_value(a,default=np.nan):
    """
    Get first value from array, or a default value if array is empty.

    :param a: The array
    :param default: The default value (optional)
    :returns: a[0] or default
    """
    if len(a):
        return a[0]
    else:
        return default

def get_block(epochs,blck_id,cond,evts,eids):
    """
    Find the trials of type blck_id and mark them by block index.

    :param epochs: MNE Epochs object
    :param blck_id: Block type to find (string)
    :param cond: Condition definition (dict)
    :param evts: Events matrix (np.array)
    :param eids: Events string/int conversion dict (dict)
    :returns: Array with string blck_id/'' at positions where trials match or don't match the condition, second array with block indices for matching trials and 0 otherwise
    """
    
    start_codes = []# +new Segment
    end_codes = [] # +last sample
    
    if cond['blocks'][blck_id]['begin'] in eids.keys():
        block_begins = get_samples(evts,[eids[cond['blocks'][blck_id]['begin']]])
    else:
        block_begins = []
    if cond['blocks'][blck_id]['end'] in eids.keys():
        block_ends = get_samples(evts,[eids[cond['blocks'][blck_id]['end']]])
    else:
        block_ends = []
        
    blocks = find_complete_intervals(block_begins,block_ends)
    
    stim_int = [eids[s] for s in cond['blocks'][blck_id]['stimuli'].values()]
    blocks = [np.logical_and(np.logical_and(epochs.events[:][:,0]>b[0],epochs.events[:][:,0]<b[1]),np.isin(epochs.events[:][:,2],stim_int)) for b in blocks]
    
    
    
    blocks_idx = reduce(lambda x,y: x+y,[(i+1)*b for i,b in enumerate(blocks)],0)
    blocks = reduce(lambda x,y: np.logical_or(x,y),blocks,False)
    blocks_type = np.where(blocks,blck_id,'')
    
    return (blocks_type,blocks_idx)

def get_metadata(raw,cond):
    """
    Use Condition definition to build metadata dataframe from MNE Raw Object.

    :param raw: RawType object
    :param cond: Condition definition in Dict/Json format (dict)
    :returns: Dataframe for use as Metadata object
    """
    evts,eids = mne.events_from_annotations(raw)
    einvs = {v:k for k,v in eids.items()}
    fdbinvs = {v:k for k,v in cond['feedback'].items()}
    rspinvs = {v:k for k,v in cond['responses'].items()}
    stimuli = {k:v for d in cond['blocks'].values() for k,v in d['stimuli'].items()}
    epochs = mne.Epochs(raw,evts,{k:eids[v] for k,v in stimuli.items() if v in eids.keys()},tmin=-.2,tmax=.5)
    
    blocks = [get_block(epochs,b,cond,evts,eids) for b in cond['blocks'].keys()]
    blocks_type = reduce(lambda x,y: np.core.char.add(x, y),[b[0] for b in blocks],'')
    blocks_idx = reduce(lambda x,y: np.add(x, y),[b[1] for b in blocks],0)
    
    stim_samples = np.hstack([epochs.events[:][:,0],[np.infty]])
    events_by_trial = [[[s[0]-b,s[1],einvs[s[2]]] for s in evts[np.logical_and(evts[:,0]>b,evts[:,0]<e)] if einvs[s[2]] not in cond['ignore']] for b,e in zip(stim_samples[:-1],stim_samples[1:])]

    triggers = np.array([get_value([f[0] for f in e if f[2] in cond['triggers']],np.infty) for e in events_by_trial])
    
    response_action = np.array([get_value([rspinvs[f[2]] for f in e if f[2] in cond['responses'].values()],'Stim/S -1') for e in events_by_trial])
    response_time = np.array([get_value([f[0] for f in e if f[2] in cond['responses'].values()])-t for e,t in zip(events_by_trial,triggers)])
    response_valence = np.squeeze([[fdbinvs[f[2]] for f in e if f[2] in cond['feedback'].values()] for e in events_by_trial])
    response_valence = 1*np.array(['positive' in r for r in response_valence])-1*np.array(['negative' in r for r in response_valence])

    md = {'Block_Type':blocks_type,'Block_Index':blocks_idx,'Response_Time':response_time,'Correctness':response_valence,'Action':response_action}
    return pd.DataFrame(md)

def get_raw(filename):
    r = mne.io.read_raw_brainvision(filename, eog=["TVEOG","BVEOG","LHEOG","RHEOG"], preload=True)
    if not 'TP10' in r.info['ch_names']:
        mne.add_reference_channels(r,'TP10',copy=False)
    r.set_eeg_reference(['TP9','TP10'])
    return r

def fix_channels(raw,bads=[]):
    raw.info['bads'] = bads
    raw.interpolate_bads(reset_bads=False)
    return raw

def ica_cleanup(raw,filter=[1,40],decim=10,components=20,tmin=100.):
    ica = mne.preprocessing.ICA(max_pca_components=components).fit(raw.copy().crop(tmin=tmin).filter(*filter),decim=decim);
    a = [None]*4
    a[0]=ica.find_bads_eog(raw,ch_name='BVEOG',start=tmin,stop=np.min([tmin+1000.,raw.times[-1]]))[0]
    a[1]=ica.find_bads_eog(raw,ch_name='TVEOG',start=tmin,stop=np.min([tmin+1000.,raw.times[-1]]))[0]
    a[2]=ica.find_bads_eog(raw,ch_name='RHEOG',start=tmin,stop=np.min([tmin+1000.,raw.times[-1]]))[0]
    a[3]=ica.find_bads_eog(raw,ch_name='LHEOG',start=tmin,stop=np.min([tmin+1000.,raw.times[-1]]))[0]
    ica.exclude = list(set([c for b in a for c in b]))
    return (ica.apply(raw),ica)

def select_dataset(directory,ext_in,ext_out=None,show_all=False,single=False):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn if os.path.isfile(os.path.join(dp, f)) and ext_in in f]
    done = [] if ext_out is None else [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn if os.path.isfile(os.path.join(dp, f)) and ext_out in f]
    subjects = [f[len(directory):-len(ext_in)] for f in files if show_all or ext_out is None or (f[:-len(ext_in)]+ext_out not in done)]
    if single:
        widget = [Dropdown(options=subjects, description='Subject: ',disabled=False)]
    else:
        widget = [Checkbox(True, description=f[len(directory):-len(ext_in)], indent=False) for f in files if show_all or ext_out is None or (f[:-len(ext_in)]+ext_out not in done)]
    return HBox(widget,layout=Layout(width='100%',display='inline-flex',flex_flow='row wrap'))

def get_selection(hbox,fragments=False):
    if type(hbox)==HBox and len(hbox.children):
        if type(hbox.children[0])==Dropdown:
            return [hbox.children[0].value] if not fragments else [c for c in hbox.children[0].options if hbox.children[0].value in c]
        else:
            return [c.description for c in hbox.children if c.value]
    else:
        return None