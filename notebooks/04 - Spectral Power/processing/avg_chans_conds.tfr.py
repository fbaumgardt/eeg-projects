import warnings  # Hide all warnings here
warnings.filterwarnings("ignore")

import mne
mne.utils.set_log_level('error')

from os import listdir
from os.path import isfile, join

#%%
condition = {
    'blocks': [{'begin': 'Stim/S102','end': 'Stim/S104'}],
    'stimuli': ['Stim/S 16','Stim/S 17','Stim/S 18','Stim/S 19'],
    'responses': [],
    'feedback': []
}
d = "../../../data/reinhartlab/multimodal/cg/"
ext = ".raw.fif.gz"
files = [join(d, f) for f in listdir(d) if isfile(join(d, f)) and ext in f]

#%%
freqs = [2,3,4,5,6,7,8,9,11,13,15,17,20,23,26,30,35,40,45,50,60,70,80,90,100]
n_cycles = 5

#%%
tfr = [None]*len(files)
for i,f in enumerate(files):
    # LOAD DATA
    raw = mne.io.read_raw_fif(f,preload=1)
    events,event_id = mne.events_from_annotations(raw)
    event_id = [event_id[k] for k in condition['stimuli']]
    raw.pick(['eeg']).filter(.1,150);
    # LOAD EPOCHS
    picks = mne.pick_types(raw.info,meg=False,eeg=True)
    enr = mne.Epochs(raw,events=events,event_id=event_id,tmin=-2.5,tmax=3,detrend=1,baseline=(-.2,0),preload=True); del raw
    # MAKE TFR BY CHANNEL
    tfr[i] = mne.time_frequency.tfr_morlet(enr,freqs,n_cycles,picks=picks[0:1],return_itc=0,average=0).apply_baseline(mode='mean',baseline=(-.2,0))
    for p in picks[1:]:
        tf = mne.time_frequency.tfr_morlet(enr,freqs,n_cycles,picks=[p],return_itc=0,average=0).apply_baseline(mode='mean',baseline=(-.2,0))
        tfr[i].data+=tf.data
    tfr[i].data/=len(picks) # average over channels
    del enr

#%%
for i,t in enumerate(tfr):
    t.save(files[i][:-len(ext)]+'.aggregate.2to100-tfr.h5')