import mne
import numpy as np

def preprocess_meg_data(fif_file):
    raw = mne.io.read_raw_fif(fifload_file, preload=True)
    raw.filter(1., 40., fir_design='firwin')
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5, baseline=(None, 0))
    return epochs.get_data()