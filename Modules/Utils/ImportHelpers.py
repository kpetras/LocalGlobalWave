import scipy.io as io 
import pickle

def load_MNE_data(filename):
    import mne
    data = mne.read_epochs(filename)
    return data

def load_MNE_fif_data(filename):
    import mne.io
    data = mne.io.read_raw_fif(filename, preload=True)
    return data

def load_channel_positions(filename):
    with open(filename,'rb') as f:
        ChannelPositions = pickle.load(f)
    #% load contact positions and surfaces
    chanPos = ChannelPositions[4][:-1,0:3]
    #[KP] Fix this
    return chanPos

def load_wavedata_object(filename):
    with open(filename,'rb') as f:
        waveData = pickle.load(f)
    return waveData

def save_wavedata_object(waveData, filename):
    f = open(filename, 'wb')
    pickle.dump(waveData, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load_mat_file(filename):
    
    return io.loadmat(filename)