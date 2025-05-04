
from Modules.Utils import WaveData as wd
from Modules.Utils import ImportHelpers
import mne
import numpy as np
import os
import pandas as pd

#%% Import from MNE-Data
root_dir = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel/Experiments/Data/data_MEEG/preproc/'
session2_dirs = []
savePath = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/'
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'log' in dirnames:
        dirnames.remove('log') # skip 'log' folder
    if 'session2' in dirnames:
        session2_dirpath = os.path.join(dirpath, 'session2')
        session2_dirs.append(session2_dirpath)
file ="resting_stateYF_preproc_raw_tsss.fif"
#remove folder 90WCLR (that one doesn't have all the task data) from session2dirs
session2_dirs = [folder for folder in session2_dirs if "90WCLR" not in folder]

dimord = "trl_chan_time"

for folder in session2_dirs:
    print ("Loading folder: " + folder)
    EEGData = np.zeros((1,1))
    GradData = np.zeros((1,1))
    MagData = np.zeros((1,1))
    stackedTrialList = [] # type: ignore
    parent_dir = os.path.dirname(folder)
    # make save folder:
    parent_dir = os.path.dirname(folder)
    parent_folder = os.path.basename(parent_dir)    
    # make save folder:
    savefolder = os.path.join(savePath, parent_folder)    
    print("Saving to: " + savefolder)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    print("Loading file: " + file)
    data = ImportHelpers.load_MNE_fif_data(folder + '/' + file)
    #low pass raw data to max 1/3 of desired final sample rate (we are interested in frequencie no higher than 40ish Hz
    # so to be on the safe side we low pass at 80Hz and then downsample to 250Hz)
    h_freq = 80
    data.filter(l_freq=None, h_freq=h_freq)
    current_sfreq = data.info["sfreq"]
    decim = np.round(current_sfreq / 250).astype(int)
    epochs = mne.make_fixed_length_epochs(data, duration=3, preload=False)
    epochs.decimate(decim, verbose=True)
    eegData = epochs.get_data(picks="eeg")
    magData = epochs.get_data(picks="mag")
    gradData = epochs.get_data(picks="grad")
    EEGData = eegData
    GradData = gradData
    MagData = magData

    #save one epoch file per subject
    #EEG
    EEGwaveData = wd.WaveData()
    EEGwaveData._time = epochs.times
    channelPositions = data.get_montage()._get_ch_pos()
    channelPositions = [channelPositions[key][:] for key in channelPositions.keys()]
    channelNames =  list(data.get_montage()._get_ch_pos().keys())
    EEGwaveData.set_channel_positions(np.array(channelPositions))
    EEGwaveData.set_channel_names(channelNames)
    EEGwaveData.set_sample_rate(epochs.info["sfreq"])
    eegDataBucket = wd.DataBucket(EEGData, "EEG", dimord,channelNames, unit="V")
    EEGwaveData.add_data_bucket(eegDataBucket)
    #could not figure out how to find the 3D MEG channel positions in MNE, so I got them from fieldtrip
    MAGpos = pd.read_csv(os.path.join(os.path.dirname(__file__),"LocalGlobalWave", "MEG_MAG_labels.csv"), delimiter='\t"', header=None, engine='python')
    MAGpos = MAGpos.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    Gradpos = pd.read_csv(os.path.join(os.path.dirname(__file__),"LocalGlobalWave", "MEG_Grad_labels.csv"), delimiter='\t"', header=None, engine='python')
    Gradpos = Gradpos.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    #MEG gradiometers
    MEG_grad_waveData = wd.WaveData()
    MEG_grad_waveData._time = epochs.times
    MEG_grad_waveData.set_sample_rate(epochs.info["sfreq"])
    positions =np.array(([float(x) for x in Gradpos[0]],[float(x) for x in Gradpos[1]], [float(x) for x in Gradpos[2]]))
    channelNames = Gradpos[3]
    MEG_grad_waveData.set_channel_positions(positions.T)
    gradiometerDataBucket = wd.DataBucket(GradData, "Grad", dimord, channelNames, unit ="T/m")
    MEG_grad_waveData.add_data_bucket(gradiometerDataBucket)
    #MEG magnetometers
    MEG_mag_waveData = wd.WaveData()
    MEG_mag_waveData._time = epochs.times
    MEG_mag_waveData.set_sample_rate(epochs.info["sfreq"])
    positions =np.array(([float(x) for x in MAGpos[0]],[float(x) for x in MAGpos[1]], [float(x) for x in MAGpos[2]]))
    channelNames = MAGpos[3]
    MEG_mag_waveData.set_channel_positions(positions.T)
    magnetometerDataBucket = wd.DataBucket(MagData, "Mag", dimord, channelNames, unit = "T")
    MEG_mag_waveData.add_data_bucket(magnetometerDataBucket)
    #save wavedata objects to file
    EEGwaveData.save_to_file(savefolder + "/EEGWaveData_RestingState_EyesClosed")
    MEG_grad_waveData.save_to_file(savefolder + "/MEG_Grad_WaveData_RestingState_EyesClosed")
    MEG_mag_waveData.save_to_file(savefolder + "/MEG_Mag_WaveData_RestingState_EyesClosed")



# %%
