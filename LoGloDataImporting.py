
#%%
from Modules.Utils import WaveData as wd
from Modules.Utils import ImportHelpers
import mne
import numpy as np
import os
import pandas as pd

#%% Import from MNE-Data
root_dir = '<path_to_data_from_zenodo>'
session2_dirs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'log' in dirnames:
        dirnames.remove('log') # skip 'log' folder
    if 'session2' in dirnames:
        session2_dirpath = os.path.join(dirpath, 'session2')
        session2_dirs.append(session2_dirpath)
files = [
"run01_preproc_raw_tsss.fif",
"run02_preproc_raw_tsss.fif",
"run03_preproc_raw_tsss.fif",
"run04_preproc_raw_tsss.fif",
"run05_preproc_raw_tsss.fif",
"run06_preproc_raw_tsss.fif",
"run07_preproc_raw_tsss.fif",
"run08_preproc_raw_tsss.fif",
"run09_preproc_raw_tsss.fif",
"run10_preproc_raw_tsss.fif"
]

trialDict = {11 : "full trav out", 12 :"full stand", 21: "fov trav out", 22 : "full trav in" }
dimord = "trl_chan_time"

for folder in session2_dirs:
    print ("Loading folder: " + folder)
    stackedEEGData = np.zeros((1,1))
    stackedGradData = np.zeros((1,1))
    stackedMagData = np.zeros((1,1))
    stackedTrialList = [] # type: ignore
    parent_dir = os.path.dirname(folder)
    # make save folder:
    parent_dir = os.path.dirname(folder)
    parent_folder = os.path.basename(parent_dir)    
    # make save folder:
    savefolder = os.path.join('<base_path>', parent_folder)    
    print("Saving to: " + savefolder)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    
    for file in files:      
        print("Loading file: " + file)
        data = ImportHelpers.load_MNE_fif_data(folder + '/' + file)
        #low pass raw data to max 1/3 of desired final sample rate (we are interested in frequencie no higher than 40ish Hz
        # so to be on the safe side we low pass at 80Hz and then downsample to 250Hz)
        h_freq = 80
        data.filter(l_freq=None, h_freq=h_freq)
        current_sfreq = data.info["sfreq"]
        decim = np.round(current_sfreq / 250).astype(int)
        events = mne.find_events(data, "STI101", min_duration = .004)
        events = events[np.where(np.logical_or(np.logical_or(events[:,2]==11, events[:,2]==12), np.logical_or( events[:,2] ==22, events[:,2] ==21)))]
        epochs = mne.Epochs(data, events, decim  = decim, tmin=-1, tmax=2)
        trialList = [trialDict[ttype] for ttype in events[:,2]]
        stackedTrialList = np.concatenate((stackedTrialList, trialList))
        eegData = epochs.get_data(picks="eeg")
        magData = epochs.get_data(picks="mag")
        gradData = epochs.get_data(picks="grad")
        if stackedEEGData.shape == (1,1):
            stackedEEGData = eegData
        else:
            stackedEEGData = np.concatenate((stackedEEGData, eegData), axis=0)
        if stackedGradData.shape == (1,1):
            stackedGradData = gradData
        else: 
            stackedGradData = np.concatenate((stackedGradData, gradData), axis=0)
        if stackedMagData.shape == (1,1): 
            stackedMagData = magData
        else: 
            stackedMagData = np.concatenate((stackedMagData, magData), axis=0)
    #save one epoch file per subject
    epochs.save(savefolder + "/block1-epo.fif")
    #EEG
    EEGwaveData = wd.WaveData()
    EEGwaveData._time = epochs.times
    EEGwaveData.set_trialInfo(stackedTrialList)
    channelPositions = data.get_montage()._get_ch_pos()
    channelPositions = [channelPositions[key][:] for key in channelPositions.keys()]
    channelNames =  list(data.get_montage()._get_ch_pos().keys())
    EEGwaveData.set_channel_positions(np.array(channelPositions))
    EEGwaveData.set_channel_names(channelNames)
    EEGwaveData.set_sample_rate(epochs.info["sfreq"])
    eegDataBucket = wd.DataBucket(stackedEEGData, "EEG", dimord,channelNames, unit="V")
    EEGwaveData.add_data_bucket(eegDataBucket)
    #could not figure out how to find the 3D MEG channel positions in MNE, so I got them from fieldtrip
    MAGpos = pd.read_csv(os.path.join(os.path.dirname(__file__), "MEG_MAG_labels.csv"), delimiter='\t"', header=None, engine='python')
    MAGpos = MAGpos.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    Gradpos = pd.read_csv(os.path.join(os.path.dirname(__file__), "MEG_Grad_labels.csv"), delimiter='\t"', header=None, engine='python')
    Gradpos = Gradpos.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    #MEG gradiometers
    MEG_grad_waveData = wd.WaveData()
    MEG_grad_waveData._time = epochs.times
    MEG_grad_waveData.set_trialInfo(stackedTrialList)
    MEG_grad_waveData.set_sample_rate(epochs.info["sfreq"])
    positions =np.array(([float(x) for x in Gradpos[0]],[float(x) for x in Gradpos[1]], [float(x) for x in Gradpos[2]]))
    channelNames = Gradpos[3]
    MEG_grad_waveData.set_channel_positions(positions.T)
    gradiometerDataBucket = wd.DataBucket(stackedGradData, "Grad", dimord, channelNames, unit ="T/m")
    MEG_grad_waveData.add_data_bucket(gradiometerDataBucket)
    #MEG magnetometers
    MEG_mag_waveData = wd.WaveData()
    MEG_mag_waveData._time = epochs.times
    MEG_mag_waveData.set_trialInfo(stackedTrialList)
    MEG_mag_waveData.set_sample_rate(epochs.info["sfreq"])
    positions =np.array(([float(x) for x in MAGpos[0]],[float(x) for x in MAGpos[1]], [float(x) for x in MAGpos[2]]))
    channelNames = MAGpos[3]
    MEG_mag_waveData.set_channel_positions(positions.T)
    magnetometerDataBucket = wd.DataBucket(stackedMagData, "Mag", dimord, channelNames, unit = "T")
    MEG_mag_waveData.add_data_bucket(magnetometerDataBucket)
    #save wavedata objects to file
    EEGwaveData.save_to_file(savefolder + "/EEGWaveData")
    MEG_grad_waveData.save_to_file(savefolder + "/MEG_Grad_WaveData")
    MEG_mag_waveData.save_to_file(savefolder + "/MEG_Mag_WaveData")



# %%
