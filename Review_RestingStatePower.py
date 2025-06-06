#%%
import sys
import os
sys.path.append('/mnt/Data/LoGlo/LocalGlobalWave/LocalGlobalWave/')

import glob
import os
import time
from importlib import reload as re

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import mne
import numpy as np
import scipy.stats as stats
from scipy.signal import coherence
from mne.time_frequency import tfr_array_multitaper, tfr_array_morlet

from Modules.Utils import ImportHelpers, WaveData as wd, HelperFuns as hf
from Modules.PlottingHelpers import Plotting as plotting
from Modules.SpatialArrangement import SensorLayout
import pickle
import itertools
from scipy.stats import mannwhitneyu
from itertools import islice
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from fooof import FOOOF, FOOOFGroup
from fooof.plts.spectra import plot_spectrum
from fooof.sim.gen import gen_aperiodic
from fooof.objs import fit_fooof_3d, combine_fooofs
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm, get_band_peak_fg
#%%
folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/"
figfolder = folder
modalities = ["EEG", "Mag", "Grad"]
conditions = ["EyesClosed", "EyesOpen"]
oscillationThresholdFlag = False
for modality in modalities:
    n_chans = 74 if modality == 'EEG' else 102 if 'Mag' in modality else 204

    file_pattern_closed = os.path.join(folder, "*", f"{modality}_18_OpticalFlowAfterFilter_Hilbert_masked_RestingStateEyesClosed")
    file_pattern_open = os.path.join(folder, "*", f"{modality}_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState")
    files_closed = sorted(glob.glob(file_pattern_closed, recursive=True))
    files_open = sorted(glob.glob(file_pattern_open, recursive=True))
    induced = np.zeros((len(files_open), 3, n_chans, 40))
    alphaList = []
    thetaList = []
    for sub, (file_closed, file_open) in enumerate(zip(files_closed, files_open)):
        dataBucketName = 'UV_Angle'
        wd_open = ImportHelpers.load_wavedata_object(file_open)
        print("length of open file: " + str(wd_open.get_data('NBFiltered').shape[1]))
        wd_closed = ImportHelpers.load_wavedata_object(file_closed)
        print("length of closed file: " + str(wd_closed.get_data('NBFiltered').shape[1]))
        #remove data buckets that are not present in both files and the "Mask" data bucket        
        for key in list(wd_open.DataBuckets.keys()):
            if key not in wd_closed.DataBuckets.keys():
                wd_open.delete_data_bucket(key)
        for key in list(wd_closed.DataBuckets.keys()):
            if key not in wd_open.DataBuckets.keys():
                wd_closed.delete_data_bucket(key)
        Mask = wd_open.get_data('Mask')
        wd_open.delete_data_bucket("Mask")
        wd_closed.delete_data_bucket("Mask")
        #prune any trials above 60
        trials_to_remove = []
        for i in range(wd_open.get_data('NBFiltered').shape[1]):
            if i >= 60:
                trials_to_remove.append(i)
        wd_open.prune_trials(trials_to_remove)
        trials_to_remove = []
        for i in range(wd_closed.get_data('NBFiltered').shape[1]):
            if i >= 60:
                trials_to_remove.append(i)
        wd_closed.prune_trials(trials_to_remove)

        waveData = hf.merge_wavedata_objects([wd_open, wd_closed])
        #add Mask Bucket back
        MaskBucket = wd.DataBucket(Mask, "Mask", 'posx_posy', waveData.get_channel_names())
        waveData.add_data_bucket(MaskBucket)

        time_vect = waveData.get_time()
        windowLength = 1 #seconds
        windowsize = int(np.floor((windowLength*waveData.get_sample_rate()))) #in samples
        fftfreqs = np.fft.fftfreq(windowsize, d=time_vect[1]-time_vect[0])#we use 1 second segments 
        fft_freqEdges = hf.bin_edges_from_centers(fftfreqs)
        #we want the first bin to be centered on 1 and the last on 40 Hz
        freqMin = 1 - ((fftfreqs[1]-fftfreqs[0])/2)
        freqMax = 40 + ((fftfreqs[1]-fftfreqs[0])/2)
        nbins = hf.find_nearest(fft_freqEdges, freqMax)[0] - hf.find_nearest(fft_freqEdges, freqMin)[0]
        nChans = waveData.get_data(modality).shape[1]   
        waveData.set_trialInfo(['eyes open'] * 60 + ['eyes closed'] * 60)
        trialInfo = np.array(waveData.get_trialInfo())
        unique_conditions = np.unique(trialInfo)
        f_sample = waveData.get_sample_rate()
        for cond_idx, cond in enumerate(unique_conditions):
            cond_mask = (trialInfo == cond)
            x= waveData.get_data(modality)[cond_mask, :, :]  
            nTrials_cond, nChans, nTimes = x.shape
            psds = []
            for trial in range(nTrials_cond):
                trial_psds = []
                for chan in range(nChans):
                    freqs, Pxx = welch(x[trial, chan, :], fs=f_sample)
                    trial_psds.append(Pxx)
                psds.append(trial_psds)
            psds = np.array(psds) 
            mean_psd = np.mean(psds, axis=0)  
            induced[sub, cond_idx, :, :] = mean_psd[:, 1:41]
        #find individual alpha 
        time_vect = waveData.get_time()
        windowLength = 1 #seconds
        #initialize arrays to store the spectra
        data = waveData.get_data(modality)

        nTrials, nChans, _ = data.shape
        #get number of bins
        freqs, Pxx_den = welch(data[0, 0, :], fs=f_sample)
        nbins = len(Pxx_den)
 
        spec = np.zeros((nTrials, nChans, nbins))
        # power spectra for each trial and channel (all conds)
        for trial in range(nTrials):
            for chan in range(nChans):
                freqs, Pxx_den = welch(data[trial, chan, :], fs=f_sample)
                spec[trial, chan, :] = Pxx_den

        # Average the spectra over trials and channels
        avgSpec = np.mean(spec, axis=(0, 1))        
        #PostStim
        fm = FOOOF()
        freq_range = [2, 40]
        #fm.report(freqs, avgSpecPostStim, freq_range)
        fm.fit(freqs, avgSpec, [3, 30])
        bands = Bands({'theta' : [4, 8]})
        # theta peaks 
        theta_peak= get_band_peak_fm(fm, bands.theta)
        thetaList.append([theta_peak])
        bands = Bands({'alpha' : [8, 12]})
        alpha_peak= get_band_peak_fm(fm, bands.alpha)
        alphaList.append([alpha_peak])

        np.save(os.path.join(folder, f"{modality}_Resting_state_power_induced_open_closed.npy"), induced)
        #save alphalist
        np.save(os.path.join(folder, f"{modality}_Resting_state_power_alpha.npy"), alphaList)
        np.save(os.path.join(folder, f"{modality}_Resting_state_power_theta.npy"), thetaList)



# %%
modalities = ["EEG", "Mag", "Grad"]
conditions = ["Eyes Open", "Eyes Closed"]
colors = ['#1f77b4', '#ff7f0e']

for modality in modalities:
    # Load saved data
    induced = np.load(os.path.join(folder, f"{modality}_Resting_state_power_induced_open_closed.npy"))
    alphaList = np.load(os.path.join(folder, f"{modality}_Resting_state_power_alpha.npy"), allow_pickle=True)
    thetaList = np.load(os.path.join(folder, f"{modality}_Resting_state_power_theta.npy"), allow_pickle=True)
    # Use the frequency vector from Welch
    # Assuming you used freqs[1:41] for PSDs
    f_sample = 500  # or whatever your sample rate is
    nperseg = 256   # or whatever you used in welch
    freqs = np.fft.rfftfreq(nperseg, 1/f_sample)[1:41]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond_idx, cond in enumerate(conditions):
        # Average over subjects and channels
        psd_mean = np.mean(induced[:, cond_idx, :, :], axis=(0, 1))
        psd_sem = np.std(induced[:, cond_idx, :, :], axis=(0, 1)) / np.sqrt(induced.shape[0])
        ax.plot(freqs, psd_mean, label=cond, color=colors[cond_idx])
        ax.fill_between(freqs, psd_mean - psd_sem, psd_mean + psd_sem, color=colors[cond_idx], alpha=0.2)

    # Overlay alpha and theta peaks for each subject
    for subj, (alpha, theta) in enumerate(zip(alphaList, thetaList)):
        if not np.isnan(alpha[0]):
            ax.scatter(alpha[0], 0, marker='^', color='purple', s=60, label='Alpha peak' if subj == 0 else "")
        if not np.isnan(theta[0]):
            ax.scatter(theta[0], 0, marker='v', color='green', s=60, label='Theta peak' if subj == 0 else "")

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (a.u.)')
    ax.set_title(f'{modality} Resting-State Power Spectra')
    ax.legend()
    ax.set_xlim([1, 40])
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{modality}_Resting_state_power_spectra.png"))
    plt.show()