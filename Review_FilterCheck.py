# %%
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import glob
import time
import os
from importlib import reload as re
import mne
from Modules.Utils import ImportHelpers
from Modules.PlottingHelpers import Plotting as plotting
from Modules.Preprocessing import Filter as filt
from Modules.Decomposition import Hilbert as hilb
from Modules.SpatialArrangement import SensorLayout
from Modules.WaveAnalysis import OpticalFlow
from Modules.Utils import WaveData as wd, HelperFuns as hf
from scipy import stats
from scipy.signal import welch
from fooof import FOOOF, FOOOFGroup
from fooof.plts.spectra import plot_spectrum
from fooof.sim.gen import gen_aperiodic
from fooof.objs import fit_fooof_3d, combine_fooofs
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm, get_band_peak_fg

#%% helper funs
def plot_data(subplot, freqs, avg_data, single_chan_data, color, label_prefix, title, chan, show_legend=False):
    plt.subplot(subplot)
    plt.plot(freqs, avg_data, color=color, linewidth=2, linestyle='-', label=label_prefix + ' Average')
    plt.plot(freqs, single_chan_data, color=color, linewidth=2, linestyle='--', label=label_prefix + ' Channel ' + str(chan))
    plt.axvline(x=5, color='darkgray', linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.grid(False)
    plt.box(False)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    if show_legend:
        plt.legend(frameon=False)

# %%
folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/LoGlo"
savepath  = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN'

for modality in ['EEG', 'Mag', 'Grad']:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    # Create a list of WaveData files in the folder
    fileName = modality + "WaveData"
    fileList = glob.glob(os.path.join(folder, "**", fileName), recursive=True)
    thetaList =[]
    alphaList =[]
    results = {
        'subject': [],
        'env_band_overlap': [],  
        'envelope_smoothness': []  
    }
    for sub,filePath in enumerate(fileList):
        subname = os.path.basename(os.path.dirname(filePath))    
        savefolder = os.path.join(savepath, subname)
        print("Processing subject: " + str(sub))
        print("Processing file: " + filePath)
        # the wavedata objects
        waveData = ImportHelpers.load_wavedata_object(filePath)

        #remove "fov" trials
        trialInfo = waveData.get_trialInfo()
        fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
        waveData.prune_trials(fovInds)
        #%_______remove! Just for debug________________
        #waveData.DataBuckets['EEG'].set_data(waveData.DataBuckets['EEG'].get_data()[5:50,:,:],'trl_chan_time')
        
        
        #set up a bunch of stuff in the first iteration
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

        #%FFT spectrum 
        chan= 94
        chan = 66
        fft_pre= np.zeros((nChans,windowsize),dtype=complex)
        x = waveData.get_data(modality)[:, :, 0:250]
        fft_pre[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
        fft_pre_chan_AVG = np.mean(fft_pre, axis=0)
        fft_pre_singleChan = fft_pre[chan,:]
        
        fft_post= np.zeros((nChans,windowsize),dtype=complex)
        x = waveData.get_data(modality)[:, :, 500:750]#last second of stimulus
        fft_post[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
        fft_post_chan_AVG = np.mean(fft_post, axis=0)
        fft_post_singleChan = fft_post[chan,:]    
        #% Plot  pre- and post-stim psd
        fig = plt.figure(figsize=(6, 6))
        plot_data(221, fftfreqs[1:40], fft_pre_chan_AVG[1:40], fft_pre_singleChan[1:40], 'darkblue', '', ' Pre-Stimulus',chan = chan, show_legend = False)
        plot_data(222, fftfreqs[1:40], fft_post_chan_AVG[1:40], fft_post_singleChan[1:40],'darkblue', '', ' Post-Stimulus',chan = chan, show_legend = True)
        plt.tight_layout()
        plt.show()  
        
        #find individual alpha from the pre-stim peak
        time_vect = waveData.get_time()
        windowLength = 1 #seconds
        f_sample = waveData.get_sample_rate()
        #initialize arrays to store the spectra
        from scipy.signal import welch
        dataPreStim = waveData.get_data(modality)[:, :, 0:250]
        dataPostStim = waveData.get_data(modality)[:, :, 500:750]

        nTrials, nChans, _ = dataPreStim.shape
        _, Pxx_den = welch(dataPreStim[0, 0, :], fs=f_sample)
        nbins = len(Pxx_den)

        specPreStim = np.zeros((nTrials, nChans, nbins))
        specPostStim = np.zeros((nTrials, nChans, nbins))

        # Compute the power spectra for each trial and channel
        for trial in range(nTrials):
            for chan in range(nChans):
                freqs, Pxx_den = welch(dataPreStim[trial, chan, :], fs=f_sample)
                specPreStim[trial, chan, :] = Pxx_den

                freqs, Pxx_den = welch(dataPostStim[trial, chan, :], fs=f_sample)
                specPostStim[trial, chan, :] = Pxx_den
        # Average the spectra over trials and channels
        avgSpecPreStim = np.mean(specPreStim, axis=(0, 1))
        avgSpecPostStim = np.mean(specPostStim, axis=(0, 1))
        #check the 5 Hz
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [2, 40]
        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        #fm.report(freqs, avgSpecPostStim, freq_range)
        fm.fit(freqs, avgSpecPostStim, [3, 30])

        bands = Bands({'theta' : [4, 8]})
        # Extract any alpha band peaks from the power spectrum model
        theta = get_band_peak_fm(fm, bands.theta)
        print(theta)
        thetaList.append([theta])

        #check pre-stim alpha
        fm = FOOOF()
        # Set the frequency range to fit the model
        freq_range = [2, 40]
        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        #fm.report(freqs, avgSpecPreStim, freq_range)
        fm.fit(freqs, avgSpecPreStim, [3, 30])

        bands = Bands({'alpha' : [8, 12]})
        #get alpha from pre-stim peak
        alpha = get_band_peak_fm(fm, bands.alpha)
        print(alpha)
        if np.any(np.isnan(alpha)):
            alpha = [10]
        if np.any(np.isnan(theta)):
            theta = 5
        #collect all theta and alpha peaks
        alphaList.append([alpha])    
        for freqInd in range(2):
            if freqInd == 0:
                freq = 5  # theta
                center_freq = 5
            else:
                freq = alpha[0]
                center_freq = alpha[0]
            f_low, f_high = freq - 1, freq + 1
        
            # Filter + Hilbert to get complex timeseries
            filt.filter_narrowband(
                waveData, dataBucketName=modality, LowCutOff=freq-1, HighCutOff=freq+1,
                type="FIR", order=100, causal=False
            )
            waveData.DataBuckets[str(freqInd)] = waveData.DataBuckets.pop("NBFiltered")
        
        temp = np.stack(
            (waveData.DataBuckets["0"].get_data(), waveData.DataBuckets["1"].get_data()), axis=0
        )
        waveData.add_data_bucket(
            wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names())
        )
        waveData.set_active_dataBucket('NBFiltered')
        
        # Get complex timeseries
        hilb.apply_hilbert(waveData, dataBucketName="NBFiltered")
        
        # Check quality of phase estimates
        for freqInd in range(2):
            if freqInd == 0:
                center_freq = 5
                f_low, f_high = 4, 6
            else:
                center_freq = alpha[0]
                f_low, f_high = center_freq - 1, center_freq + 1
        
            envelope_spectra = []
            smoothness_vals = []
            for trial in range(nTrials):
                for chan in range(nChans):
                    envelope = np.abs(waveData.get_data("AnalyticSignal")[freqInd, trial, chan, :])
                    f, Pxx = welch(envelope, fs=f_sample)
                    # Percent of envelope power inside band 
                    in_band = (f >= f_low) & (f <= f_high)
                    overlap_ratio = np.sum(Pxx[in_band]) / np.sum(Pxx)
                    envelope_spectra.append(overlap_ratio)
        
                    # Envelope smoothness
                    lowpassed_env = mne.filter.filter_data(envelope, sfreq = f_sample,l_freq=None, h_freq=center_freq / 3)
                    smoothness = np.sqrt(np.mean((envelope - lowpassed_env) ** 2))
                    smoothness_vals.append(smoothness)
        
            results['subject'].append(f'Subj {sub+1}_freq{center_freq:.2f}')
            results['env_band_overlap'].append(np.mean(envelope_spectra))
            results['envelope_smoothness'].append(np.mean(smoothness_vals))


    import matplotlib.pyplot as plt

    subjects = results['subject']
    overlap = results['env_band_overlap']
    smoothness = results['envelope_smoothness']

    fig, ax1 = plt.subplots(figsize=(14, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Subject (freq)')
    ax1.set_ylabel('Envelope Power in Phase Band', color=color)
    ax1.plot(subjects, overlap, 'o-', color=color, label='Env-band overlap')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(subjects)))
    ax1.set_xticklabels(subjects, rotation=90)
    ax1.axhline(0.1, color='red', linestyle='--', label='10% Threshold')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Envelope Smoothness (RMS error)', color=color)
    ax2.plot(subjects, smoothness, 's-', color=color, label='Smoothness')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("Envelope Quality Metrics Across Subjects/Frequencies")
    plt.tight_layout()
    plt.show()


# %%
