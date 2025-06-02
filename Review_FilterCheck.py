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

for modality in ['EEG', 'MEG_Mag_', 'MEG_Grad_']:
    results = []
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    # Create a list of WaveData files in the folder
    fileName = modality + "WaveData"
    fileList = glob.glob(os.path.join(folder, "**", fileName), recursive=True)
    thetaList =[]
    alphaList =[]
    n_chans = 74 if modality == 'EEG' else 102 if modality == 'MEG_Mag_' else 204
    induced_pre = np.zeros((len(fileList), n_chans, 40))
    induced_post = np.zeros((len(fileList), n_chans, 40))
    evoked_pre = np.zeros((len(fileList), n_chans, 40))
    evoked_post = np.zeros((len(fileList), n_chans, 40))
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

        #%FFT spectrum (induced)
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
        plot_data(221, fftfreqs[1:40], fft_pre_chan_AVG[1:40], fft_pre_singleChan[1:40], 'darkblue', '', ' Pre-Stimulus induced',chan = chan, show_legend = False)
        plot_data(222, fftfreqs[1:40], fft_post_chan_AVG[1:40], fft_post_singleChan[1:40],'darkblue', '', ' Post-Stimulus induced',chan = chan, show_legend = True)
        plt.tight_layout()
        plt.show()  
        induced_pre[sub, :, :]  = fft_pre[:, 1:41]
        induced_post[sub, :, :] = fft_post[:, 1:41]

        evoked_data_pre = np.mean(waveData.get_data(modality)[:, :, 0:250], axis=0)  # shape: (nChans, time)
        evoked_data_post = np.mean(waveData.get_data(modality)[:, :, 500:750], axis=0)
        # FFT of the evoked signal
        evoked_fft_pre = np.abs(np.fft.fft(evoked_data_pre, axis=1) / evoked_data_pre.shape[1])
        evoked_fft_post = np.abs(np.fft.fft(evoked_data_post, axis=1) / evoked_data_post.shape[1])
        evoked_fft_pre_chan_AVG = np.mean(evoked_fft_pre, axis=0)
        evoked_fft_post_chan_AVG = np.mean(evoked_fft_post, axis=0)
        evoked_pre[sub,:,:] = evoked_fft_pre[:, 1:41]
        evoked_post[sub,:,:] = evoked_fft_post[:, 1:41]

        fig = plt.figure(figsize=(6, 6))
        plot_data(221, fftfreqs[1:40], evoked_fft_pre_chan_AVG[1:40], evoked_fft_pre[chan, 1:40], 'darkblue', '', ' Pre-Stimulus evoked', chan=chan, show_legend=False)
        plot_data(222, fftfreqs[1:40], evoked_fft_post_chan_AVG[1:40], evoked_fft_post[chan, 1:40], 'darkblue', '', ' Post-Stimulus evoked', chan=chan, show_legend=True)
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

        #spectrum of the average
        evokedPreStim = np.mean(dataPreStim, axis=0)
        evokedPostStim = np.mean(dataPostStim, axis=0)
        freqs, Pxx_evoked_avg = welch(evokedPreStim, fs=f_sample)
        freqs, Pxx_den_evoked_post = welch(evokedPostStim, fs=f_sample)

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
        #PostStim
        fm = FOOOF()
        freq_range = [2, 40]
        #fm.report(freqs, avgSpecPostStim, freq_range)
        fm.fit(freqs, avgSpecPostStim, [3, 30])
        bands = Bands({'theta' : [4, 8]})
        # theta peaks 
        theta_peak_post = get_band_peak_fm(fm, bands.theta)
        print(theta)
        thetaList.append([theta])
        bands = Bands({'alpha' : [8, 12]})
        alpha_peak_post = get_band_peak_fm(fm, bands.alpha)


        #check pre-stim alpha
        fm = FOOOF()
        freq_range = [2, 40]
        #fm.report(freqs, avgSpecPreStim, freq_range)
        fm.fit(freqs, avgSpecPreStim, [3, 30])

        bands = Bands({'alpha' : [8, 12]})
        #get alpha from pre-stim peak
        alpha_peak_pre = get_band_peak_fm(fm, bands.alpha)
        bands = Bands({'theta' : [4, 8]})
        theta_peak_pre = get_band_peak_fm(fm, bands.theta)
        
        
        if np.any(np.isnan(alpha_peak_pre)):
            alpha = [10]
        if np.any(np.isnan(theta_peak_post)):
            theta = 5
        #collect 
        alphaList.append([alpha_peak_pre])    
        for freqInd in range(2):
            if freqInd == 0:
                freq = 5  # theta
                center_freq = 5
            else:
                freq = alpha_peak_pre[0]
                center_freq = alpha_peak_pre[0]
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
        
        for freqInd, center_freq in enumerate([5, alphaList[-1][0][0]]):
            f_low, f_high = center_freq - 1, center_freq + 1

            analytic_signal = waveData.get_data("AnalyticSignal")[freqInd]
            for trial in range(analytic_signal.shape[0]):
                for chan in range(nChans):
                    signal = analytic_signal[trial, chan]
                    envelope = np.abs(signal)

                    # Envelope spectrum
                    f, Pxx = welch(envelope, fs=f_sample, nperseg=min(1024, len(envelope)))
                    in_band = (f >= f_low) & (f <= f_high)
                    overlap_ratio = np.sum(Pxx[in_band]) / np.sum(Pxx)
                    
                    # Phase correlation
                    phase = np.angle(signal)
                    phase_env_corr = np.corrcoef(envelope, np.cos(phase))[1, 0]
                    dphase = np.diff(np.unwrap(phase))
                    inst_freq = dphase * f_sample / (2 * np.pi)

                    # Envelope smoothness
                    lowpassed_env = mne.filter.filter_data(envelope, sfreq=f_sample, l_freq=None,
                                                h_freq=center_freq / 3, verbose=False)
                    smoothness = np.sqrt(np.mean((envelope - lowpassed_env) ** 2))

                    discontinuity_index = np.std(np.abs(dphase))
                    
                    results.append({
                        'Subject': subname,
                        'Frequency': center_freq,
                        'Channel': chan,
                        'EnvelopeSpectralOverlap': overlap_ratio,
                        'EnvelopeSmoothness': smoothness,
                        'PhaseDiscontinuity': discontinuity_index,
                        'PhaseEnvCorr': phase_env_corr,
                        'InstFreqMean': np.mean(inst_freq),
                        'InstFreqStd': np.std(inst_freq),
                        'AlphaPeakPre': alpha_peak_pre,
                        'ThetaPeakPre': theta_peak_pre,
                        'AlphaPeakPost': alpha_peak_post,
                        'ThetaPeakPost': theta_peak_post
                    })

    # Convert to DataFrame if not already
    df = pd.DataFrame(results)
    #write to csv
    df.to_csv(os.path.join(savepath, modality + 'PhaseEstimateQualityMetrics.csv'), index=False)
    #save psd data 
    np.save(os.path.join(savepath, modality + 'InducedPre.npy'), induced_pre)
    np.save(os.path.join(savepath, modality + 'InducedPost.npy'), induced_post)
    np.save(os.path.join(savepath, modality + 'EvokedPre.npy'), evoked_pre)
    np.save(os.path.join(savepath, modality + 'EvokedPost.npy'), evoked_post)

    lkfhl
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
for band in ['AlphaPeakPre', 'ThetaPeakPre', 'AlphaPeakPost', 'ThetaPeakPost']:
    df[[f'{band}_Freq', f'{band}_Power', f'{band}_BW']] = pd.DataFrame(df[band].tolist(), index=df.index)
# Metrics and thresholds
metrics_info = {
    'EnvelopeSpectralOverlap': {'threshold': 0.1, 'color': 'red', 'label': 'Max recommended overlap'},
    'EnvelopeSmoothness': {'threshold': 0.01, 'color': 'green', 'label': 'Smoothness ceiling'},
    'PhaseDiscontinuity': {'threshold': 0.1, 'color': 'purple', 'label': 'Discontinuity tolerance'},
    'PhaseEnvCorr': {'threshold': 0.1, 'color': 'gray', 'label': 'Corr should be ≈ 0'},
    'InstFreqStd': {'threshold': 1.0, 'color': 'orange', 'label': 'Excess jitter warning'},
    'AlphaPeakPost_Power': {'threshold': 0.5, 'color': 'brown', 'label': 'Min power'},
    'AlphaPeakPost_BW': {'threshold': 4.0, 'color': 'blue', 'label': 'Max bandwidth'}
}


# Plotting
plt.figure(figsize=(4 * len(metrics_info), 6))  # Wider figure to fit all

for i, (metric, info) in enumerate(metrics_info.items()):
    plt.subplot(1, len(metrics_info), i + 1)
    sns.boxplot(data=df, x='Frequency', y=metric, hue='Modality', showfliers=False)
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.axhline(info['threshold'], color=info['color'], linestyle='--', linewidth=1, label=info['label'])
    plt.legend(fontsize='small')

plt.suptitle("Phase & Spectral Quality Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
