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
import pandas as pd
#%% helper funs
def plot_data(subplot, freqs, avg_data=None, single_chan_data=None, color=None, label_prefix=None, title=None, chan=None, 
              avg_ci=None, single_chan_ci=None, show_legend=False):
    plt.subplot(subplot)
    if avg_data is not None:
        plt.plot(freqs, avg_data, color=color, linewidth=2, linestyle='--', label=label_prefix + ' Average')
    if avg_ci is not None:
        plt.fill_between(freqs, avg_data - avg_ci, avg_data + avg_ci, color=color, alpha=0.3)
    if single_chan_data is not None:
        plt.plot(freqs, single_chan_data, color=color, linewidth=2, linestyle='-.', label=label_prefix + ' Channel ' + str(chan))
    if single_chan_ci is not None:
        plt.fill_between(freqs, single_chan_data - single_chan_ci, single_chan_data + single_chan_ci, color=color, alpha=0.1)
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
    if modality == 'MEG_Mag_':
        modality = 'Mag'
    if modality == 'MEG_Grad_':
        modality = 'Grad'
    unique_conditions = ['full trav in', 'full trav out','full stand']
    induced_pre = np.zeros((len(fileList), 3, n_chans, 40))
    induced_post = np.zeros((len(fileList), 3, n_chans, 40))
    evoked_pre = np.zeros((len(fileList), 3, n_chans, 40))
    evoked_post = np.zeros((len(fileList), 3, n_chans, 40))
    
    induced_pre_welch = np.zeros((len(fileList), 3, n_chans, 40))
    induced_post_welch = np.zeros((len(fileList), 3, n_chans, 40))
    evoked_pre_welch = np.zeros((len(fileList), 3, n_chans, 40))
    evoked_post_welch = np.zeros((len(fileList), 3, n_chans, 40))
    for sub,filePath in enumerate(fileList):
        subname = os.path.basename(os.path.dirname(filePath))    
        savefolder = os.path.join(savepath, subname)
        print("Processing subject: " + str(sub))
        print("Processing file: " + filePath)
        # the wavedata objects
        waveData = ImportHelpers.load_wavedata_object(filePath)
        fsample = waveData.get_sample_rate()
        #remove "fov" trials
        trialInfo = waveData.get_trialInfo()
        fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
        waveData.prune_trials(fovInds)
        trialInfo = waveData.get_trialInfo()
        #%_______remove! Just for debug________________
        #waveData.DataBuckets['EEG'].set_data(waveData.DataBuckets['EEG'].get_data()[5:50,:,:],'trl_chan_time')
        
        if sub == 0:
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
        
        trialInfo = np.array([str(t).strip() for t in waveData.get_trialInfo()])
        for cond_idx, cond in enumerate(unique_conditions):
            cond_mask = (trialInfo == cond)
            if not np.any(cond_mask):
                continue
            else: 
                print(f"Processing condition: {cond} for subject {subname}")

            # Induced: FFT per trial, then average over trials for this condition
            x_pre = waveData.get_data(modality)[cond_mask, :, 0:250]  # (nTrials_cond, nChans, time)
            x_post = waveData.get_data(modality)[cond_mask, :, 500:750]
            # FFT: mean over trials, then abs, then mean over channels
            fft_pre = np.mean(np.abs(np.fft.fft(x_pre, axis=2) / x_pre.shape[2]), axis=0)  # (nChans, freq)
            fft_post = np.mean(np.abs(np.fft.fft(x_post, axis=2) / x_post.shape[2]), axis=0)
            induced_pre[sub, cond_idx, :, :] = fft_pre[:, 1:41]
            induced_post[sub, cond_idx, :, :] = fft_post[:, 1:41]
            #KP Test
            x_pre = waveData.get_data(modality)[cond_mask, :, 0:250]  # (nTrials_cond, nChans, time)
            x_post = waveData.get_data(modality)[cond_mask, :, 500:750]

            nTrials_cond, nChans, nTimes = x_pre.shape
            psd_pre = np.zeros((nTrials_cond, nChans, 41))   # 0-40 Hz (assuming fs >= 80 Hz)
            psd_post = np.zeros((nTrials_cond, nChans, 41))

            for trial in range(nTrials_cond):
                for chan in range(nChans):
                    freqs, Pxx_pre = welch(x_pre[trial, chan, :], fs=f_sample, nperseg=min(256, nTimes))
                    freqs, Pxx_post = welch(x_post[trial, chan, :], fs=f_sample, nperseg=min(256, nTimes))
                    # Store only 0-40 Hz (assuming freqs[0:41] covers 0-40 Hz)
                    psd_pre[trial, chan, :] = Pxx_pre[:41]
                    psd_post[trial, chan, :] = Pxx_post[:41]

            # Average over trials (induced power)
            induced_pre_welch[sub, cond_idx, :, :] = np.mean(psd_pre, axis=0)
            induced_post_welch[sub, cond_idx, :, :] = np.mean(psd_post, axis=0)
            #KP test end

            # Plot induced for this condition
            # fig = plt.figure(figsize=(6, 6))
            # fig.suptitle(f'Subject {subname} - Condition: {cond}')
            # plot_data(221, fftfreqs[1:40], np.mean(fft_pre, axis=0)[1:40], fft_pre[chan, 1:40], 'darkblue', '', 'Pre-Stimulus induced', chan=chan, show_legend=False)
            # plot_data(222, fftfreqs[1:40], np.mean(fft_post, axis=0)[1:40], fft_post[chan, 1:40], 'darkblue', '', 'Post-Stimulus induced', chan=chan, show_legend=True)
            # plt.tight_layout()
            # plt.show()

            # Evoked: average over trials, then FFT
            evoked_data_pre = np.mean(waveData.get_data(modality)[cond_mask, :, 0:250], axis=0)  # (nChans, time)
            evoked_data_post = np.mean(waveData.get_data(modality)[cond_mask, :, 500:750], axis=0)
            evoked_fft_pre = np.abs(np.fft.fft(evoked_data_pre, axis=1) / evoked_data_pre.shape[1])  # (nChans, freq)
            evoked_fft_post = np.abs(np.fft.fft(evoked_data_post, axis=1) / evoked_data_post.shape[1])
            evoked_pre[sub, cond_idx, :, :] = evoked_fft_pre[:, 1:41]
            evoked_post[sub, cond_idx, :, :] = evoked_fft_post[:, 1:41]

            evoked_data_pre = np.mean(x_pre, axis=0)   # (nChans, time)
            evoked_data_post = np.mean(x_post, axis=0) # (nChans, time)
            psd_evoked_pre = np.zeros((nChans, 41))
            psd_evoked_post = np.zeros((nChans, 41))
            for chan in range(nChans):
                freqs, Pxx_evoked_pre = welch(evoked_data_pre[chan, :], fs=f_sample, nperseg=min(256, nTimes))
                freqs, Pxx_evoked_post = welch(evoked_data_post[chan, :], fs=f_sample, nperseg=min(256, nTimes))
                psd_evoked_pre[chan, :] = Pxx_evoked_pre[:41]
                psd_evoked_post[chan, :] = Pxx_evoked_post[:41]
            evoked_pre_welch[sub, cond_idx, :, :] = psd_evoked_pre
            evoked_post_welch[sub, cond_idx, :, :] = psd_evoked_post

    #         # Plot evoked for this condition
    #         # fig = plt.figure(figsize=(6, 6))
    #         # fig.suptitle(f'Subject {subname} - Condition: {cond}')
    #         # plot_data(221, fftfreqs[1:40], np.mean(evoked_fft_pre, axis=0)[1:40], evoked_fft_pre[chan, 1:40], 'darkred', '', 'Pre-Stimulus evoked', chan=chan, show_legend=False)
    #         # plot_data(222, fftfreqs[1:40], np.mean(evoked_fft_post, axis=0)[1:40], evoked_fft_post[chan, 1:40], 'darkred', '', 'Post-Stimulus evoked', chan=chan, show_legend=True)
    #         # plt.tight_layout()
    #         # plt.show()

        
    #     #find individual alpha from the pre-stim peak
    #     time_vect = waveData.get_time()
    #     windowLength = 1 #seconds
    #     f_sample = waveData.get_sample_rate()
    #     #initialize arrays to store the spectra
    #     from scipy.signal import welch
    #     dataPreStim = waveData.get_data(modality)[:, :, 0:250]
    #     dataPostStim = waveData.get_data(modality)[:, :, 500:750]

    #     nTrials, nChans, _ = dataPreStim.shape
    #     _, Pxx_den = welch(dataPreStim[0, 0, :], fs=f_sample)
    #     nbins = len(Pxx_den)

    #     #spectrum of the average
    #     evokedPreStim = np.mean(dataPreStim, axis=0)
    #     evokedPostStim = np.mean(dataPostStim, axis=0)
    #     freqs, Pxx_evoked_avg = welch(evokedPreStim, fs=f_sample)
    #     freqs, Pxx_den_evoked_post = welch(evokedPostStim, fs=f_sample)

    #     specPreStim = np.zeros((nTrials, nChans, nbins))
    #     specPostStim = np.zeros((nTrials, nChans, nbins))       

    #     # Compute the power spectra for each trial and channel 
    #     for trial in range(nTrials):
    #         for chan in range(nChans):
    #             freqs, Pxx_den = welch(dataPreStim[trial, chan, :], fs=f_sample)
    #             specPreStim[trial, chan, :] = Pxx_den

    #             freqs, Pxx_den = welch(dataPostStim[trial, chan, :], fs=f_sample)
    #             specPostStim[trial, chan, :] = Pxx_den
    #     # Average the spectra over trials and channels
    #     avgSpecPreStim = np.mean(specPreStim, axis=(0, 1))
    #     avgSpecPostStim = np.mean(specPostStim, axis=(0, 1))
    #     #PostStim
    #     fm = FOOOF()
    #     freq_range = [2, 40]
    #     #fm.report(freqs, avgSpecPostStim, freq_range)
    #     fm.fit(freqs, avgSpecPostStim, [3, 30])
    #     bands = Bands({'theta' : [4, 8]})
    #     # theta peaks 
    #     theta_peak_post = get_band_peak_fm(fm, bands.theta)
    #     thetaList.append([theta_peak_post])
    #     bands = Bands({'alpha' : [8, 12]})
    #     alpha_peak_post = get_band_peak_fm(fm, bands.alpha)


    #     #check pre-stim alpha
    #     fm = FOOOF()
    #     freq_range = [2, 40]
    #     #fm.report(freqs, avgSpecPreStim, freq_range)
    #     fm.fit(freqs, avgSpecPreStim, [3, 30])

    #     bands = Bands({'alpha' : [8, 12]})
    #     #get alpha from pre-stim peak
    #     alpha_peak_pre = get_band_peak_fm(fm, bands.alpha)
    #     bands = Bands({'theta' : [4, 8]})
    #     theta_peak_pre = get_band_peak_fm(fm, bands.theta)
        
        
    #     if np.any(np.isnan(alpha_peak_pre)):
    #         alpha_peak_pre = [10]
    #     if np.any(np.isnan(theta_peak_post)):
    #         theta_peak_post = 5
    #     #collect 
    #     alphaList.append([alpha_peak_pre])    
    #     for freqInd in range(2):
    #         if freqInd == 0:
    #             freq = 5  # theta
    #             center_freq = 5
    #         else:                    
    #             freq = alpha_peak_pre[0]
    #             center_freq = alpha_peak_pre[0]
    #         f_low, f_high = freq - 1, freq + 1
        
    #         # Filter + Hilbert to get complex timeseries
    #         filt.filter_narrowband(
    #             waveData, dataBucketName=modality, LowCutOff=freq-1, HighCutOff=freq+1,
    #             type="FIR", order=100, causal=False
    #         )
    #         waveData.DataBuckets[str(freqInd)] = waveData.DataBuckets.pop("NBFiltered")
        
    #     temp = np.stack(
    #         (waveData.DataBuckets["0"].get_data(), waveData.DataBuckets["1"].get_data()), axis=0
    #     )
    #     waveData.add_data_bucket(
    #         wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names())
    #     )
    #     waveData.set_active_dataBucket('NBFiltered')
        
    #     # Get complex timeseries
    #     hilb.apply_hilbert(waveData, dataBucketName="NBFiltered")
        
    #     for freqInd, center_freq in enumerate([5, alphaList[-1][0][0]]):
    #         f_low, f_high = center_freq - 1, center_freq + 1
    #         analytic_signal = waveData.get_data("AnalyticSignal")[freqInd]
    #         time_vect = waveData.get_time()
    #         pre_mask = (time_vect >= -0.7) & (time_vect < 0)
    #         stim_mask = (time_vect >= .25) & (time_vect <= 1.7)

    #         for trial in range(analytic_signal.shape[0]):
    #             for chan in range(nChans):
    #                 signal = analytic_signal[trial, chan]
    #                 envelope_full = np.abs(signal)
    #                 lowpassed_env_full = mne.filter.filter_data(
    #                     envelope_full, sfreq=f_sample, l_freq=None,
    #                     h_freq=center_freq / 3, verbose=False
    #                 )
    #                 phase_full = np.angle(signal)
    #                 dphase_full = np.diff(np.unwrap(phase_full))
    #                 inst_freq_full = dphase_full * f_sample / (2 * np.pi)
            
    #                 # Envelope spectrum 
    #                 f_env, Pxx_env = welch(envelope_full, fs=f_sample, nperseg=min(1024, len(envelope_full)))
    #                 in_band = (f_env >= f_low) & (f_env <= f_high)
    #                 overlap_ratio_full = np.sum(Pxx_env[in_band]) / np.sum(Pxx_env)
            
    #                 # split into preStim and Stim 
    #                 for period, mask in zip(['preStim', 'Stim'], [pre_mask, stim_mask]):

    #                     envelope = envelope_full[mask]
    #                     phase = phase_full[mask]
    #                     dphase = np.diff(np.unwrap(phase))
    #                     inst_freq = dphase * f_sample / (2 * np.pi)
    #                     lowpassed_env = lowpassed_env_full[mask]
    #                     smoothness = np.sqrt(np.mean((envelope - lowpassed_env) ** 2))
    #                     phase_env_corr = np.corrcoef(envelope, np.cos(phase))[1, 0]
    #                     discontinuity_index = np.std(np.abs(dphase))
            
    #                     results.append({
    #                         'Subject': subname,
    #                         'Frequency': center_freq,
    #                         'Channel': chan,
    #                         'Period': period,
    #                         'EnvelopeSpectralOverlap': overlap_ratio_full,  # Use full-signal value for both periods
    #                         'EnvelopeSmoothness': smoothness,
    #                         'PhaseDiscontinuity': discontinuity_index,
    #                         'PhaseEnvCorr': phase_env_corr,
    #                         'InstFreqMean': np.mean(inst_freq),
    #                         'InstFreqStd': np.std(inst_freq),
    #                         'AlphaPeakPre': alpha_peak_pre,
    #                         'ThetaPeakPre': theta_peak_pre,
    #                         'AlphaPeakPost': alpha_peak_post,
    #                         'ThetaPeakPost': theta_peak_post
    #                     })

    # # Convert to DataFrame if not already
    # df = pd.DataFrame(results)
    # #write to csv
    # df.to_csv(os.path.join(savepath, modality + 'PhaseEstimateQualityMetrics.csv'), index=False)
    # #save psd data 
    # np.save(os.path.join(savepath, modality + 'InducedPre.npy'), induced_pre)
    # np.save(os.path.join(savepath, modality + 'InducedPost.npy'), induced_post)
    np.save(os.path.join(savepath, modality + 'IndicedPreWelch.npy'), induced_pre_welch)
    np.save(os.path.join(savepath, modality + 'IndicedPostWelch.npy'), induced_post_welch)
    # np.save(os.path.join(savepath, modality + 'EvokedPre.npy'), evoked_pre)
    # np.save(os.path.join(savepath, modality + 'EvokedPost.npy'), evoked_post)
    np.save(os.path.join(savepath, modality + 'EvokedPreWelch.npy'), evoked_pre_welch)
    np.save(os.path.join(savepath, modality + 'EvokedPostWelch.npy'), evoked_post_welch)

#%%
unique_conditions = ['full trav in', 'full trav out','full stand']
for modality in ['EEG', 'Mag', 'Grad']:
    #load the csv and data
    df = pd.read_csv(os.path.join(savepath, modality +'PhaseEstimateQualityMetrics.csv'))
    # Add column for freuqband 
    df['Band'] = df['Frequency'].apply(lambda x: 'Theta (5 Hz)' if np.isclose(x, 5, atol = 1.5) else 'Alpha (~10 Hz)')

    df_subject = (
        df.groupby(['Subject', 'Band', 'Period'], as_index=False)
        .mean(numeric_only=True)
    )

    # thresholds for reliability 
    thresholds = {
        'EnvelopeSpectralOverlap': 0.05,
        'EnvelopeSmoothness': 0.001,
        'PhaseDiscontinuity': 0.05,
        'PhaseEnvCorr': 0.005,
        'InstFreqStd': 1.0  # Example threshold for frequency std
    }

    metrics = [
        ('EnvelopeSpectralOverlap', 'Envelope Spectral Overlap'),
        ('EnvelopeSmoothness', 'Envelope Smoothness (RMS error)'),
        ('PhaseDiscontinuity', 'Phase Discontinuity'),
        ('PhaseEnvCorr', 'Phase-Envelope Correlation'),
        ('InstFreqMean', 'Instantaneous Frequency Mean (Hz)'),
        ('InstFreqStd', 'Instantaneous Frequency Std (Hz)')
    ]

    order = ['Theta (5 Hz)', 'Alpha (~10 Hz)']

    for metric, ylabel in metrics:
        plt.figure(figsize=(7, 5))
        ax = sns.boxplot(x='Band', y=metric, data=df_subject, palette='Set2', order=order)
        sns.stripplot(
            x='Band', y=metric, data=df_subject,
            hue='Period',  # Color by period
            dodge=True,    # Separate dots by hue
            alpha=0.7, jitter=True, size=6, order=order, ax=ax,
            palette={'preStim': 'b', 'Stim': 'r'}
        )
        plt.title(ylabel)
        plt.xlabel('Frequency Band')
        plt.ylabel(ylabel)
        # Remove duplicate legend entries and set title
        handles, labels = ax.get_legend_handles_labels()
        # Only keep period handles (first two), drop threshold if present
        period_labels = ['preStim', 'Stim']
        period_handles = [h for h, l in zip(handles, labels) if l in period_labels]
        period_labels = [l for l in labels if l in period_labels]
        plt.legend(period_handles, period_labels, title='Period', loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, modality + f'_{metric}_byBand.png'), dpi=300)
        plt.savefig(os.path.join(savepath, modality + f'_{metric}_byBand.svg'), format='svg')
        plt.show()


    # Alpha peaks
    def extract_peak_component(cell, idx):
        """Extract idx-th number from a string/list/array, or return np.nan."""
        if isinstance(cell, float) or isinstance(cell, int):
            return cell if idx == 0 else np.nan
        if cell is None or pd.isna(cell):
            return np.nan
        if isinstance(cell, str):
            # Try to extract numbers from the string
            arr = np.fromstring(cell.replace('[','').replace(']',''), sep=' ')
            if arr.size > idx:
                return arr[idx]
            else:
                return np.nan
        if isinstance(cell, (list, tuple, np.ndarray)):
            return cell[idx] if len(cell) > idx else np.nan
        return np.nan

    # Apply to your DataFrame
    df['AlphaPeakPre_Freq'] = df['AlphaPeakPre'].apply(lambda x: extract_peak_component(x, 0))
    df['AlphaPeakPre_Power'] = df['AlphaPeakPre'].apply(lambda x: extract_peak_component(x, 1))
    df['AlphaPeakPre_BW'] = df['AlphaPeakPre'].apply(lambda x: extract_peak_component(x, 2))

    df['AlphaPeakPost_Freq'] = df['AlphaPeakPost'].apply(lambda x: extract_peak_component(x, 0))
    df['AlphaPeakPost_Power'] = df['AlphaPeakPost'].apply(lambda x: extract_peak_component(x, 1))
    df['AlphaPeakPost_BW'] = df['AlphaPeakPost'].apply(lambda x: extract_peak_component(x, 2))



    plt.figure(figsize=(9, 6))
    plt.errorbar(
        df['AlphaPeakPre_Freq'],
        df['AlphaPeakPre_Power'],
        xerr=df['AlphaPeakPre_BW'] / 2,
        fmt='o', color='b', ecolor='lightblue', elinewidth=2, capsize=4, label='Pre-stim'
    )
    plt.errorbar(
        df['AlphaPeakPost_Freq'],
        df['AlphaPeakPost_Power'],
        xerr=df['AlphaPeakPost_BW'] / 2,
        fmt='o', color='r', ecolor='salmon', elinewidth=2, capsize=4, label='Post-stim'
    )
    plt.xlabel('Alpha Peak Frequency (Hz)')
    plt.ylabel('Alpha Peak Power (FOOOF amplitude)')
    plt.title('Alpha Peak Frequency vs Power (Pre/Post-Stim) with Bandwidth')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, modality + 'Fooof_AlphaPeak_Freq_vs_Power.png'), dpi=300)
    plt.savefig(os.path.join(savepath, modality + 'Fooof_AlphaPeak_Freq_vs_Power.svg'), format='svg')
    plt.show()

    #PSD plots
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
from scipy.stats import wilcoxon

# Load the data
induced_pre = np.load(os.path.join(savepath, modality + 'IndicedPreWelch.npy'))
induced_post = np.load(os.path.join(savepath, modality + 'IndicedPostWelch.npy'))
evoked_pre = np.load(os.path.join(savepath, modality + 'EvokedPreWelch.npy'))
evoked_post = np.load(os.path.join(savepath, modality + 'EvokedPostWelch.npy'))

induced_pre = induced_pre[:, :, :, 1:41]  
induced_post = induced_post[:, :, :, 1:41]
evoked_pre = evoked_pre[:, :, :, 1:41]
evoked_post = evoked_post[:, :, :, 1:41]

chan = 66
colors = ['#9c1f27','#ba7b02', '#0d586b']
freqs = np.arange(1, 41)  # 1 to 40 Hz
theta_mask = (freqs >= 3) & (freqs <= 7)
alpha_mask = (freqs >= 8) & (freqs <= 12)
band_mask = theta_mask | alpha_mask
band_freqs = freqs[band_mask]

# Titles and data pairs
data_sets = [
    ('Pre-Stimulus Induced', induced_pre),
    ('Post-Stimulus Induced', induced_post),
    ('Pre-Stimulus Evoked', evoked_pre),
    ('Post-Stimulus Evoked', evoked_post)
]

fig = plt.figure(figsize=(12, 10))

for i, (title, data) in enumerate(data_sets):
    subplot = 221 + i
    ax = plt.subplot(subplot)

    # Friedman test 
    data_band = data[:, :, :, :][:, :, :, band_mask]  
    data_avg = np.mean(data_band, axis=2)  
    p_vals = []
    for f in range(data_avg.shape[2]):
        freq_data = [data_avg[:, cond_idx, f] for cond_idx in range(data_avg.shape[1])]
        stat, p = friedmanchisquare(*freq_data)
        p_vals.append(p)

    # fdfr
    rej, pvals_corrected = fdrcorrection(p_vals, alpha=0.05)
    significant_freqs = band_freqs[rej]
    print(f"\nCorrected p-values for {title}:")
    for freq, pval_corr, is_sig in zip(band_freqs, pvals_corrected, rej):
        sig_str = " (significant)" if is_sig else ""
        print(f"Freq {freq} Hz: p_corr = {pval_corr:.4g}{sig_str}")


    pairwise_results = {}  
    for f, freq in enumerate(band_freqs):
        freq_data = [data_avg[:, cond_idx, f] for cond_idx in range(data_avg.shape[1])]
        if rej[f]:
            pvals = []
            pairs = []
            for (i, j) in combinations(range(len(unique_conditions)), 2):
                stat, p = wilcoxon(freq_data[i], freq_data[j])
                pvals.append(p)
                pairs.append((unique_conditions[i], unique_conditions[j]))
            # FDR 
            rej_pair, pvals_corr = fdrcorrection(pvals, alpha=0.05)
            sig_pairs = [(pairs[k], pvals_corr[k]) for k in range(len(pairs)) if rej_pair[k]]
            if sig_pairs:
                pairwise_results[freq] = sig_pairs

    
    print("Significant pairwise differences (Wilcoxon, FDR-corrected) at each frequency:")
    for freq, sigs in pairwise_results.items():
        for (cond1, cond2), pval in sigs:
            print(f"Frequency {freq} Hz: {cond1} vs {cond2}, p={pval:.4g}")

    # Plot 
    for cond_idx, (cond_label, color) in enumerate(zip(unique_conditions, colors)):
        subj_avg = np.mean(data[:, cond_idx, :, :], axis=1)  
        avg_data = np.mean(subj_avg, axis=0)               
        avg_ci = np.std(subj_avg, axis=0, ddof=1) / np.sqrt(subj_avg.shape[0])
        freqs_plot = freqs[1:41]
        ax.plot(freqs_plot, avg_data[1:41], color=color, label=cond_label)
        ax.fill_between(freqs_plot, avg_data[1:41] - avg_ci[1:41], avg_data[1:41] + avg_ci[1:41],
                        color=color, alpha=0.3)

    for f in significant_freqs:
        ax.axvline(x=f, color='black', linestyle='--', alpha=0.4, zorder=0)

    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(savepath, modality + '_PSD_Welch_Comparison_avgOverSubs_perCondition_wStats.png'), dpi=300)
plt.savefig(os.path.join(savepath, modality + '_PSD_Welch_Comparison_avgOverSubs_perCondition_wStats.svg'), format='svg')
plt.show()




