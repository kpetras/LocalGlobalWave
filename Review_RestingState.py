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
def plot_data(freqs, avg_data, single_chan_data, color, label_prefix, title, chan, show_legend=False):
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
folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/"
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
# Create a list of WaveData files in the folder
fileName = "EEGWaveData_RestingState"
fileList = glob.glob(os.path.join(folder, "*", fileName), recursive=True)
thetaList =[]
alphaList =[]
for sub,filePath in enumerate(fileList):
    subname = os.path.basename(os.path.dirname(filePath))    
    savefolder = os.path.dirname(filePath)
    print("Processing subject: " + str(sub))
    print("Processing file: " + filePath)
    # the wavedata objects
    waveData = ImportHelpers.load_wavedata_object(filePath)

      
    #set up a bunch of stuff in the first iteration
    time_vect = waveData.get_time()
    windowLength = 3 #seconds
    windowsize = int(np.floor((windowLength*waveData.get_sample_rate()))) #in samples
    fftfreqs = np.fft.fftfreq(windowsize, d=time_vect[1]-time_vect[0])#we use 1 second segments 
    fft_freqEdges = hf.bin_edges_from_centers(fftfreqs)

    #we want the first bin to be centered on 1 and the last on 40 Hz
    freqMin = 1 - ((fftfreqs[1]-fftfreqs[0])/2)
    freqMax = 40 + ((fftfreqs[1]-fftfreqs[0])/2)
    freq_min_idx = hf.find_nearest(fft_freqEdges, freqMin)[0]
    freq_max_idx = hf.find_nearest(fft_freqEdges, freqMax)[0]
    nbins = freq_max_idx - freq_min_idx
    nChans = waveData.get_data('EEG').shape[1]    
    evoked_dict = {}

    #%FFT spectrum 
    chan= 94
    chan = 66
    fft= np.zeros((nChans,windowsize),dtype=complex)
    x = waveData.get_data('EEG')
    fft[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
    fft_chan_AVG = np.mean(fft, axis=0)
    fft_singleChan = fft[chan,:]    

    fig = plt.figure(figsize=(6, 6))
    plot_data(fftfreqs[freq_min_idx:freq_max_idx], fft_chan_AVG[freq_min_idx:freq_max_idx], fft_singleChan[freq_min_idx:freq_max_idx], 'darkblue', '', ' Resting State, 3 sec average',chan = chan, show_legend = False)
    plt.tight_layout()
    plt.show()  
     
    #find individual alpha from the pre-stim peak
    time_vect = waveData.get_time()
    windowLength = 3 #seconds
    f_sample = waveData.get_sample_rate()
    #initialize arrays to store the spectra
    from scipy.signal import welch
    data = waveData.get_data('EEG')

    nTrials, nChans, _ = data.shape
    _, Pxx_den = welch(data[0, 0, :], fs=f_sample)
    nbins = len(Pxx_den)

    spec = np.zeros((nTrials, nChans, nbins))

    # Compute the power spectra for each trial and channel
    for trial in range(nTrials):
        for chan in range(nChans):
            freqs, Pxx_den = welch(data[trial, chan, :], fs=f_sample)
            spec[trial, chan, :] = Pxx_den


    # Average the spectra over trials and channels
    avgSpec = np.mean(spec, axis=(0, 1))
    #check the 5 Hz
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [2, 40]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    #fm.report(freqs, avgSpecPostStim, freq_range)
    fm.fit(freqs, avgSpec, [3, 30])

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
    fm.fit(freqs, avgSpec, [3, 30])

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
            freq = 5 #theta[0] we know the freq of interest here
        else:
            freq = alpha[0]
         #% do filter + Hilbert to get complex Timeseries 
        filt.filter_narrowband(waveData, dataBucketName = "EEG", LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
        waveData.DataBuckets[str(freqInd)] =  waveData.DataBuckets.pop("NBFiltered")
    temp = np.stack((waveData.DataBuckets["0"].get_data(), waveData.DataBuckets["1"].get_data()),axis=0)
    waveData.add_data_bucket(wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names()))


    waveData.set_active_dataBucket('NBFiltered')
    if 'Mag' in fileName:
        chanInds = [(28,80),(0,51)]
    elif 'Grad' in fileName:
        origChanpos=waveData.get_channel_positions()        
        hf.combine_grad_sensors(waveData)
        chanInds = [(0,51),(28,80)]        
    elif fileName.startswith('EEG'):
        chanInds = [(1,72),(38,28)]
    else:
        chanInds= True
    Surface, PolySurface = SensorLayout.create_surface_from_points(waveData,
                                                            type='channels',
                                                            num_points=1000,
                                                            plotting=True)
    
    SensorLayout.distance_along_surface(waveData, Surface, tolerance=0.1, get_extent = chanInds, plotting= True)
    SensorLayout.distmat_to_2d_coordinates_Isomap(waveData) #can also use MDS here
    
    #plot topo
    timeInds = 300
    dataInds = (0, slice(None), slice(None))
    plotting.plot_topomap(waveData,dataBucketName = "NBFiltered", dataInds = dataInds, timeInds= timeInds, trlInd = 0)

    grid_x, grid_y, mask =SensorLayout.interpolate_pos_to_grid(
        waveData, 
        dataBucketName = "NBFiltered",
        numGridBins=18,
        return_mask = True,
        mask_stretching = True)


    # make new distMat based on the interpolated grid
    positions = np.dstack((grid_x, grid_y)).reshape(-1, 2)
    distMat = SensorLayout.regularGrid(waveData, positions)
    original_data_bucket = "NBFiltered"
    interpolated_data_bucket = "NBFilteredInterpolated"

    OrigInd  = (0,0,slice(None),300)
    InterpInd =(0,0,slice(None),slice(None),300)
    fig = plotting.plot_interpolated_data(waveData, original_data_bucket, interpolated_data_bucket,
                                            grid_x, grid_y, OrigInd, InterpInd,  type='')
    # # get complex timeseries
    hilb.apply_hilbert(waveData, dataBucketName = "NBFilteredInterpolated")

    SensorLayout.apply_mask(waveData, mask, dataBucketName = 'AnalyticSignal', overwrite = True, storeMask = True)

    tStart = time.time()
    print("OpticalFlow started")
    OpticalFlow.create_uv(waveData, 
            applyGaussianBlur=False, 
            type = "angle", 
            Sigma=1, 
            alpha = 0.1, 
            nIter = 200, 
            dataBucketName='AnalyticSignal',
            is_phase = False)
    trialToPlot = 5
    print('optical flow took: ', time.time()-tStart)
    
    trialToPlot = 5
    waveData.DataBuckets["UV_Angle"] =  waveData.DataBuckets.pop("UV")
    waveData.set_active_dataBucket('UV_Angle')
    # ani = plotting.plot_optical_flow(waveData, 
    #                                 UVBucketName = 'UV_Angle',
    #                                 PlottingDataBucketName = 'AnalyticSignal', 
    #                                 dataInds = (1,trialToPlot, slice(None), slice(None), slice(None)),
    #                                 plotangle=True,
    #                                 normVectorLength = True)  
    # ani.save('OpticalFlowAfterFilter_Hilbert1.gif')

    waveData.save_to_file(savefolder + '/EEG_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState')
waveData.delete_data_bucket("NBFilteredInterpolated")
waveData.delete_data_bucket("0")
waveData.delete_data_bucket("1")
waveData.delete_data_bucket("NBFiltered")
#save the individual theta and alpha values
import pandas as pd
theta_values = [item[0] for item in thetaList]
alpha_values = [item[0] for item in alphaList]

# Create a DataFrame
df = pd.DataFrame({
    'Theta': theta_values,
    'Alpha': alpha_values
})

# Save to CSV
df.to_csv(savefolder + '/theta_alpha_values.csv', index=False)



   
#%%______________________________________________________________________
#% Part 2: Find Motifs
#%________________________________________________________________________



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


#%%________Set files___________________________________________
# folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/"
# allMotifsFile = 'RestingStateMotifsEEG_NoThreshold_EyesClosed'
# figfolder = folder 
# fileList = glob.glob(os.path.join(folder, "*",  "EEG_18_OpticalFlowAfterFilter_Hilbert_masked_RestingStateEyesClosed"), recursive=True)
# oscillationThresholdFlag = False 

# allMotifsFile = 'RestingStateMotifsMag_NoThreshold_EyesClosed'
# figfolder = folder
# fileList = glob.glob(os.path.join(folder, "*",  "Mag_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState"), recursive=True)
# oscillationThresholdFlag = False 

# allMotifsFile = 'RestingStateMotifsGrad_NoThreshold_EyesClosed'
# figfolder = folder
# fileList = glob.glob(os.path.join(folder, "*",  "Grad_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState"), recursive=True)
# oscillationThresholdFlag = False 

folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/"
figfolder = folder
modalities = ["EEG", "Mag", "Grad"]
conditions = ["EyesClosed", "EyesOpen"]
oscillationThresholdFlag = False
for modality in modalities:
    file_pattern_closed = os.path.join(folder, "*", f"{modality}_18_OpticalFlowAfterFilter_Hilbert_masked_RestingStateEyesClosed")
    file_pattern_open = os.path.join(folder, "*", f"{modality}_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState")
    files_closed = sorted(glob.glob(file_pattern_closed, recursive=True))
    files_open = sorted(glob.glob(file_pattern_open, recursive=True))


    GA_motif_counts = []
    allmotifs = []
    allTrialInfo = []

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
        #if gradiometer data, merge optical Flow
        if modality == 'Grad':
            combined_uv_map = waveData.get_data('UV_Angle_GradX') + waveData.get_data('UV_Angle_GradY')
            CombinedUVBucket = wd.DataBucket(combined_uv_map, "CombinedUV", 'freq_trl_posx_posy_time', waveData.get_channel_names())
            waveData.add_data_bucket(CombinedUVBucket)
            waveData.log_history(["CombinedUV", "VectorSum"])
            dataBucketName = 'CombinedUV'
            waveData.delete_data_bucket('UV_Angle_GradX')
            waveData.delete_data_bucket('UV_Angle_GradY')


        freqs= [5, 10]
        # Find Motifs   
        sample_rate = waveData.get_sample_rate()  # your sample rate
        if oscillationThresholdFlag:
            if 'Grad' in filePath:
                powerBucketName = "PLV_and_Power" 
            else:
                powerBucketName = "PLV_andAnalyticSignal" 
                #this one has three sets of data: PLV theta. analytic signal theta, analytic signal alpha. make temp one with analytic signal
                tempData = waveData.get_data(powerBucketName)[[1, 2], :, :, :, :]
                tempDataBucket = wd.DataBucket(tempData, 'temp', 
                                            waveData.DataBuckets[dataBucketName].get_dimord(), 
                                            waveData.DataBuckets[dataBucketName].get_channel_names() )
                waveData.add_data_bucket(tempDataBucket)
                powerBucketName = 'temp'
        else:
            powerBucketName = None

        for freqInd in range(waveData.get_data(dataBucketName).shape[0]):
            threshold = .85
            pixelThreshold = .4
            mergeThreshold = .7
            minFrames = int(np.floor((waveData.get_sample_rate() / freqs[1])))
            nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd]))
            
            motifs = hf.find_wave_motifs(waveData, 
                                                dataBucketName=dataBucketName, 
                                                oscillationThresholdDataBucket = powerBucketName,
                                                oscillationThresholdFlag = oscillationThresholdFlag,
                                                baselinePeriod=None,
                                                threshold = threshold, 
                                                nTimepointsEdge=nTimepointsEdge,
                                                mergeThreshold = mergeThreshold, 
                                                minFrames=minFrames, 
                                                pixelThreshold = pixelThreshold, 
                                                magnitudeThreshold=.1,
                                                dataInds = (freqInd, slice(None), slice(None), slice(None), slice(None)),
                                                Mask = True)


            # Add 'subject' field to each motif
            for motif in motifs:
                motif['subject'] = sub
                motif['frequency'] = freqInd
            allmotifs.extend(motifs)

    #save all motifs
    allMotifsFile = f"RestingStateMotifs{modality}_NoThreshold_EyesOpenAndClosed"
    with open(folder + allMotifsFile + '.pickle', 'wb') as handle:
        pickle.dump(allmotifs, handle, protocol=pickle.HIGHEST_PROTOCOL)



#%%
#Only use if cell above is not run
filePath = fileList[0]
waveData = ImportHelpers.load_wavedata_object(filePath)
#load random waveData object to get shapes and timepoints
#if gradiometer data, merge optical Flow
BucketNames = waveData.DataBuckets.keys()
for key in waveData.DataBuckets.keys():
    if "UV" in key:
        first_key_with_uv = key
        break
dataBucketName = first_key_with_uv

# #____________________________________________________
# #temp, just to compare plots with task data 
# folder  = "/mnt/Data/DuguelabCluster/wavesim/LoGlo/"
# allMotifsFile = 'RestingStateMotifsGrad_NoThreshold'
# #___________________________________________________

#%% all resting state motifs
folder = "/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/"

# Find all matching files
all_data = []
# Loop over all matching files
for modality in modalities:
    allMotifsFile = f"RestingStateMotifs{modality}_NoThreshold_EyesOpenAndClosed"

    # Extract the base filename (without extension) for use in titles and output paths
    fullFileName = allMotifsFile + '.pickle'
    print(f"Processing file: {allMotifsFile}")

    GA_motif_counts = []
    allTrialInfo = []
    #% average top motifs per subject
    with open(folder   + fullFileName, 'rb') as handle:
        allmotifs = pickle.load(handle)


    GA_motifs = {}
    for freqInd in range(2):
        filtered_motifs = [motif for motif in allmotifs if motif['frequency'] == freqInd]
        merge_threshold = .7
        pixelThreshold = .4
        print("Merges if vector angles are below " + str(np.degrees(np.arccos(merge_threshold))) + " degrees")
        GA_motifs[freqInd] = hf.merge_motifs_across_subjects(filtered_motifs, 
                                                            mergeThreshold = merge_threshold, 
                                                            pixelThreshold = pixelThreshold)
        
    #save GA_motifs
    with open(figfolder + 'GA_motifs' + allMotifsFile+ '.pickle', 'wb') as handle:
        pickle.dump(GA_motifs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        max_timepoint = waveData.get_data(dataBucketName).shape[-1]

    nSubjects = 19
    nTimepoints = max_timepoint
    nFrequencies = 2 

    #% find Indices of Motif to keep and reduce GA_motifs to GA_sorted
    nTrials = waveData.get_data(dataBucketName).shape[1]
    theta_array = np.zeros((nSubjects, nTrials, nTimepoints), dtype=int)-1
    alpha_array = np.zeros((nSubjects, nTrials, nTimepoints), dtype=int)-1
    freqs = ['theta', 'alpha']
    freq_arrays = [theta_array, alpha_array]
    for freq in range(len(GA_motifs)):
        tempGA = GA_motifs[freq]
        for sub in range(nSubjects):          
            tempList = [[-1 for _ in range(nTimepoints)] for _ in range(nTrials)]
            for motifInd, motif in enumerate(tempGA):
                trial_frames_list = motif['trial_frames']
                for trial_frame in trial_frames_list:
                    subject_number, (trial, (start_timepoint, end_timepoint)) = trial_frame                    
                    if subject_number == sub:
                        if 0 <= trial < nTrials and 0 <= start_timepoint < nTimepoints and 0 <= end_timepoint <= nTimepoints:
                            for timepoint in range(start_timepoint, end_timepoint):
                                tempList[trial][timepoint] = motifInd 
            # Convert tempList to a NumPy array and store it in the corresponding frequency array
            freq_arrays[freq][sub, :, :] = np.array(tempList)


    data = [[],[]]

    # Iterate through subjects and trials
    for sub in range(nSubjects):
        for trial in range(nTrials):
            for timepoint in range(nTimepoints):
                for freqInd, freq_array in enumerate(freq_arrays):
                    motifInd = freq_array[sub, trial, timepoint]
                    data[freqInd].append([sub, trial, timepoint, motifInd])

    dfTheta = pd.DataFrame(data[0], columns=['Subject', 'Trial',  'Timepoint', 'MotifInd'])
    dfAlpha = pd.DataFrame(data[1], columns=['Subject', 'Trial',  'Timepoint', 'MotifInd'])
    #Make dataframe with all modalities and conditions    
    for df, freqName in zip([dfTheta, dfAlpha], ['Theta', 'Alpha']):
        df['Modality'] = modality
        df['Condition'] = df['Trial'].apply(lambda x: "EyesOpen" if x < 60 else "EyesClosed")
        df['Frequency'] = freqName
        all_data.append(df)
        print(df.shape[0])
    all_df = pd.concat(all_data, ignore_index=True)
#save
all_df.to_csv(f"{figfolder}AllModalities_FullDataFrameAllCondsAllSubsGAMotifs.csv", index=False)


#%%Plot
import matplotlib.colors as mcolors

# Define motif colors (pairs: [EyesOpen, EyesClosed])
motif_colors = [
    ("#888888", "#cccccc"),  # gray (darker, lighter)
    ("#f28c00", "#ffd699"),  # orange (darker, lighter)
    ("#416ae4", "#a6baff"),  # blue (darker, lighter)
    ("#378b8c", "#8fd6d7"),  # teal (darker, lighter)
    ("#7bc35b", "#c6eab2"),  # green (darker, lighter)
    ("#d67258", "#f2b8a0"),  # brownish (darker, lighter)
    ("#480384", "#b299c6"),  # purple (darker, lighter)
]
#load all_df
subject_level_rows = []

all_df = pd.read_csv(f"{figfolder}AllModalities_FullDataFrameAllCondsAllSubsGAMotifs.csv")
for modality in modalities:
    allMotifsFile = f"RestingStateMotifs{modality}_NoThreshold_EyesOpenAndClosed"
    with open(figfolder + 'GA_motifs' + allMotifsFile+ '.pickle','rb') as handle:        
        GA_motifs = pickle.load(handle)

    all_df_modality = all_df[all_df['Modality'] == modality]

    for freqInd, freqName in enumerate(['Theta', 'Alpha']):
        df = all_df_modality[all_df_modality['Frequency'] == freqName]

        # Find the 6 most common motifInds (excluding -1)
        motif_counts = df[df['MotifInd'] != -1]['MotifInd'].value_counts().sort_values(ascending=False)
        top6_motif_inds = motif_counts.head(6).index.tolist()

        # Always include -1 (no motif) at the start
        motif_order = [-1] + top6_motif_inds

        # Filter DataFrame to only these motifInds, and set categorical order for plotting
        filtered_df = df[df['MotifInd'].isin(motif_order)].copy()
        filtered_df['MotifInd'] = pd.Categorical(filtered_df['MotifInd'], categories=motif_order, ordered=True)
        filtered_df = filtered_df.sort_values('MotifInd')

        # Calculate proportions per subject for error bars
        nTrials = 60
        nTimepoints = nTimepoints
        total_timepoints = nTrials * nTimepoints

        # Group by subject, condition, motif
        subj_prop = (
            filtered_df.groupby(['Subject', 'Condition', 'MotifInd'])
            .size()
            .reset_index(name='Count')
        )
        subj_prop['Proportion'] = subj_prop['Count'] / total_timepoints
        for _, row in subj_prop.iterrows():
            subject_level_rows.append({
                'Modality': modality,
                'Frequency': freqName,
                'MotifInd': row['MotifInd'],
                'Condition': row['Condition'],
                'Subject': row['Subject'],
                'Proportion': row['Proportion']
            })


        # Pivot to get mean and sem for each motif/condition
        means = subj_prop.groupby(['MotifInd', 'Condition'])['Proportion'].mean().unstack()
        stds = subj_prop.groupby(['MotifInd', 'Condition'])['Proportion'].std().unstack()

        motif_inds = means.index.tolist()
        x = np.arange(len(motif_inds))
        width = 0.35

        fig = plt.figure(figsize=(3 * len(motif_order), 10))
        gs = fig.add_gridspec(3, len(motif_order), height_ratios=[1,2,2])
        ax_bar = fig.add_subplot(gs[0, :])

        # Bar plot with error bars
        bars1 = ax_bar.bar(
            x - width/2, means['EyesOpen'], width, 
            yerr=stds['EyesOpen'], label='EyesOpen', 
            color=[motif_colors[i][0] for i in range(len(motif_order))],
            capsize=5
        )
        bars2 = ax_bar.bar(
            x + width/2, means['EyesClosed'], width, 
            yerr=stds['EyesClosed'], label='EyesClosed', 
            color=[motif_colors[i][1] for i in range(len(motif_order))],
            capsize=5
        )

        ax_bar.set_title(
            f'Proportion of Timepoints for Top 6 Motifs ({freqName}, {modality})\n'
        )
        ax_bar.set_xlabel('MotifInd')
        ax_bar.set_ylabel('Proportion of Timepoints')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(motif_inds)
        ax_bar.legend(title='Condition')
        ax_bar.set_ylim(0, 1)
        # Quiver plots (second row)
        for i, motifInd in enumerate(motif_order):
            ax = fig.add_subplot(gs[1, i])  # Add to second row, appropriate column
            color = motif_colors[i][0]  # Use EyesOpen (darker) color for quiver
            if motifInd == -1:
                # Placeholder for "no motif"
                ax.text(0.5, 0.5, 'No Motif', fontsize=12, ha='center', va='center', color=color)
                ax.set_facecolor('white')
                ax.set_title(f'Motif {motifInd}')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                motif_array = GA_motifs[freqInd][motifInd]['average']
                ax.quiver(-np.real(motif_array), -np.imag(motif_array), color=color)
                ax.set_facecolor('white')
                ax.set_aspect('equal')
                ax.set_title(f'Motif {motifInd}')

        # Polar plots (third row)
        for i, motifInd in enumerate(motif_order):
            ax = fig.add_subplot(gs[2, i], polar=True)
            color = motif_colors[i][0]
            if motifInd == -1:
                ax.text(0.5, 0.5, 'No Motif', fontsize=12, ha='center', va='center', color=color, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Motif {motifInd}')
            else:
                motif_array = GA_motifs[freqInd][motifInd]['average']
                mask = waveData.get_data("Mask")
                direction = np.arctan2(-np.imag(motif_array[mask]), -np.real(motif_array[mask]))
                ax.hist(direction.flatten(), bins=40, color=color)
                ax.set_title(f'Motif {motifInd}')


        plt.tight_layout()
        output_path = f"{figfolder}{allMotifsFile}_{freqName}_MotifBarAndQuiverPlots.svg"
        plt.savefig(output_path, format='svg', dpi=1200)
        plt.show()

subject_level_df = pd.DataFrame(subject_level_rows)
print(subject_level_df.head())
subject_level_df.to_csv(f"{figfolder}SubjectLevelProportionsOfTimepointsShowingMotif.csv", index=False)

#Stats
results = []
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

for modality in subject_level_df['Modality'].unique():
    for freq in subject_level_df['Frequency'].unique():
        df_sub = subject_level_df[(subject_level_df['Modality'] == modality) & (subject_level_df['Frequency'] == freq)]
        for motif in df_sub['MotifInd'].unique():
            motif_df = df_sub[df_sub['MotifInd'] == motif]
            # Pivot to get paired data
            pivot = motif_df.pivot(index='Subject', columns='Condition', values='Proportion')
            # Only keep subjects with both conditions
            pivot = pivot.dropna(subset=['EyesOpen', 'EyesClosed'])
            if len(pivot) < 2:
                continue  # Not enough data for stats
            # Paired t-test
            t_stat, t_p = ttest_rel(pivot['EyesOpen'], pivot['EyesClosed'])
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = wilcoxon(pivot['EyesOpen'], pivot['EyesClosed'])
            except ValueError:
                w_stat, w_p = (None, None)
            results.append({
                'Modality': modality,
                'Frequency': freq,
                'MotifInd': motif,
                'n_subjects': len(pivot),
                'EyesOpen_mean': pivot['EyesOpen'].mean(),
                'EyesClosed_mean': pivot['EyesClosed'].mean(),
                't_stat': t_stat,
                't_p': t_p,
                'w_stat': w_stat,
                'w_p': w_p
            })

stats_df = pd.DataFrame(results)
print(stats_df)




# %%
