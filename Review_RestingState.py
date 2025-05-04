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

# Add the parent directory of the script (LocalGlobalWave) to the Python path
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

allMotifsFile = 'RestingStateMotifsGrad_NoThreshold_EyesClosed'
figfolder = folder
fileList = glob.glob(os.path.join(folder, "*",  "Grad_18_OpticalFlowAfterFilter_Hilbert_masked_RestingState"), recursive=True)
oscillationThresholdFlag = False 

GA_motif_counts = []
allmotifs = []
allTrialInfo = []


for sub, filePath in enumerate(fileList):
    print("Processing file: " + filePath)
    dataBucketName = 'UV_Angle'
    subfolder  = os.path.basename(os.path.dirname(filePath))
    waveData = ImportHelpers.load_wavedata_object(filePath)

    #if gradiometer data, merge optical Flow
    if 'Grad' in filePath:
        combined_uv_map = waveData.get_data('UV_Angle_GradX') + waveData.get_data('UV_Angle_GradY')
        CombinedUVBucket = wd.DataBucket(combined_uv_map, "CombinedUV", 'freq_trl_posx_posy_time', waveData.get_channel_names())
        waveData.add_data_bucket(CombinedUVBucket)
        waveData.log_history(["CombinedUV", "VectorSum"])
        dataBucketName = 'CombinedUV'
        waveData.delete_data_bucket('UV_Angle_GradX')
        waveData.delete_data_bucket('UV_Angle_GradY')
    if 'Simulations' in filePath:
        dataBucketName = 'UV'


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
with open(folder +  allMotifsFile +  '.pickle', 'wb') as handle:
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

#____________________________________________________
#temp, just to compare plots with task data 
folder  = "/mnt/Data/DuguelabCluster/wavesim/LoGlo/"
allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
#___________________________________________________
GA_motif_counts = []
allTrialInfo = []
#% average top motifs per subject
with open(folder   + allMotifsFile + '.pickle', 'rb') as handle:
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
    

max_timepoint = waveData.get_data(dataBucketName).shape[-1]

nSubjects = len(fileList)
nTimepoints = max_timepoint
nFrequencies = 2  
#%% find Indices of Motif to keep and reduce GA_motifs to GA_sorted
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

time_vector = waveData.get_time()[:-2]

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
    
cmap = mcolors.ListedColormap(['grey', '#480384', '#f28c00', '#d67258', '#416ae4', '#378b8c', '#7bc35b'])
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Iterate over frequencies and their corresponding dataframes
for freqInd, (df, freqName) in enumerate(zip([dfTheta, dfAlpha], ['Theta', 'Alpha'])):
    # Reduce proportions plot to show only the 6 most frequent motifs (including -1)
    top_6_motifs = df['MotifInd'].value_counts(normalize=True).sort_values(ascending=False).head(7)
    top_6_indices = top_6_motifs.index

    # Create the figure and subplots
    fig, axs = plt.subplots(2, len(top_6_indices), figsize=(18, 12), gridspec_kw={'height_ratios': [1, 2]})

    # Bar plot (top row, spanning all columns)
    ax_bar = fig.add_subplot(2, 1, 1)
    colors = [cmap(norm(motifInd)) for motifInd in top_6_indices]
    top_6_motifs.plot(kind='bar', color=colors, edgecolor='black', ax=ax_bar)
    ax_bar.set_xlabel('MotifInd')
    ax_bar.set_ylabel('Proportion of Timepoints')
    ax_bar.set_title(f'Proportion of Timepoints for Top 6 Motifs ({freqName})')
    ax_bar.set_xticks(range(len(top_6_indices)))
    ax_bar.set_xticklabels(top_6_indices, rotation=45)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    # Quiver plots (second row)
    for i, motifInd in enumerate(top_6_indices):
        ax = axs[1, i]
        if motifInd == -1:
            # Placeholder for "no motif"
            ax.text(0.5, 0.5, 'No Motif', fontsize=12, ha='center', va='center', color='red')
            ax.set_facecolor('white')
            ax.set_title(f'Motif {motifInd}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Plot the actual quiver plot for valid motifs
            motif_array = GA_motifs[freqInd][motifInd]['average']
            ax.quiver(-np.real(motif_array), -np.imag(motif_array), color='black')
            ax.set_facecolor('white')
            ax.set_aspect('equal')
            ax.set_title(f'Motif {motifInd}')

    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = f"{figfolder}{allMotifsFile}_{freqName}_MotifBarAndQuiverPlots.svg"
    plt.savefig(output_path, format='svg', dpi=1200)
    plt.show()

# # dfTheta.to_csv(f"{figfolder}ThetaMotifCountsFull.csv", index=False)
# # dfAlpha.to_csv(f"{figfolder}AlphaMotifCountsFull.csv", index=False)
# # Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
# GA_sorted = [[],[]]
# for ind, df in enumerate([dfTheta, dfAlpha]):
#     motif_count = df.groupby(['Timepoint', 'MotifInd', 'Subject']).size().reset_index(name='Count')

#     # make all possible combinations
#     timepoints = df['Timepoint'].unique()
#     motif_inds = df['MotifInd'].unique()
#     subjects = df['Subject'].unique()

#     complete_index = pd.MultiIndex.from_product([ timepoints, motif_inds, subjects], names=[ 'Timepoint', 'MotifInd', 'Subject'])
#     # Reindex to include all combinations, fill missing values with 0
#     motif_count = motif_count.set_index([ 'Timepoint', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()

#     average_count = motif_count.groupby(['MotifInd'])['Count'].mean().reset_index()
#     average_count = average_count.sort_values(by='Count', ascending=False)
#         # Get the top 6 'MotifInd'

#     amount_of_motifs = 7
#     top_motif_inds = average_count.head(amount_of_motifs)['MotifInd'].reset_index(drop=True)

#     for motifInd in top_motif_inds[1:]:            
#         GA_sorted[ind].append(GA_motifs[ind][motifInd])

# #in GA_motifs ignore "subject" field. The correct subject is in the "trial_frames" field
# with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'wb') as handle:
#     pickle.dump(GA_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)

# max_timepoint = waveData.get_data(dataBucketName).shape[-1]

# #%%___________________ Make cleaned up dataFrames from GA_sorted
# nSubjects = len(fileList)

# theta_array = np.zeros((nSubjects, nTrials, nTimepoints), dtype=int)-1
# alpha_array = np.zeros((nSubjects, nTrials, nTimepoints), dtype=int)-1
# freqs = ['theta', 'alpha']
# freq_arrays = [theta_array, alpha_array]
# for freq in range(len(GA_sorted)):
#     tempGA = GA_sorted[freq]
#     for sub in range(nSubjects):  
#         tempList = [[-1 for _ in range(nTimepoints)] for _ in range(nTrials)]
#         for motifInd, motif in enumerate(tempGA):
#             trial_frames_list = motif['trial_frames']
#             for trial_frame in trial_frames_list:
#                 subject_number, (trial, (start_timepoint, end_timepoint)) = trial_frame
#                 if subject_number == sub:
#                     if 0 <= trial < nTrials and 0 <= start_timepoint < nTimepoints and 0 <= end_timepoint <= nTimepoints:
#                         for timepoint in range(start_timepoint, end_timepoint):
#                             tempList[trial][timepoint] = motifInd 
#         # Convert tempList to a NumPy array and store it in the corresponding frequency array
#         freq_arrays[freq][sub, :nTrials, :] = np.array(tempList)

# time_vector = waveData.get_time()[:-2]

# data = []
# # Iterate through subjects and trials
# for sub in range(nSubjects):
#     for trial in range(nTrials):
#         for timepoint in range(nTimepoints):
#             for freqInd, freq_array in enumerate(freq_arrays):
#                 motifInd = freq_array[sub, trial, timepoint]
#                 data.append([sub, trial, timepoint, freqInd, motifInd])

# df = pd.DataFrame(data, columns=['Subject', 'Trial', 'Timepoint', 'Frequency', 'MotifInd'])
# df.to_csv(f"{figfolder}MotifCountsFull.csv", index=False)

# # Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
# motif_counts = df.groupby([ 'Timepoint', 'Frequency', 'MotifInd', 'Subject']).size().reset_index(name='Count')

# # make all possible combinations
# timepoints = df['Timepoint'].unique()
# frequencies = df['Frequency'].unique()
# motif_inds = df['MotifInd'].unique()
# subjects = df['Subject'].unique()

# complete_index = pd.MultiIndex.from_product([ timepoints, frequencies, motif_inds, subjects], names=[ 'Timepoint', 'Frequency', 'MotifInd', 'Subject'])
# # Reindex to include all combinations, fill missing values with 0
# motif_counts = motif_counts.set_index([ 'Timepoint', 'Frequency', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()
# #%% All subjects, all trials, all times
# #  Define colormap and normalization
# cmap = mcolors.ListedColormap(['grey', '#480384', '#f28c00', '#d67258', '#416ae4', '#378b8c', '#7bc35b'])
# bounds = [-1, 0, 1, 2, 3, 4, 5]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# # Plot
# for freqInd, freq in enumerate(freqs):
#     fig, axs = plt.subplots(nrows=nSubjects, figsize=(10, 3 * nSubjects))  # One row per subject
#     for subInd in range(nSubjects):
#         ax = axs[subInd] if nSubjects > 1 else axs
#         data = freq_arrays[freqInd][subInd, :, :]
#         im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm, extent=[time_vector[0], time_vector[-1], 0, data.shape[0]])
#         ax.set_xlabel('Time')
#         ax.set_ylabel(f'Subject {subInd}')
#         if subInd == 0:
#             ax.set_title(f'Frequency: {freq}')
#         fig.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.savefig(f"{figfolder}MotifCountsAllSubsAllTrialsAllTimes_{freq}.svg", format='svg', dpi=1200)
#     plt.show()

