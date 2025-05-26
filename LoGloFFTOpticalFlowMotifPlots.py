#%%
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

#%%
folder = "<folder_path>"
#%%________Set files___________________________________________

# allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 

# allMotifsFile = 'AllCondsMotifsMEG_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 

# allMotifsFile = 'AllCondsMotifsGrad_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 

allMotifsFile = 'AllCondsMotifsSimulations_NoThreshold'
figfolder = '<figfolder_path>' 
fileList = glob.glob(os.path.join(folder, 'Simulations', 'sub*_Filter_Hilbert_OpticalFlow'))
oscillationThresholdFlag = False

GA_motif_counts = []
allmotifs = []
allTrialInfo = []
#%%______________________________________________________________________

for sub, filePath in enumerate(fileList):
    print("Processing file: " + filePath)
    dataBucketName = 'UV_Angle'
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

    allTrialInfo.append(waveData.get_trialInfo())
    condInfo  = waveData.get_trialInfo()
    conds = np.unique(condInfo)
    #exclude conditios containing 'fov' 
    conds = [cond for cond in conds if 'fov' not in cond]
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
        if 'Simulations' in filePath:
            threshold = .8
            pixelThreshold = .6
            mergeThreshold = .8
        else:
            threshold = .85
            pixelThreshold = .4
            mergeThreshold = .7

        minFrames = int(np.floor((waveData.get_sample_rate() / freqs[1])))
        nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd]))
        baselinePeriod = (100,240) if freqInd == 0 else None 
        motifs = hf.find_wave_motifs(waveData, 
                                            dataBucketName=dataBucketName, 
                                            oscillationThresholdDataBucket = powerBucketName,
                                            oscillationThresholdFlag = oscillationThresholdFlag,
                                            baselinePeriod=baselinePeriod,
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
with open(folder + allMotifsFile +  '.pickle', 'wb') as handle:
                pickle.dump(allmotifs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'wb') as handle:
                pickle.dump(allTrialInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

GA_motif_counts = []
allTrialInfo = []
#% average top motifs per subject
with open(folder + allMotifsFile +  '.pickle', 'rb') as handle:
    allmotifs = pickle.load(handle)
with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'rb') as handle:
    allTrialInfo = pickle.load(handle)

GA_motifs = {}
for freqInd in range(2):
    filtered_motifs = [motif for motif in allmotifs if motif['frequency'] == freqInd]
    merge_threshold = .8
    pixelThreshold = .4
    print("Merges if vector angles are below " + str(np.degrees(np.arccos(merge_threshold))) + " degrees")
    GA_motifs[freqInd] = hf.merge_motifs_across_subjects(filtered_motifs, 
                                                         mergeThreshold = merge_threshold, 
                                                         pixelThreshold = pixelThreshold)
    

max_timepoint = waveData.get_data(dataBucketName).shape[-1]

#%%___________________ make padded TrialInfo
nSubjects = len(fileList)
nMaxTrials = max(len(sublist) for sublist in allTrialInfo)  # Maximum number of trials across subjects (sub 18 only has 460)
#make padded Cond-Info
allTrialInfoPadded = []
for sub in range(nSubjects):
    allTrialInfoPadded.append(allTrialInfo[sub] + ['none'] * (nMaxTrials - len(allTrialInfo[sub])))
nTimepoints = max_timepoint
nFrequencies = 2  
allTrialInfo =allTrialInfoPadded
#%% find Indices of Motif to keep and reduce GA_motifs to GA_sorted
theta_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
alpha_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
freqs = ['theta', 'alpha']
freq_arrays = [theta_array, alpha_array]
for freq in range(len(GA_motifs)):
    tempGA = GA_motifs[freq]
    for sub in range(nSubjects):  
        nTrials = 460 if sub == 18 else nMaxTrials # Subject 18 only has 460 trials
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
        freq_arrays[freq][sub, :nTrials, :] = np.array(tempList)


conds = ['full stand', 'full trav in', 'full trav out', 'none']  

# Split data by conditions
cond_indices = {cond: [] for cond in conds[:-1]}
for sub in range(nSubjects):
    for trial in range(nMaxTrials):
        condition = allTrialInfoPadded[sub][trial]
        if condition in cond_indices:
            cond_indices[condition].append((sub, trial))

time_vector = waveData.get_time()[:-2]

data = [[],[]]

# Iterate through subjects and trials
for sub in range(nSubjects):
    for trial in range(nMaxTrials):
        condition = allTrialInfoPadded[sub][trial]
        for timepoint in range(nTimepoints):
            for freqInd, freq_array in enumerate(freq_arrays):
                motifInd = freq_array[sub, trial, timepoint]
                data[freqInd].append([sub, trial, condition, timepoint, motifInd])

dfTheta = pd.DataFrame(data[0], columns=['Subject', 'Trial', 'Condition', 'Timepoint', 'MotifInd'])
dfAlpha = pd.DataFrame(data[1], columns=['Subject', 'Trial', 'Condition', 'Timepoint', 'MotifInd'])
    
# dfTheta.to_csv(f"{figfolder}ThetaMotifCountsFull.csv", index=False)
# dfAlpha.to_csv(f"{figfolder}AlphaMotifCountsFull.csv", index=False)
# Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
GA_sorted = [[],[]]
for ind, df in enumerate([dfTheta, dfAlpha]):
    motif_count = df.groupby(['Condition', 'Timepoint', 'MotifInd', 'Subject']).size().reset_index(name='Count')

    # make all possible combinations
    conditions = df['Condition'].unique()
    timepoints = df['Timepoint'].unique()
    motif_inds = df['MotifInd'].unique()
    subjects = df['Subject'].unique()

    complete_index = pd.MultiIndex.from_product([conditions, timepoints, motif_inds, subjects], names=['Condition', 'Timepoint', 'MotifInd', 'Subject'])
    # Reindex to include all combinations, fill missing values with 0
    motif_count = motif_count.set_index(['Condition', 'Timepoint', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()

    average_count = motif_count.groupby(['MotifInd'])['Count'].mean().reset_index()
    average_count = average_count.sort_values(by='Count', ascending=False)
        # Get the top 6 'MotifInd'

    amount_of_motifs = 7
    top_motif_inds = average_count.head(amount_of_motifs)['MotifInd'].reset_index(drop=True)

    for motifInd in top_motif_inds[1:]:            
        GA_sorted[ind].append(GA_motifs[ind][motifInd])

#in GA_motifs ignore "subject" field. The correct subject is in the "trial_frames" field
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'wb') as handle:
    pickle.dump(GA_sorted, handle, protocol=pickle.HIGHEST_PROTOCOL)

max_timepoint = waveData.get_data(dataBucketName).shape[-1]

#%%___________________ Make cleaned up dataFrames from GA_sorted
nSubjects = len(fileList)
nMaxTrials = max(len(sublist) for sublist in allTrialInfo)  # Maximum number of trials across subjects (sub 18 only has 460)

theta_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
alpha_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
freqs = ['theta', 'alpha']
freq_arrays = [theta_array, alpha_array]
for freq in range(len(GA_sorted)):
    tempGA = GA_sorted[freq]
    for sub in range(nSubjects):  
        nTrials = 460 if sub == 18 else nMaxTrials # Subject 18 only has 460 trials
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
        freq_arrays[freq][sub, :nTrials, :] = np.array(tempList)


conds = ['full stand', 'full trav in', 'full trav out', 'none'] 

# Split data by conditions
cond_indices = {cond: [] for cond in conds[:-1]}
for sub in range(nSubjects):
    for trial in range(nMaxTrials):
        condition = allTrialInfoPadded[sub][trial]
        if condition in cond_indices:
            cond_indices[condition].append((sub, trial))

time_vector = waveData.get_time()[:-2]

data = []
# Iterate through subjects and trials
for sub in range(nSubjects):
    for trial in range(nMaxTrials):
        condition = allTrialInfoPadded[sub][trial]
        for timepoint in range(nTimepoints):
            for freqInd, freq_array in enumerate(freq_arrays):
                motifInd = freq_array[sub, trial, timepoint]
                data.append([sub, trial, condition, timepoint, freqInd, motifInd])

df = pd.DataFrame(data, columns=['Subject', 'Trial', 'Condition', 'Timepoint', 'Frequency', 'MotifInd'])
df.to_csv(f"{figfolder}MotifCountsFull.csv", index=False)

# Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
motif_counts = df.groupby(['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject']).size().reset_index(name='Count')

# make all possible combinations
conditions = df['Condition'].unique()
timepoints = df['Timepoint'].unique()
frequencies = df['Frequency'].unique()
motif_inds = df['MotifInd'].unique()
subjects = df['Subject'].unique()

complete_index = pd.MultiIndex.from_product([conditions, timepoints, frequencies, motif_inds, subjects], names=['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject'])
# Reindex to include all combinations, fill missing values with 0
motif_counts = motif_counts.set_index(['Condition', 'Timepoint', 'Frequency', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()
#%% All subjects, all trials, all times
#  Define colormap and normalization
cmap = mcolors.ListedColormap(['grey', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
# Plot
for freqInd, freq in enumerate(freqs):
    fig, axs = plt.subplots(nrows=nSubjects, ncols=len(conds[:-1]), figsize=(10 * len(conds[:-1]), 3 * nSubjects))  # One column per condition
    for condInd, cond in enumerate(conds[:-1]):
        for subInd in range(nSubjects):
            ax = axs[subInd, condInd] if nSubjects > 1 else axs[condInd]
            trials = [trial for sub, trial in cond_indices[cond] if sub == subInd]
            if trials:
                data = np.vstack([freq_arrays[freqInd][subInd, trial, :] for trial in trials])
                im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm, extent=[time_vector[0], time_vector[-1], 0, data.shape[0]])
            ax.set_xlabel('Time')
            ax.set_ylabel('subject' + str(subInd))
            if subInd == 0:
                ax.set_title(f'Condition: {cond}')
            fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{figfolder}MotifCountsAllSubsAllTrialsAllTimes_{freq}.svg", format='svg', dpi=1200)
    plt.show()

#%%cut data to avoid edges
goodTimeRange = [-0.6, 1.5]
if "Simulations" in filePath:
    time_values = waveData.get_time()
else:
    time_values = waveData.get_time()[:-1]
start_time_idx = hf.find_nearest(time_values, goodTimeRange[0])[0]
end_time_idx = hf.find_nearest(time_values, goodTimeRange[1])[0]
goodTimeValues = time_values[start_time_idx:end_time_idx]
#%% Plot the motif counts for each motif
# Define conditions and colors
conditions = ['full stand', 'full trav in', 'full trav out']
colors = ['#0d586b', '#9c1f27', '#ba7b02']
qcolors = ['#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b']
subject_colors = plt.cm.tab20(np.linspace(0, 1, 19))

# Define preStim and postStim intervals
preStim = [-.5, -0.01]
preStimInds = [hf.find_nearest(time_values, preStim[0])[0], hf.find_nearest(time_values, preStim[1])[0]]
postStim = [0.5, 1.49]
postStimInds = [hf.find_nearest(time_values, postStim[0])[0], hf.find_nearest(time_values, postStim[1])[0]]

maxY = []
# Determine global min and max for the box plots
global_min = float('inf')
global_max = float('-inf')

for freq, _ in enumerate(freqs):
    for motifInd in range(len(GA_sorted[freq])):
        for cond in conditions:
            cond_data = motif_counts[(motif_counts['Condition'] == cond) & (motif_counts['Frequency'] == freq) & (motif_counts['MotifInd'] == motifInd)]
            for subject in cond_data['Subject'].unique():
                subject_data = cond_data[cond_data['Subject'] == subject]
                preStim_avg = subject_data[(subject_data['Timepoint'] >= preStimInds[0]) & (subject_data['Timepoint'] <= preStimInds[1])]['Count'].mean()
                postStim_avg = subject_data[(subject_data['Timepoint'] >= postStimInds[0]) & (subject_data['Timepoint'] <= postStimInds[1])]['Count'].mean()
                difference = postStim_avg - preStim_avg
                global_min = min(global_min, difference)
                global_max = max(global_max, difference)

# Iterate over frequencies and motifs
for freq, _ in enumerate(freqs):
    for motifInd in range(len(GA_sorted[freq])):
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 2, 1, 0])

        # Quiver plot
        ax0 = plt.subplot(gs[0])
        ax0.quiver(-np.real(GA_sorted[freq][motifInd]['average']), -np.imag(GA_sorted[freq][motifInd]['average']), color='black')
        ax0.set_facecolor('white')
        ax0.set_aspect('equal')
        for spine in ax0.spines.values():
            spine.set_edgecolor(qcolors[motifInd % len(qcolors)])
            spine.set_linewidth(2)

        # Line plot
        ax1 = plt.subplot(gs[1])
        ax1.set_facecolor('white')

        for cond, color in zip(conditions, colors):
            cond_data = motif_counts[(motif_counts['Condition'] == cond) & (motif_counts['Frequency'] == freq) & (motif_counts['MotifInd'] == motifInd)]
            mean = np.array(cond_data.groupby('Timepoint')['Count'].mean())[start_time_idx:end_time_idx]
            sem = np.array(cond_data.groupby('Timepoint')['Count'].sem())[start_time_idx:end_time_idx]
            ax1.plot(goodTimeValues, mean, color=color, linewidth=2)
            ax1.fill_between(goodTimeValues, mean - sem, mean + sem, color=color, alpha=0.2)
            maxY.append(mean + sem)

        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, preStim[0])[0]], color='red', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, preStim[1])[0]], color='red', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, postStim[0])[0]], color='green', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, postStim[1])[0]], color='green', linestyle=':', linewidth=2)

        # Adding a black line at time = 0
        ax1.axvline(x=time_values[hf.find_nearest(time_values, 0)[0]], color='black', linestyle='-', linewidth=3, alpha=0.5)
        ppatches = [patches.Patch(color=color, label=cond) for cond, color in zip(conditions, colors)]
        ax1.legend(handles=ppatches, loc='lower right')
        ax1.set_title(f'Motif {motifInd}')

        # Box plots
        ax2 = plt.subplot(gs[2])  
        ax2.set_facecolor('white')
        n_conditions = len(conditions)
        index = np.arange(n_conditions) 
        spacing = 0.5
        # Box plot
        all_differences = []

        for i, cond in enumerate(conditions):
            cond_data = motif_counts[(motif_counts['Condition'] == cond) & (motif_counts['Frequency'] == freq) & (motif_counts['MotifInd'] == motifInd)]
            preStim_averages = []
            postStim_averages = []

            for subject in cond_data['Subject'].unique():
                subject_data = cond_data[cond_data['Subject'] == subject]
                preStim_avg = subject_data[(subject_data['Timepoint'] >= preStimInds[0]) & (subject_data['Timepoint'] <= preStimInds[1])]['Count'].mean()
                postStim_avg = subject_data[(subject_data['Timepoint'] >= postStimInds[0]) & (subject_data['Timepoint'] <= postStimInds[1])]['Count'].mean()
                preStim_averages.append(preStim_avg)
                postStim_averages.append(postStim_avg)

            # Calculate the difference between pre-stim and post-stim
            differences = [post - pre for pre, post in zip(preStim_averages, postStim_averages)]
            all_differences.append(differences)

            ax2.boxplot(differences, positions=[index[i] * spacing], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor=colors[i], color=colors[i]),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color=colors[i]),
                        capprops=dict(color=colors[i]),
                        flierprops=dict(marker='o', color=colors[i],  markeredgecolor = None, markerfacecolor=colors[i]))
        # Draw a dashed line at y=0
        ax2.axhline(y=0, color='gray', linestyle='--')

        tick_positions = index * spacing
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(conditions)  # Label each box with the condition

        # Set the y-axis limits to the global min and max
        ax2.set_ylim([global_min, global_max])

        plt.tight_layout()
        ax1.set_ylim([0, np.max(maxY)+10])
        plt.savefig(f"{figfolder}MotifCounts_{freq}_{motifInd}Threshold.svg", format='svg', dpi=1200)
        plt.show()

#%%
from scipy import stats
fullTimeRange = waveData.get_time()

colors = ['#0d586b', '#9c1f27', '#ba7b02']
secondaryColors = ['#92bfca','#c79295', '#c8b285' ]


# Iterate over all unique motif indices
for motifInd in range(len(GA_sorted[freq])):
    for freq in frequencies:
        fig, ax = plt.subplots(figsize=(5, 5))  
        bar_width = 0.7
        indices = []
        for cond_index, cond in enumerate(conditions):
            base_index = cond_index * 2
            indices.append([base_index - 0.4, base_index + 0.4])
            cond_data = motif_counts[(motif_counts['Condition'] == cond) & (motif_counts['Frequency'] == freq) & (motif_counts['MotifInd'] == motifInd)]
           
            preStim_averages = []
            postStim_averages = []
            differences = []  
            
            for subject in range(len(allTrialInfo)):  
                subject_df = cond_data[cond_data['Subject'] == subject]                
                preStim_avg = subject_df[(subject_df['Timepoint'] >= preStimInds[0]) & (subject_df['Timepoint'] <= preStimInds[1])]['Count'].mean()
                postStim_avg = subject_df[(subject_df['Timepoint'] >= postStimInds[0]) & (subject_df['Timepoint'] <= postStimInds[1])]['Count'].mean()                
                preStim_averages.append(preStim_avg)
                postStim_averages.append(postStim_avg)
                differences.append(postStim_avg - preStim_avg)  
            
            # Stats
            mean_difference = np.mean(differences)
            std_difference = np.std(differences)
            t_stat, p_value = stats.ttest_rel(postStim_averages, preStim_averages)

            x_pos = indices[cond_index][0] + bar_width / 2
            y_pos = -0.2
            annotation_text = f"T={t_stat:.2f}, p={p_value:.3f}"
            text_color = 'red' if p_value < 0.05 else 'black'
            ax.text(x_pos, y_pos, annotation_text, ha='center', va='top', fontsize=8, color=text_color)            
            index = indices[cond_index]
            overall_preStim_avg = np.mean(preStim_averages)
            overall_postStim_avg = np.mean(postStim_averages)
            
            ax.bar(index[0], overall_preStim_avg, bar_width, color=secondaryColors[cond_index],  label=f'{condition} Pre-Stim' if freq == frequencies[0] else "")
            ax.bar(index[1], overall_postStim_avg, bar_width, color=colors[cond_index], label=f'{condition} Post-Stim' if freq == frequencies[0] else "")
            
            for subj_index, (preStimSingle, postStimSingle) in enumerate(zip(preStim_averages, postStim_averages)):
                ax.scatter(index[0], preStimSingle, color='black')
                ax.scatter(index[1], postStimSingle, color='black')
                line_color = 'blue' if postStimSingle > preStimSingle else 'red'
                ax.plot([index[0], index[1]], [preStimSingle, postStimSingle], color=line_color, linestyle='-', linewidth=1)

        ax.set_title(f'Motif {motifInd}, Frequency: {freq} Average Counts')
        ax.set_xticks([i[0] + 0.2 for i in indices])
        ax.set_xticklabels(conditions)
        ax.set_facecolor('white')
        ax.tick_params(axis='y', colors='black', grid_color = 'black', grid_alpha = 0.3)
        plt.savefig(f"{figfolder}BarGraphMotifCounts_{motifInd}_{freq}.svg", format='svg', dpi=1200)
        plt.show()


# Streamlines
conditions = ['full stand', 'full trav in', 'full trav out']
colors =  ['#0d586b', '#9c1f27', '#ba7b02']
qcolors = ['#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b']

for freqInd, freq in enumerate(freqs):
    fig, axs = plt.subplots(1, len(GA_sorted[freqInd]), figsize=(12, 6), gridspec_kw={'wspace': 0.3})
    for motifind in range(len(GA_sorted[freqInd])):
        average = GA_sorted[freqInd][motifind]['average']
        X, Y = np.meshgrid(np.arange(average.shape[1]), np.arange(average.shape[0]))
        U = -np.real(average)
        V = -np.imag(average)
        velocity = np.sqrt(U**2 + V**2)
        axs[motifind].quiver( U, V, color=qcolors[motifind])
        axs[motifind].streamplot( X, Y, U, V, color=velocity, cmap='viridis')
        axs[motifind].set_aspect('equal')
        axs[motifind].spines['bottom'].set_color(qcolors[motifind])
        axs[motifind].spines['top'].set_color(qcolors[motifind]) 
        axs[motifind].spines['right'].set_color(qcolors[motifind])
        axs[motifind].spines['left'].set_color(qcolors[motifind])
        for axis in ['top','bottom','left','right']:
            axs[motifind].spines[axis].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(f"{figfolder}MotifStreamlines_GA_{freq}.svg", format='svg', dpi=1200)
    plt.show()

# Polar histograms
for freqInd, freq in enumerate(freqs):
    fig, axs = plt.subplots(1, len(GA_sorted[freqInd]), figsize=(12, 6), subplot_kw={'polar': True}, gridspec_kw={'wspace': 0.3})
    for motifind in range(len(GA_sorted[freqInd])):
        average = GA_sorted[freqInd][motifind]['average']
        direction = np.arctan2(-np.imag(average[waveData.get_data("Mask")]), -np.real(average[waveData.get_data("Mask")]))
        axs[motifind].hist(direction.flatten(), bins=40, color=qcolors[motifind])
    plt.tight_layout()
    plt.savefig(f"{figfolder}PolarHistogram_GA_{freq}.svg", format='svg', dpi=1200)
    plt.show()

#%% Single subject plots___________________________________________________________________________________________________

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from itertools import islice

conditions = ['full stand', 'full trav in', 'full trav out']
colors = ['#0d586b', '#9c1f27', '#ba7b02']
qcolors = ['#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b']


#get global maximum y-value
global_max_y = 0
for freqInd in range(len(freqs)):
    for sub in range(len(fileList)):
        for cond in conditions:
            for motifInd in range(len(GA_sorted[freqInd])):
                subData = motif_counts[(motif_counts['Condition'] == cond) 
                                       & (motif_counts['Frequency'] == freqInd) 
                                       & (motif_counts['MotifInd'] == motifInd)
                                       & (motif_counts['Subject'] == sub)]['Count']
                global_max_y = max(global_max_y, np.max(subData))

for freqInd in range(len(freqs)):
    for sub in range(len(fileList)):
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, len(conditions))  # One row per condit
        for condInd, cond in enumerate(conditions):
            ax = plt.subplot(gs[0, condInd])
            ax.set_facecolor('white')
            for motifInd in range(len(GA_sorted[freqInd])):
                subData = motif_counts[(motif_counts['Condition'] == cond) 
                                       & (motif_counts['Frequency'] == freqInd) 
                                       & (motif_counts['MotifInd'] == motifInd)
                                       & (motif_counts['Subject'] == sub)]['Count']

                ax.plot(time_values, subData, color=qcolors[motifInd % len(qcolors)], linewidth=3)

            # Adding a black line at time = 0
            ax.axvline(x=time_values[hf.find_nearest(time_values, 0)[0]], color='black', linestyle='--', linewidth=2)
            ax.set_title(f'Condition: {cond}')
            ax.set_ylim([0, global_max_y + 10])

        plt.suptitle(f'Subject {sub} - Frequency {freqs[freqInd]}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{figfolder}Subject{sub}_MotifCounts_{freqs[freqInd]}.svg", format='svg', dpi=1200)
        plt.show()
#%%