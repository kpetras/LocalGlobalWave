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

from Modules.Utils import ImportHelpers, WaveData as wd, HelperFuns as hf
from Modules.PlottingHelpers import Plotting as plotting
from Modules.SpatialArrangement import SensorLayout
import pickle
import itertools
from itertools import islice

import pandas as pd

#%%
folder = '<folder_path>'
#%%________Set files___________________________________________

allMotifsFile = 'Motifs_EEG_avg_OpticalFlowAfterFilter_Hilbert'
figfolder = '<figfolder_path>'
filePath = glob.glob(os.path.join(folder, "EEG_Average_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
oscillationThresholdFlag = False 

# allMotifsFile = 'Motifs_Mag_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>'
# filePath = glob.glob(os.path.join(folder, "Mag_Average_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
# oscillationThresholdFlag = False 

# allMotifsFile = 'Motifs_Grad_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>'
# filePath = glob.glob(os.path.join(folder, "GradAverage_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
# oscillationThresholdFlag = False  

# allMotifsFile = 'Motifs_Sim_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>'
# filePath = glob.glob(os.path.join(folder, "Sim_Average_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
# oscillationThresholdFlag = False
#______________________________________________________________________
GA_motif_counts = []
allmotifs = [[],[]]
allTrialInfo = []

print("Processing file: " + filePath)
waveData = ImportHelpers.load_wavedata_object(filePath)

conds = ['stand', 'trav in', 'trav out']   
freqs= [5, 10]

# Find Motifs   
sample_rate = waveData.get_sample_rate()  # your sample rate

powerBucketName = None

#combine cond dataBuckets
tempData = []
for cond in conds:
    if "Grad" in filePath:
        dataBucketName = 'UV_Angle_' + cond + '_fromAvgAnalytic_merged'
    else:
        dataBucketName = 'UV_Angle_' + cond + '_fromAvgAnalytic'  
    tempData.append(waveData.get_data(dataBucketName))
combinedData = np.concatenate(tempData, axis=1)
CombinedDataBucket = wd.DataBucket(combinedData, "UVfromAvgAnalytic_allConds", 'freq_trl_posx_posy_time', waveData.get_channel_names())
waveData.add_data_bucket(CombinedDataBucket)
condInfo = np.array(list(itertools.chain.from_iterable([[cond]*waveData.get_data(dataBucketName).shape[1] for cond in conds])))

for freqInd in range(waveData.get_data('UVfromAvgAnalytic_allConds').shape[0]):
    dataBucketName = "UVfromAvgAnalytic_allConds" 
    minFrames = int(np.floor((waveData.get_sample_rate() / freqs[1])))
    nTimepointsEdge = 5
    baselinePeriod = (100,240) if freqInd == 0 else None 
    motifs = hf.find_wave_motifs(waveData, 
                                        dataBucketName=dataBucketName, 
                                        oscillationThresholdDataBucket = powerBucketName,
                                        oscillationThresholdFlag = oscillationThresholdFlag,
                                        baselinePeriod=baselinePeriod,
                                        threshold = .8, 
                                        nTimepointsEdge=nTimepointsEdge,
                                        mergeThreshold = .7, 
                                        minFrames=minFrames, 
                                        pixelThreshold = .4, 
                                        magnitudeThreshold=.1,
                                        dataInds = (freqInd, slice(None), slice(None), slice(None), slice(None)),
                                        Mask = True)


    for motif in motifs:
        motif['frequency'] = freqInd
    allmotifs[freqInd].extend(motifs)

trial_to_cond_map = {i: cond for i, cond in enumerate(condInfo)}

max_timepoint = waveData.get_data(dataBucketName).shape[-1]
timeVec = waveData.get_time()

#%%cut down allmotifs to the top 6 (the rest is unimportant because rare)
#match at least the first two as well as possible to the single trial directions, then sort by count
if "EEG" in filePath:
    freq = 0
    new_order = [1,0,2,3,4,5]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]
    freq=1
    new_order= [0,1,2,4,5]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]
if "Mag" in filePath:
    freq = 0
    new_order = [0,1,3,2,4,5]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]
    freq=1
    new_order= [1,0,2,3,4,5]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]
if "Grad" in filePath:
    freq = 0
    new_order = [1,0,2,4,3]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]
    freq=1
    new_order= [0,1,3,2,4,5]
    allmotifs[freq] = [allmotifs[freq][i] for i in new_order]

#save all motifs
with open(folder + allMotifsFile +  '.pickle', 'wb') as handle:
                pickle.dump(allmotifs, handle, protocol=pickle.HIGHEST_PROTOCOL)

nTrials  = len(trial_to_cond_map)
theta_array = np.zeros((nTrials, max_timepoint), dtype=int)-1
alpha_array = np.zeros((nTrials, max_timepoint), dtype=int)-1
freqs = ['theta', 'alpha']
freq_arrays = [theta_array, alpha_array]
for freq in range(len(freqs)):
    tempGA = allmotifs[freq]
    tempList = [[-1 for _ in range(max_timepoint)] for _ in range(nTrials)]
    for motifInd, motif in enumerate(tempGA):
        trial_frames_list = motif['trial_frames']
        for trial_frame in trial_frames_list:
            (trial, (start_timepoint, end_timepoint)) = trial_frame
            if 0 <= trial < nTrials and 0 <= start_timepoint < max_timepoint and 0 <= end_timepoint <= max_timepoint:
                for timepoint in range(start_timepoint, end_timepoint):
                    tempList[trial][timepoint] = motifInd 
    # Convert tempList to a NumPy array and store it in the corresponding frequency array
    freq_arrays[freq][:nTrials, :] = np.array(tempList)

#plot some to check:
for trial in range(10):
    plt.plot(freq_arrays[0][trial, :])

time_vector = waveData.get_time()[:-2]
data = []
# Iterate through subjects and trials
for trial in range(nTrials):
    condition = condInfo[trial]
    for timepoint in range(max_timepoint):
        for freqInd, freq_array in enumerate(freq_arrays):
            motifInd = freq_array[trial, timepoint]
            data.append([trial, condition, timepoint, freqInd, motifInd])

df = pd.DataFrame(data, columns=['Trial', 'Condition', 'Timepoint', 'Frequency', 'MotifInd'])
df.to_csv(f"{figfolder}MotifCountsFull.csv", index=False)


# Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
motif_counts = df.groupby(['Condition', 'Timepoint', 'Frequency', 'MotifInd']).size().reset_index(name='Count')

# make all possible combinations
conditions = df['Condition'].unique()
timepoints = df['Timepoint'].unique()
frequencies = df['Frequency'].unique()
motif_inds = df['MotifInd'].unique()

complete_index = pd.MultiIndex.from_product([conditions, timepoints, frequencies, motif_inds], names=['Condition', 'Timepoint', 'Frequency', 'MotifInd'])
# Reindex to include all combinations, fill missing values with 0
motif_counts = motif_counts.set_index(['Condition', 'Timepoint', 'Frequency', 'MotifInd']).reindex(complete_index, fill_value=0).reset_index()
#%% All subjects, all trials, all times
#  Define colormap and normalization
cmap = mcolors.ListedColormap(['gray', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cond_indices = {cond: df[df['Condition'] == cond]['Trial'].unique() for cond in conds}
# Plot
for freqInd, freq in enumerate(freqs):
    fig, axs = plt.subplots(nrows=1, ncols=len(conds), figsize=(10 * len(conds), 3))  # One column per condition
    for condInd, cond in enumerate(conds):
        ax = axs[condInd]
        trials = [trial for trial in cond_indices[cond]]
        if trials:
            data = np.vstack([freq_arrays[freqInd][trial, :] for trial in trials])
            im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm, extent=[time_vector[0], time_vector[-1], 0, data.shape[0]])
        ax.set_xlabel('Time')
        ax.set_ylabel('Subject')
        ax.set_title(f'Condition: {cond}')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    #plt.savefig(f"{figfolder}MotifCountsAllSubsTrialAverageAllTimes_{freq}.svg", format='svg', dpi=1200)
    plt.show()

#%%cut data to avoid edges
goodTimeRange = [-0.6, 1.5]
if "Simulations" in filePath:
    time_values = timeVec
else:
    time_values = timeVec[:-1]
start_time_idx = hf.find_nearest(time_values, goodTimeRange[0])[0]
end_time_idx = hf.find_nearest(time_values, goodTimeRange[1])[0]
goodTimeValues = time_values[start_time_idx:end_time_idx]
#%% Plot the motif counts for each motif
conditions = ['stand', 'trav in', 'trav out']
colors = ['#0d586b', '#9c1f27', '#ba7b02']
qcolors = ['#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b']

preStim = [-.5, -0.01]
preStimInds = [hf.find_nearest(time_values, preStim[0])[0], hf.find_nearest(time_values, preStim[1])[0]]
postStim = [0.5, 1.49]
postStimInds = [hf.find_nearest(time_values, postStim[0])[0], hf.find_nearest(time_values, postStim[1])[0]]
maxY = []
for freq, _ in enumerate(freqs):
    for motifInd in range(len(allmotifs[freq])):
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

        # Quiver plot
        ax0 = plt.subplot(gs[0])
        ax0.quiver(-np.real(allmotifs[freq][motifInd]['average']), -np.imag(allmotifs[freq][motifInd]['average']), color='black')
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
            ax1.plot(goodTimeValues, mean, color=color, linewidth=3)

        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, preStim[0])[0]], color='red', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, preStim[1])[0]], color='red', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, postStim[0])[0]], color='green', linestyle=':', linewidth=2)
        ax1.axvline(x=goodTimeValues[hf.find_nearest(goodTimeValues, postStim[1])[0]], color='green', linestyle=':', linewidth=2)

        # Adding a black line at time = 0
        ax1.axvline(x=time_values[hf.find_nearest(time_values, 0)[0]], color='black', linestyle='-', linewidth=2)

        ppatches = [patches.Patch(color=color, label=cond) for cond, color in zip(conditions, colors)]
        ax1.legend(handles=ppatches, loc='lower right')
        ax1.set_title(f'Motif {motifInd}')    
        plt.tight_layout()
        plt.savefig(f"{figfolder}MotifCounts_{freq}_{motifInd}Threshold.svg", format='svg', dpi=1200)
        plt.show()

#%%
from scipy import stats
fullTimeRange = waveData.get_time()

colors = ['#0d586b', '#9c1f27', '#ba7b02']
secondaryColors = ['#92bfca','#c79295', '#c8b285' ]


# Iterate over all unique motif indices
for motifInd in range(len(allmotifs[freq])):
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
            
            preStim_avg = cond_data[(cond_data['Timepoint'] >= preStimInds[0]) & (cond_data['Timepoint'] <= preStimInds[1])]['Count'].mean()
            postStim_avg = cond_data[(cond_data['Timepoint'] >= postStimInds[0]) & (cond_data['Timepoint'] <= postStimInds[1])]['Count'].mean()                
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
            


        ax.set_title(f'Motif {motifInd}, Frequency: {freq} Average Counts')
        ax.set_xticks([i[0] + 0.2 for i in indices])
        ax.set_xticklabels(conditions)
        ax.set_facecolor('white')
        ax.tick_params(axis='y', colors='black', grid_color = 'black', grid_alpha = 0.3)
        plt.savefig(f"{figfolder}BarGraphMotifCounts_{motifInd}_{freq}.svg", format='svg', dpi=1200)
        plt.show()


# Streamlines
conditions = ['stand', 'trav in', 'trav out']
colors =  ['#0d586b', '#9c1f27', '#ba7b02']
qcolors = ['#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b']

for freqInd, freq in enumerate(freqs):
    fig, axs = plt.subplots(1, len(allmotifs[freqInd]), figsize=(12, 6), gridspec_kw={'wspace': 0.3})
    for motifind in range(len(allmotifs[freqInd])):
        average = allmotifs[freqInd][motifind]['average']
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
    fig, axs = plt.subplots(1, len(allmotifs[freqInd]), figsize=(12, 6), subplot_kw={'polar': True}, gridspec_kw={'wspace': 0.3})
    for motifind in range(len(allmotifs[freqInd])):
        average = allmotifs[freqInd][motifind]['average']
        direction = np.arctan2(-np.imag(average[waveData.get_data("Mask")]), -np.real(average[waveData.get_data("Mask")]))
        axs[motifind].hist(direction.flatten(), bins=40, color=qcolors[motifind])
    plt.tight_layout()
    plt.savefig(f"{figfolder}PolarHistogram_GA_{freq}.svg", format='svg', dpi=1200)
    plt.show()



#%%