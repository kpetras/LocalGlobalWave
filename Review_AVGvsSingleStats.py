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


from Modules.Utils import ImportHelpers, WaveData as wd, HelperFuns as hf
from Modules.PlottingHelpers import Plotting as plotting
from Modules.SpatialArrangement import SensorLayout
import pickle
import itertools
from itertools import islice
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
import pandas as pd
   
#%%
folder = '/mnt/Data/LoGlo/AVG/'
figsavefolder = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/AVG/' 
stats_all = []
p_all = []
#load random waveData to get info
waveData = ImportHelpers.load_wavedata_object(folder + '/EEG_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
timeVec = waveData.get_time()
fsample = waveData.get_sample_rate()

#%% 

# Initialize lists to store combined data
combined_below = []
combined_above = []


for modality in ['EEG', 'Mag', 'Grad']:
    if modality == 'EEG':
        allMotifsFile = 'Motifs_EEG_avg_OpticalFlowAfterFilter_Hilbert'
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/EEG/' 
        filePath = glob.glob(os.path.join(folder, "EEG_Average_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
        oscillationThresholdFlag = False
    elif modality == 'Mag':
        allMotifsFile = 'Motifs_Mag_avg_OpticalFlowAfterFilter_Hilbert'
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Mag/' 
        filePath = glob.glob(os.path.join(folder, "Mag_Average_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
        oscillationThresholdFlag = False  
    elif modality == 'Grad':
        allMotifsFile = 'Motifs_Grad_avg_OpticalFlowAfterFilter_Hilbert'
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Grad/' 
        filePath = glob.glob(os.path.join(folder, "GradAverage_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)[0]
        oscillationThresholdFlag = False

    file_path = folder +  allMotifsFile + '.pickle'

    with open(file_path, 'rb') as handle:
        GA_sorted = pickle.load(handle)

    csv_file_path = f"{figfolder}MotifCountsFull.csv"
    df = pd.read_csv(csv_file_path)
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
    #remove everything that has condition none

    # Proportion of motifs, pre- vs. post Stim
    goodTimeRange = [-0.6, 1.5]
    start_time_idx = hf.find_nearest(timeVec, goodTimeRange[0])[0]
    end_time_idx = hf.find_nearest(timeVec, goodTimeRange[1])[0]
    goodTimeValues = timeVec[start_time_idx:end_time_idx]
    preStim = [-.5, -0.01]
    preStimInds = [hf.find_nearest(timeVec, preStim[0])[0], hf.find_nearest(timeVec, preStim[1])[0]]
    postStim = [0.5, 1.49]
    postStimInds = [hf.find_nearest(timeVec, postStim[0])[0], hf.find_nearest(timeVec, postStim[1])[0]]
    
    # Plot counts for each motif
    df_below = motif_counts[(motif_counts['Timepoint'] >= preStimInds[0]) & (motif_counts['Timepoint'] <= preStimInds[1])]
    df_above = motif_counts[(motif_counts['Timepoint'] >= postStimInds[0]) & (motif_counts['Timepoint'] <= postStimInds[1])]

    motif_counts_below = df_below.groupby(['Frequency', 'MotifInd'])['Count'].sum().reset_index()
    motif_counts_above = df_above.groupby(['Frequency', 'MotifInd'])['Count'].sum().reset_index()
    motif_counts_below['Proportion'] = motif_counts_below.groupby('Frequency')['Count'].transform(lambda x: x / x.sum())
    motif_counts_above['Proportion'] = motif_counts_above.groupby('Frequency')['Count'].transform(lambda x: x / x.sum())

    motif_counts_below['Modality'] = modality
    motif_counts_above['Modality'] = modality

    combined_below.append(motif_counts_below)
    combined_above.append(motif_counts_above)

#%% summary stats
combined_below = pd.concat(combined_below)
combined_above = pd.concat(combined_above)

# Plotting
cmap = mcolors.ListedColormap(['grey', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
colors = [cmap(i) for i in range(len(combined_below['MotifInd'].unique()))]
frequencies = combined_below['Frequency'].unique()
freqnames = ['Theta', 'Alpha']

for freq in frequencies:
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    data_below = combined_below[combined_below['Frequency'] == freq]
    data_above = combined_above[combined_above['Frequency'] == freq]

    for i, modality in enumerate(['EEG', 'Mag', 'Grad']):
        data_below_modality = data_below[data_below['Modality'] == modality]
        data_above_modality = data_above[data_above['Modality'] == modality]

        # Plot stacked bar for pre StimPeriod
        bottom = 0
        for j, motif in enumerate(data_below_modality['MotifInd'].unique()):
            proportion = data_below_modality[data_below_modality['MotifInd'] == motif]['Proportion'].values[0]
            ax[0, i].bar(modality, proportion, bottom=bottom, color=colors[j], label=motif if i == 0 else "")
            bottom += proportion
        ax[0, i].set_title(f'{modality} - Proportion of Motifs pre StimPeriod')
        ax[0, i].set_ylim(0, 1)

        # Plot stacked bar for post StimPeriod
        bottom = 0
        for j, motif in enumerate(data_above_modality['MotifInd'].unique()):
            proportion = data_above_modality[data_above_modality['MotifInd'] == motif]['Proportion'].values[0]
            ax[1, i].bar(modality, proportion, bottom=bottom, color=colors[j], label=motif if i == 0 else "")
            bottom += proportion
        ax[1, i].set_title(f'{modality} - Proportion of Motifs post StimPeriod')
        ax[1, i].set_ylim(0, 1)

    # Add legend to the first subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.suptitle(f'Proportion of Motifs for {freqnames[freq]} Frequency')
    #plt.savefig(f"{figfolder}StackedBarProportionOfMotifs_{freqnames[freq]}.svg", format='svg', dpi=1200)
    plt.show()

# %% Single Trial Data
#_______________________________________________________________________________________________________
   
folder = '/mnt/Data/LoGlo/'
stats_all = []
p_all = []
#load random waveData to get info
waveData = ImportHelpers.load_wavedata_object('/mnt/Data/LoGlo/8KYLY7/EEG_OpticalFlowAfterFilter_Hilbert')
timeVec = waveData.get_time()
fsample = waveData.get_sample_rate()

#%%
# allMotifsFile = 'AllCondsMotifsSimulations'
# figfolder = '/home/kirsten/Dropbox/loglofigures_simulations/' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Simulations_Filter_Hilbert_OpticalFlow"), recursive=True)
# oscillationThresholdFlag = True 

# allMotifsFile = 'AllCondsMotifsSimulations_NoThreshold'
# figfolder = '/home/kirsten/Dropbox/loglofigures_simulations/NoThreshold/' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Simulations_Filter_Hilbert_OpticalFlow"), recursive=True)
# oscillationThresholdFlag = False 

combined_below_ST = []
combined_above_ST = []

for modality in ['EEG', 'Mag', 'Grad']:
    if modality == 'EEG':
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures/NoThreshold/' 
        fileList = glob.glob(os.path.join(folder, "*", "**", "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
        oscillationThresholdFlag = False
    elif modality == 'Mag':
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_meg/NoThreshold/' 
        fileList = glob.glob(os.path.join(folder, "*", "**", "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
        oscillationThresholdFlag = False 
    elif modality == 'Grad':
        figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_grad/NoThreshold/' 
        fileList = glob.glob(os.path.join(folder, "*", "**", "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
        oscillationThresholdFlag = False

    csv_file_path = f"{figfolder}MotifCountsFull.csv"
    df = pd.read_csv(csv_file_path)
    df = df[df['Condition'] != 'none']
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
    #remove everything that has condition none

    # Proportion of motifs, pre- vs. post Stim
    goodTimeRange = [-0.6, 1.5]
    start_time_idx = hf.find_nearest(timeVec, goodTimeRange[0])[0]
    end_time_idx = hf.find_nearest(timeVec, goodTimeRange[1])[0]
    goodTimeValues = timeVec[start_time_idx:end_time_idx]
    preStim = [-.5, -0.01]
    preStimInds = [hf.find_nearest(timeVec, preStim[0])[0], hf.find_nearest(timeVec, preStim[1])[0]]
    postStim = [0.5, 1.49]
    postStimInds = [hf.find_nearest(timeVec, postStim[0])[0], hf.find_nearest(timeVec, postStim[1])[0]]
    
    # Plot the motif counts for each motif
    df_below = motif_counts[(motif_counts['Timepoint'] >= preStimInds[0]) & (motif_counts['Timepoint'] <= preStimInds[1])]
    df_above = motif_counts[(motif_counts['Timepoint'] >= postStimInds[0]) & (motif_counts['Timepoint'] <= postStimInds[1])]

    motif_counts_below = df_below.groupby(['Frequency', 'MotifInd'])['Count'].sum().reset_index()
    motif_counts_above = df_above.groupby(['Frequency', 'MotifInd'])['Count'].sum().reset_index()
    motif_counts_below['Proportion'] = motif_counts_below.groupby('Frequency')['Count'].transform(lambda x: x / x.sum())
    motif_counts_above['Proportion'] = motif_counts_above.groupby('Frequency')['Count'].transform(lambda x: x / x.sum())

    motif_counts_below['Modality'] = modality
    motif_counts_above['Modality'] = modality
    combined_below_ST.append(motif_counts_below)
    combined_above_ST.append(motif_counts_above)



# Combine data for all modalities
combined_below_ST = pd.concat(combined_below_ST)
combined_above_ST = pd.concat(combined_above_ST)

# Plotting
cmap = mcolors.ListedColormap(['grey', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
colors = [cmap(i) for i in range(len(combined_below_ST['MotifInd'].unique()))]
frequencies = combined_below_ST['Frequency'].unique()
freqnames = ['Theta', 'Alpha']

for freq in frequencies:
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    data_below_ST = combined_below_ST[combined_below_ST['Frequency'] == freq]
    data_above_ST = combined_above_ST[combined_above_ST['Frequency'] == freq]

    for i, modality in enumerate(['EEG', 'Mag', 'Grad']):
        data_below_modality_ST = data_below_ST[data_below_ST['Modality'] == modality]
        data_above_modality_ST = data_above_ST[data_above_ST['Modality'] == modality]

        # Plot stacked bar for pre StimPeriod
        bottom = 0
        for j, motif in enumerate(data_below_modality_ST['MotifInd'].unique()):
            proportion = data_below_modality_ST[data_below_modality_ST['MotifInd'] == motif]['Proportion'].values[0]
            ax[0, i].bar(modality, proportion, bottom=bottom, color=colors[j], label=motif if i == 0 else "")
            bottom += proportion
        ax[0, i].set_title(f'{modality} - Proportion of Motifs pre StimPeriod')
        ax[0, i].set_ylim(0, 1)

        # Plot stacked bar for post StimPeriod
        bottom = 0
        for j, motif in enumerate(data_above_modality_ST['MotifInd'].unique()):
            proportion = data_above_modality_ST[data_above_modality_ST['MotifInd'] == motif]['Proportion'].values[0]
            ax[1, i].bar(modality, proportion, bottom=bottom, color=colors[j], label=motif if i == 0 else "")
            bottom += proportion
        ax[1, i].set_title(f'{modality} - Proportion of Motifs post StimPeriod')
        ax[1, i].set_ylim(0, 1)

    # Add legend to the first subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.suptitle(f'Proportion of Motifs for {freqnames[freq]} Frequency')
    #plt.savefig(f"{figfolder}StackedBarProportionOfMotifs_{freqnames[freq]}.svg", format='svg', dpi=1200)
    plt.show()


# Add a column to indicate data type and period
combined_below['DataType'] = 'AVG'
combined_below['Period'] = 'Pre'
combined_above['DataType'] = 'AVG'
combined_above['Period'] = 'Post'
combined_below_ST['DataType'] = 'ST'
combined_below_ST['Period'] = 'Pre'
combined_above_ST['DataType'] = 'ST'
combined_above_ST['Period'] = 'Post'

# Concatenate all data
all_data = pd.concat([
    combined_below, combined_above, combined_below_ST, combined_above_ST
], ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt
for freqInd, freq in enumerate(freqnames):
    no_motif_df = all_data[(all_data['MotifInd'] == -1) & (all_data['Frequency'] == freqInd)].copy()
    motif_df = all_data[(all_data['MotifInd'] == -1) & (all_data['Frequency'] == freqInd)].copy()

    # Calculate "proportion of motif" (1 - proportion of no motif)
    motif_df['ProportionMotif'] = 1 - no_motif_df['Proportion']

    # Prepare data for plotting
    # Separate pre and post
    pre_df = motif_df[motif_df['Period'] == 'Pre']
    post_df = motif_df[motif_df['Period'] == 'Post']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Upper: Pre
    sns.barplot(
        data=pre_df,
        x='Modality', y='ProportionMotif',
        hue='DataType',
        ci='sd',
        ax=axes[0],
        dodge=True
    )
    axes[0].set_title('Proportion of Motif (1 - No Motif), Pre-Stimulus')
    axes[0].set_ylabel('Proportion of Motif')
    axes[0].set_xlabel('')
    axes[0].set_ylim(0, 1)  
    axes[0].legend(title='DataType')

    # Lower: Post
    sns.barplot(
        data=post_df,
        x='Modality', y='ProportionMotif',
        hue='DataType',
        ci='sd',
        ax=axes[1],
        dodge=True
    )
    axes[1].set_title('Proportion of Motif (1 - No Motif), Post-Stimulus')
    axes[1].set_ylabel('Proportion of Motif')
    axes[1].set_xlabel('Modality')
    axes[1].set_ylim(0, 1) 
    axes[1].legend(title='DataType')

    plt.tight_layout()
    plt.savefig(f"{figsavefolder}ProportionOfMotif_PrePost_AVGvsST{freq}.svg", format='svg', dpi=1200)
    plt.savefig(f"{figsavefolder}ProportionOfMotif_PrePost_AVGvsST{freq}.png", format='png', dpi=1200)
    plt.show()