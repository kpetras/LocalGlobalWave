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
import pandas as pd
   
#%%
folder = '<folder_path>'
stats_all = []
p_all = []
#load random waveData to get info
waveData = ImportHelpers.load_wavedata_object('<wavedata_path>')
timeVec = waveData.get_time()
fsample = waveData.get_sample_rate()

#%%
# allMotifsFile = 'AllCondsMotifsSimulations'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Simulations_Filter_Hilbert_OpticalFlow"), recursive=True)
# oscillationThresholdFlag = True 

# allMotifsFile = 'AllCondsMotifsSimulations_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Simulations_Filter_Hilbert_OpticalFlow"), recursive=True)
# oscillationThresholdFlag = False 

combined_below = []
combined_above = []

for modality in ['EEG', 'Mag', 'Grad']:
    if modality == 'EEG':
        figfolder = '<figfolder_path>' 
        fileList = glob.glob(os.path.join(folder, "*", "**", "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
        oscillationThresholdFlag = False
    elif modality == 'Mag':
        figfolder = '<figfolder_path>' 
        fileList = glob.glob(os.path.join(folder, "*", "**", "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
        oscillationThresholdFlag = False 
    elif modality == 'Grad':
        figfolder = '<figfolder_path>' 
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
    combined_below.append(motif_counts_below)
    combined_above.append(motif_counts_above)

    #---------------------Stats-------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    # do Chi-Square test for each frequency
    freqs = ['theta', 'alpha']
    for freq in freqs:
        freq_df = df[df['Frequency'] == freqs.index(freq)]

        time_points = freq_df['Timepoint'].unique()
        test_statistics = []
        p_values = []

        for time_point in time_points[start_time_idx:end_time_idx]:
            contingency_table = pd.crosstab(freq_df[freq_df['Timepoint'] == time_point]['Condition'],
                                            freq_df[freq_df['Timepoint'] == time_point]['MotifInd'])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            test_statistics.append(chi2)
            p_values.append(p)

        # do FDR correction for multiple testing
        rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=0.001, method='fdr_bh')

        # Plot 
        plt.style.use('seaborn-v0_8-paper')
        plt.figure(figsize=(10, 6))
        plt.plot(timeVec[time_points[start_time_idx:end_time_idx]], test_statistics, marker='o', linestyle='-', label='Chi-Square Test Statistics')

        # Mark significant points
        significant_mask = rejected
        significant_test_statistics = np.array(test_statistics)[significant_mask]
        significant_time_points = np.array(time_points[start_time_idx:end_time_idx])[significant_mask]
        significant_times = timeVec[significant_time_points]
        plt.scatter(significant_times, significant_test_statistics, color='red', label='Significant (Corrected p < 0.001)', zorder=5)

        plt.title(f'Chi-Square Test Statistics Over Time for {freq} Hz')
        plt.xlabel('Time')
        plt.ylabel('Chi-Square Test Statistics')
        plt.legend()
        plt.savefig(f"{figfolder}ChiSquareTestStatistics_{freq}.svg", format='svg', dpi=1200)
        plt.show()

        if significant_times.any():
            print(f'Significant time points for {freq} Hz at corrected p-value < 0.001: {significant_times}')
        else:
            print(f'No significant time points found for {freq} Hz.')

        stats_all.append(test_statistics)
        p_all.append(p_values)

# Combine data for all modalities
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
    plt.savefig(f"{figfolder}StackedBarProportionOfMotifs_{freqnames[freq]}.svg", format='svg', dpi=1200)
    plt.show()

#%%_________Summary plot___________________________________________________________________________________________________
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

colors = ['#1f77b4', '#d62728', '#2ca02c']  # still ugly, but less ugly than before
data_types = ['EEG', 'Magnetometers', 'Gradiometers']  # must be ame oder as in loop above

for freq_index in range(len(freqs)):
    # Create a new figure with white background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white') 
    plt.style.use('seaborn-v0_8-paper')

    ax.axhline(0, color='black')  
    ax.axvline(0, color='black')  

    for i, color in enumerate(colors):
        index = i * len(freqs) + freq_index
        test_statistics = stats_all[index]
        p_values = p_all[index]

        ax.plot(goodTimeValues, test_statistics, marker='o', linestyle='-', 
                 label=f'{data_types[i]} Freq {freq_index}', color=color)
    
    ax.set_facecolor('white')
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset_proportion = 0.05  # Adjust as needed

    for i, color in enumerate(colors):
        index = i * len(freqs) + freq_index
        p_values = p_all[index]
        #  FDR correction
        rejected, _, _, _ = multipletests(p_values, alpha=0.001, method='fdr_bh')
        significant_mask = rejected
        
        # offest to make it less ugly
        offset = (i + 1) * offset_proportion * y_range
        significant_y_value = y_min - offset  
        
        significant_time_points = [tp for tp, sig in zip(goodTimeValues, significant_mask) if sig]
        if significant_time_points:  
            ax.scatter(significant_time_points, [significant_y_value] * len(significant_time_points), 
                        color=color, edgecolor='black', zorder=5)

    ax.set_title(f'Chi-Square Test Statistics Over Time for Frequency {freq_index}')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Chi-Square Test Statistics')
    ax.legend()
    ax.grid(False)  

    plt.savefig(f"<save_path>/ChiSquareTestStatisticsAll_Freq{freq_index}.svg", 
                    format='svg', dpi=1200, facecolor='white')

    plt.show()




