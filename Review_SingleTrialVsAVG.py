#%%
import sys
import os

# Add the parent directory of the script (LocalGlobalWave) to the Python path
sys.path.append('/mnt/Data/LoGlo/LocalGlobalWave/LocalGlobalWave/')
from Modules.Utils import ImportHelpers, WaveData as wd, HelperFuns as hf
from Modules.PlottingHelpers import Plotting as plotting
from Modules.SpatialArrangement import SensorLayout
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import pandas as pd
from matplotlib.colors import ListedColormap
import itertools
#%%________Set files___________________________________________
folder = '/mnt/Data/LoGlo/'
avg_folder = '/mnt/Data/LoGlo/AVG/'
figsavefolder = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/AVG/' 

allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
MotifsFromGA_File = 'Motifs_EEG_avg_OpticalFlowAfterFilter_Hilbert'
figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures/NoThreshold/' 
avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/EEG/' 
fileList = glob.glob(os.path.join(folder, "*",  "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
oscillationThresholdFlag = False 
waveData = ImportHelpers.load_wavedata_object(avg_folder + 'EEG_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
modality = 'EEG'


# allMotifsFile = 'AllCondsMotifsMEG_NoThreshold'
# MotifsFromGA_File = 'Motifs_Mag_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_meg/NoThreshold/' 
# avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Mag/' 
# fileList = glob.glob(os.path.join(folder, "*",  "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# waveData = ImportHelpers.load_wavedata_object(avg_folder + 'Mag_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
# modality = 'Mag'

# allMotifsFile = 'AllCondsMotifsGrad_NoThreshold'
# MotifsFromGA_File = 'Motifs_Grad_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_grad/NoThreshold/' 
# avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Grad/' 
# fileList = glob.glob(os.path.join(folder, "*",  "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# waveData = ImportHelpers.load_wavedata_object(avg_folder + 'Grad_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
# modality = 'Grad'


# 
#Load GA motifs

# load single trial motifs
filePath = fileList[0]

GA_motif_counts = []
allTrialInfo = []
#% single trial top motifs per subject
with open(folder + allMotifsFile +  '.pickle', 'rb') as handle:
    ST_motifs = pickle.load(handle)
#% motifs from averaged data
with open(avg_folder + MotifsFromGA_File + '.pickle', 'rb') as handle:
    GA_motifs = pickle.load(handle)
#load csv of GA motifs
GA_motif_df = pd.read_csv(f"{avg_figfolder}MotifCountsFull.csv")
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'rb') as handle:
    Motif_GA = pickle.load(handle)
with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'rb') as handle:
    allTrialInfo = pickle.load(handle)    
with open(figfolder + 'MatchSingleTrialsToTemplate_MotifsFromAVG_UVmaps.pickle', 'rb') as handle:
    templateMatch = pickle.load(handle)
nSubs=19
conds = ['full stand', 'full trav in', 'full trav out']#order is important here. Needs to match that of the GA motifs
avgCondInfo = np.array(list(itertools.chain.from_iterable([[cond]*nSubs for cond in conds])))
trial_to_cond_map = {i: cond for i, cond in enumerate(avgCondInfo)}      
freqs=[5,10]

# #%% Plot avg 
# # Polar histograms
# for freqInd, freq in enumerate(freqs):
#     fig, axs = plt.subplots(1, len(GA_motifs[freqInd]), figsize=(12, 6), subplot_kw={'polar': True}, gridspec_kw={'wspace': 0.3})
#     for motifind in range(len(GA_motifs[freqInd])):
#         average = GA_motifs[freqInd][motifind]['average']
#         direction = np.arctan2(-np.imag(average[waveData.get_data("Mask")]), -np.real(average[waveData.get_data("Mask")]))
#         axs[motifind].hist(direction.flatten(), bins=40, color='k')
#     plt.tight_layout()
#     plt.savefig(figsavefolder + f"{modality}_AVG_Motifs_FreqInd_{freqInd}.svg", format='svg')
#     plt.savefig(figsavefolder + f"{modality}_AVG_Motifs_FreqInd_{freqInd}.jpg", format='jpg')
#     plt.show()

# for freqInd, freq in enumerate(freqs):
#     fig, axs = plt.subplots(1, len(GA_motifs[freqInd]), figsize=(12, 6), gridspec_kw={'wspace': 0.3})
#     for motifind in range(len(GA_motifs[freqInd])):
#         average = GA_motifs[freqInd][motifind]['average']
#         # Quiver plot: plot all vectors from origin
#         axs[motifind].quiver(-np.real(GA_motifs[freqInd][motifind]['average']), -np.imag(GA_motifs[freqInd][motifind]['average']), color='black')
#         axs[motifind].set_title(f"Motif {motifind}")
#         axs[motifind].set_aspect('equal')
#         axs[motifind].set_facecolor('white')
#         for spine in axs[motifind].spines.values():
#             spine.set_edgecolor('k')
#             spine.set_linewidth(2)


#     plt.tight_layout()
#     plt.savefig(figsavefolder + f"{modality}_AVG_Motifs_FreqInd_{freqInd}_quiver.svg", format='svg')
#     plt.savefig(figsavefolder + f"{modality}_AVG_Motifs_FreqInd_{freqInd}_quiver.jpg", format='jpg')
#     plt.show()



# %% Try: compare single trial defined motifs to average directly
ST_original_motif_df = pd.read_csv(f"{figfolder}MotifCountsFull.csv")

# Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
motif_counts_ST_original = ST_original_motif_df.groupby(['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject']).size().reset_index(name='Count')

# make all possible combinations
conditions = ST_original_motif_df['Condition'].unique()
timepoints = ST_original_motif_df['Timepoint'].unique()
frequencies = ST_original_motif_df['Frequency'].unique()
motif_inds = ST_original_motif_df['MotifInd'].unique()
subjects = ST_original_motif_df['Subject'].unique()

complete_index = pd.MultiIndex.from_product([conditions, timepoints, frequencies, motif_inds, subjects], names=['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject'])
# Reindex to include all combinations, fill missing values with 0
motif_counts_ST_original = motif_counts_ST_original.set_index(['Condition', 'Timepoint', 'Frequency', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()


allmatchproportions = []
sampling_rate = waveData.get_sample_rate()
time_vector = waveData.get_time()
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['gray', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
conds = ['stand', 'trav in', 'trav out']
nTimepoints_per_freq = []

for freqInd in range(len(freqs)):
    nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd]))
    nTimepoints = len(waveData.get_time()[nTimepointsEdge:-(nTimepointsEdge+1)])
    nTimepoints_per_freq.append(nTimepoints)

# Arrays: shape (nFreqs, nConds, nSubs, nTimepoints)
non_match_but_singleTrialMotif = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]
no_singleTrialMotif = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]


#%% plots
for freqInd in range(len(freqs)):
    nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd]))
    nTimepoints = len(waveData.get_time()[nTimepointsEdge:-(nTimepointsEdge+1)])
    nTimepoints_per_freq.append(nTimepoints)

# Arrays: shape (nFreqs, nConds, nSubs, nTimepoints)
match_counts_arr = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]
no_temporalMatch_arr = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]
no_match_to_AVG_arr = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]
no_singleTrialMotif = [np.full((len(conds), nSubs, nTimepoints_per_freq[f]), np.nan) for f in range(len(freqs))]

for sub in range(nSubs):
    SingleTrialcondInfo = allTrialInfo[sub]
    for freqInd in range(2):  
        nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd])) 
        timepoints = waveData.get_time()[nTimepointsEdge:-(nTimepointsEdge+1)]
        if len(allmatchproportions) <= freqInd:
            allmatchproportions.append(np.zeros((nSubs, len(conds), len(timepoints)))) #needs to happen here because timepoints is not known earlier, but only once per freq
        
        for condInd, cond in enumerate(conds):
            cmap = mcolors.ListedColormap(['gray', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
            bounds = [-1, 0, 1, 2, 3, 4, 5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            filtered_df = GA_motif_df[(GA_motif_df['Condition'] == cond) & (GA_motif_df['Frequency'] == freqInd)]
            data_pivot = filtered_df.pivot(index='Trial', columns='Timepoint', values='MotifInd')
            # Convert to array to make less annoying
            subaveragedata = data_pivot.values[sub]#only use the current sub          


            condtrialsinsingle = [i for i, x in enumerate(SingleTrialcondInfo) if x.replace('full ', '') == cond]                      
            TemplateMatchimage = templateMatch[sub, freqInd, condtrialsinsingle, nTimepointsEdge:-nTimepointsEdge]
            colors = ['gray', '#480384', '#f28c00', '#d67258', '#416ae4', '#378b8c', '#7bc35b'][0:len(np.unique(TemplateMatchimage))]            
            motif_image = np.squeeze(np.stack([subaveragedata[nTimepointsEdge:-nTimepointsEdge], subaveragedata[nTimepointsEdge:-nTimepointsEdge]]))
            unique_motif_values = np.unique(motif_image)
            custom_cmap_motif = ListedColormap(colors[:len(unique_motif_values)])       

            # motif_image: shape (1, n_timepoints)
            motif_sequence = motif_image[0]  # shape (n_timepoints,)
            motif_ids = np.unique(motif_sequence)
            # single trial data    
            # For each motif in motif_sequence, calculate number of matches
            # For each timepoint, count how many single trials match this motif
            matches = (TemplateMatchimage == motif_sequence) & (TemplateMatchimage!= -1)
            match_counts = matches.sum(axis=0) 
            #same for non-matching trials (like there is a motif, but not the same as in the avg)
            non_matches = (TemplateMatchimage != motif_sequence) & (TemplateMatchimage!= -1)
            non_matching_counts = non_matches.sum(axis=0)
            minus1s =(TemplateMatchimage == -1)
            minus1_counts = minus1s.sum(axis=0)
            # Store the counts for this subject and condition
            match_counts_arr[freqInd][condInd, sub, :] = match_counts
            no_temporalMatch_arr[freqInd][condInd, sub, :] = non_matching_counts
            no_match_to_AVG_arr[freqInd][condInd, sub, :] = minus1_counts

            mins1InOriginalSingleSub = motif_counts_ST_original[
                (motif_counts_ST_original['Subject'] == sub) &
                (motif_counts_ST_original['Condition'].str.contains(cond)) &
                (motif_counts_ST_original['Frequency'] == freqInd) &
                (motif_counts_ST_original['MotifInd'] == -1)
            ]
            mins1InOriginalSingleSub_counts = mins1InOriginalSingleSub['Count'].values[nTimepointsEdge:-nTimepointsEdge]
            no_singleTrialMotif[freqInd][condInd, sub, :] = mins1InOriginalSingleSub_counts

            
            # Plotting
            # # Calculate pixel edges for imshow extent
            # dt = np.mean(np.diff(timepoints))
            # edges = np.concatenate(([timepoints[0] - dt/2], timepoints + dt/2))

            # fig, (ax2, ax_match) = plt.subplots(
            #     2, 1, figsize=(12, 7), 
            #     gridspec_kw={'height_ratios': [.3, 2]}
            # )
            # fig.suptitle(f"Subject: {sub}, Frequency Index: {freqInd}, Condition: {cond}", fontsize=16)

            # # Top: motif_image using imshow for correct alignment
            # ax2.imshow(
            #     motif_image,
            #     aspect='auto',
            #     cmap=custom_cmap_motif,
            #     extent=[edges[0], edges[-1], 0, 1],
            #     interpolation='nearest'
            # )
            # ax2.set_xlim(timepoints[0], timepoints[-1])
            # ax2.set_title('Motif Sequence of Averaged Data')
            # ax2.set_xlabel('Time (s)')
            # ax2.set_yticks([])
            # ax2.grid(False)

            # # Plot counts
            # for motif in motif_ids:
            #     color_idx = np.where(unique_motif_values == motif)[0][0]
            #     color = colors[color_idx]
            #     # Find contiguous segments where motif_sequence == motif
            #     mask = motif_sequence == motif
            #     if np.any(mask):
            #         # Find contiguous segments
            #         idx = np.where(mask)[0]
            #         if len(idx) > 0:
            #             # Split into contiguous blocks
            #             splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            #             for block in splits:
            #                 color = custom_cmap_motif((motif - np.min(np.unique(motif_sequence))) / (np.ptp(np.unique(motif_sequence)) if np.ptp(np.unique(motif_sequence)) > 0 else 1))
            #                 ax_match.plot(
            #                     timepoints[block], match_counts[block],
            #                     color=color, linewidth=2
            #                 )
            # n_trials = motif_counts_ST_original[
            #     (motif_counts_ST_original['Condition'].str.contains(cond)) &
            #     (motif_counts_ST_original['Frequency'] == freqInd) &
            #     (motif_counts_ST_original['Subject'] == sub)
            # ]['Count'].groupby(motif_counts_ST_original['Timepoint']).sum().iloc[0]

            # # Compute "other motif" count: n_trials - (-1 counts) - match_counts
            # other_motif_counts = n_trials - mins1InOriginalSingleSub_counts - match_counts

            # ax_match.plot(
            #     timepoints, other_motif_counts, color='red', linewidth=2, linestyle='-', label='Other motif (not -1, and has no match in AVG)'
            # )

            # ax_match.plot(
            #     timepoints, non_matching_counts, color='black', linewidth=2, label='Non-matching Trials'
            # )
            # ax_match.plot(
            #     timepoints, minus1_counts, color='gray', linewidth=2, linestyle='--', label='-1 in Single Trials with avg template'
            # )
            # # Add -1 counts from original single-subject data
            # ax_match.plot(
            #     timepoints, mins1InOriginalSingleSub_counts, color='lightcoral', linewidth=2, linestyle='--', label='-1 in original Single Trials'
            # )
            # # gray bars where average motif is -1
            # for t, motif_val in enumerate(motif_sequence):
            #     if motif_val == -1:
            #         ax_match.axvspan(edges[t], edges[t+1], color='lightgray', alpha=0.5, zorder=0)
            # ax_match.set_ylabel('Count')
            # ax_match.set_xlabel('Time (s)')
            # ax_match.set_xlim(timepoints[0], timepoints[-1])
            # ax_match.set_ylim(0, max(1, np.nanmax([non_matching_counts, minus1_counts])))
            # ax_match.set_title('Count of Trials Matching Each Motif & Non-matching/-1 Count')
            # ax_match.legend()
            # ax_match.grid(True)

            # plt.tight_layout(rect=[0, 0, 1, 0.97])
            # #save a svg and as jpg
            # if not os.path.exists(figsavefolder):
            #     os.makedirs(figsavefolder)
            # plt.savefig(figsavefolder + f"{modality}_AVGvsSingle_Sub_{sub}_FreqInd_{freqInd}_Cond_{cond}.svg", format='svg')
            # plt.savefig(figsavefolder + f"{modality}_AVGvsSingle_Sub_{sub}_FreqInd_{freqInd}_Cond_{cond}.jpg", format='jpg')
            # plt.show()




#%% Make all subs plot as proportion and only match vs non-,atch
import matplotlib.pyplot as plt
import numpy as np

for freqInd, freq in enumerate(freqs):
    nTimepoints = nTimepoints_per_freq[freqInd]
    timepoints = waveData.get_time()[int(2 * (waveData.get_sample_rate() / freqs[freqInd])):-(int(2 * (waveData.get_sample_rate() / freqs[freqInd]))+1)]
    for condInd, cond in enumerate(conds):
        match_counts = match_counts_arr[freqInd][condInd, :, :]
        non_matching_counts = no_temporalMatch_arr[freqInd][condInd, :, :]
        mins1_original = no_singleTrialMotif[freqInd][condInd, :, :]

        nTrialsperCond = 160

        # Proportions (subjects x time)
        match_prop = match_counts / nTrialsperCond
        nonmatch_prop = non_matching_counts / nTrialsperCond
        minus1_prop = mins1_original / nTrialsperCond

        # Split indices for pre and post stim
        pre_mask = timepoints < 0
        post_mask = timepoints >= 0

        # Mean and SEM across time and subjects for each period
        def mean_sem(arr, mask):
            vals = np.nanmean(arr[:, mask], axis=1)  # mean over time, for each subject
            return np.nanmean(vals), np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))

        # Pre-stim
        pre_match, pre_match_sem = mean_sem(match_prop, pre_mask)
        pre_nonmatch, pre_nonmatch_sem = mean_sem(nonmatch_prop, pre_mask)
        pre_minus1, pre_minus1_sem = mean_sem(minus1_prop, pre_mask)
        # Post-stim
        post_match, post_match_sem = mean_sem(match_prop, post_mask)
        post_nonmatch, post_nonmatch_sem = mean_sem(nonmatch_prop, post_mask)
        post_minus1, post_minus1_sem = mean_sem(minus1_prop, post_mask)

        # Combine for plotting: [pre_match, pre_nonmatch, pre_minus1, post_match, post_nonmatch, post_minus1]
        means = [pre_match, pre_nonmatch, pre_minus1, post_match, post_nonmatch, post_minus1]
        sems = [pre_match_sem, pre_nonmatch_sem, pre_minus1_sem, post_match_sem, post_nonmatch_sem, post_minus1_sem]

        # X locations: 0,1,2 for pre, 4,5,6 for post (leave a gap)
        x = np.array([0, 1, 2, 4, 5, 6])
        bar_labels = ['Match', 'Non-match', 'No motif', 'Match', 'Non-match', 'No motif']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, yerr=sems, capsize=5, color=['#4C72B0']*3 + ['#DD8452']*3)

        # Set x-ticks in the center of each group
        ax.set_xticks([1, 5])
        ax.set_xticklabels(['Pre-stim', 'Post-stim'])
        # Add minor ticks for each bar
        for i, label in enumerate(bar_labels):
            ax.text(x[i], -0.04, label, ha='center', va='top', fontsize=10, rotation=45, color='k', transform=ax.get_xaxis_transform())

        ax.set_ylabel('Mean Proportion')
        ax.set_title(f'Proportions by Period, Freq {freq} Hz, Condition: {cond}')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        #plt.savefig(figsavefolder + f"{modality}_BarTriplets_Proportions_Freq_{freq}_Cond_{cond}.svg", format='svg')
        #plt.savefig(figsavefolder + f"{modality}_BarTriplets_Proportions_Freq_{freq}_Cond_{cond}.jpg", format='jpg')
        plt.show()
# %%
