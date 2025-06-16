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
# folder = '/mnt/Data/LoGlo/'
# avg_folder = '/mnt/Data/LoGlo/AVG/'
# figsavefolder = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/AVG/' 

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
# waveData = ImportHelpers.load_wavedata_object(avg_folder + 'GradAverage_18_OpticalFlowAfterFilter_Hilbert_masked')
# modality = 'Grad'


# 
#Load GA motifs

# load single trial motifs
filePath = fileList[0]

GA_motif_counts = []
allTrialInfo = []
#% motifs from averaged data
with open(avg_folder + MotifsFromGA_File + '.pickle', 'rb') as handle:
    GA_motifs = pickle.load(handle)
#load csv of GA motifs
GA_motif_df = pd.read_csv(f"{avg_figfolder}MotifCountsFull.csv")
#load single trial motifs
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'rb') as handle:
    Motif_FromSingleTrials = pickle.load(handle)
with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'rb') as handle:
    allTrialInfo = pickle.load(handle) 
with open(figfolder + 'MatchSingleTrialsToTemplate_MotifsFromAVG_UVmaps.pickle', 'rb') as handle:
    templateMatch = pickle.load(handle)   

nSubs=19
conds = ['full stand', 'full trav in', 'full trav out']#order is important here. Needs to match that of the GA motifs
avgCondInfo = np.array(list(itertools.chain.from_iterable([[cond]*nSubs for cond in conds])))
trial_to_cond_map = {i: cond for i, cond in enumerate(avgCondInfo)}      
freqs=[5,10]
nTimepoints = 750
max_timepoint = nTimepoints-1

#%%___________________ Make cleaned up dataFrames from GA_sorted
nSubjects = len(fileList)
nMaxTrials = max(len(sublist) for sublist in allTrialInfo)  # Maximum number of trials across subjects (sub 18 only has 460)
#make array with all motifInds for each subject, trial and timepoint
theta_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
alpha_array = np.zeros((nSubjects, nMaxTrials, nTimepoints), dtype=int)-1
freqs = ['theta', 'alpha']
freq_arrays = [theta_array, alpha_array]
for freq in range(len(Motif_FromSingleTrials)):
    tempGA = Motif_FromSingleTrials[freq]
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

MotifInSingleTrial = MotifInSingleTrial = np.stack(freq_arrays, axis=0)

#same trick for motifs from AVG
nTrials  = len(trial_to_cond_map)
theta_array = np.zeros((nTrials, max_timepoint), dtype=int)-1
alpha_array = np.zeros((nTrials, max_timepoint), dtype=int)-1
freqs = ['theta', 'alpha']
freq_arrays = [theta_array, alpha_array]
for freq in range(len(freqs)):
    tempGA = GA_motifs[freq]
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
MotifInAVG = np.stack(freq_arrays, axis=0) #note that becase I labeled trials stupidly, trials are currently mixing conds and subs 
nSubs=19
conds = ['full stand', 'full trav in', 'full trav out']#order is important here. Needs to match that of the GA motifs
avgCondInfo = np.array(list(itertools.chain.from_iterable([[cond]*nSubs for cond in conds])))
trial_to_cond_map = {i: cond for i, cond in enumerate(avgCondInfo)}  
#make into -freq,subs,cond,timepoint
MotifInAVG = MotifInAVG.reshape(len(freqs), len(conds), nSubs, max_timepoint)
MotifInAVG = MotifInAVG.transpose(0, 2, 1, 3)

templateMatch = np.transpose(templateMatch, (1, 0, 2, 3))

preStim = [-0.5, -0.01]
postStim = [0.5, 1.49]
time_values = waveData.get_time()[:-1]  # shape: (750,) or (749,)

# Find indices for pre and post stim
preStimInds = [hf.find_nearest(time_values, preStim[0])[0], hf.find_nearest(time_values, preStim[1])[0]]
postStimInds = [hf.find_nearest(time_values, postStim[0])[0], hf.find_nearest(time_values, postStim[1])[0]]

# Slice arrays for pre-stim and post-stim
# For MotifInSingleTrial and templateMatch (shape: 2, 19, 480, 750)
MotifInSingleTrial_pre = MotifInSingleTrial[..., preStimInds[0]:preStimInds[1]+1]
MotifInSingleTrial_post = MotifInSingleTrial[..., postStimInds[0]:postStimInds[1]+1]

templateMatch_pre = templateMatch[..., preStimInds[0]:preStimInds[1]+1]
templateMatch_post = templateMatch[..., postStimInds[0]:postStimInds[1]+1]

# For MotifInAVG (shape: 2, 19, 3, 749)
MotifInAVG_pre = MotifInAVG[..., preStimInds[0]:preStimInds[1]+1]
MotifInAVG_post = MotifInAVG[..., postStimInds[0]:postStimInds[1]+1]

#%% plot
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

arrays = {
    "MotifInSingleTrial": (MotifInSingleTrial_pre, MotifInSingleTrial_post),
    "MotifInAVG": (MotifInAVG_pre, MotifInAVG_post),
    "templateMatch": (templateMatch_pre, templateMatch_post)
}
array_order = ["MotifInSingleTrial", "MotifInAVG", "templateMatch", "SingleAndAvgNoTemplate"]
freq_labels = ['theta', 'alpha']
bar_colors = ['#90ee90', '#228B22', '#4682B4', '#FFA500']  # last is orange

n_freqs = MotifInSingleTrial.shape[0]
n_subs = MotifInSingleTrial.shape[1]
confidence_level = 0.95  # for 95% confidence intervals

for freq in range(n_freqs):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Layout parameters
    group_spacing = 1.0  # Space between pre and post groups
    bar_width = 0.35  # Width of each individual bar
    bar_spacing = 0.05  # Space between bars within a group
    
    # Calculate positions for each bar group
    x_base = np.arange(len(array_order)) * (2 * group_spacing)
    x_pre = x_base - bar_width/2 - bar_spacing/2
    x_post = x_base + bar_width/2 + bar_spacing/2

    # Calculate means and confidence intervals
    pre_means, pre_cis, post_means, post_cis = [], [], [], []

    # Standard bars
    for arr_name in array_order[:-1]:
        arr_pre, arr_post = arrays[arr_name]
        if arr_name == "MotifInAVG":
            arr_pre_freq = arr_pre[freq]  # (n_subs, n_conds, time)
            arr_post_freq = arr_post[freq]
            pre_props = np.array([
                np.count_nonzero(arr_pre_freq[sub] != -1) / arr_pre_freq[sub].size
                for sub in range(n_subs)
            ])
            post_props = np.array([
                np.count_nonzero(arr_post_freq[sub] != -1) / arr_post_freq[sub].size
                for sub in range(n_subs)
            ])
        else:
            arr_pre_freq = arr_pre[freq]  # (n_subs, trials, time)
            arr_post_freq = arr_post[freq]
            pre_props = np.array([
                np.count_nonzero(arr_pre_freq[sub] != -1) / arr_pre_freq[sub].size
                for sub in range(n_subs)
            ])
            post_props = np.array([
                np.count_nonzero(arr_post_freq[sub] != -1) / arr_post_freq[sub].size
                for sub in range(n_subs)
            ])
        
        # Calculate confidence intervals
        pre_mean = pre_props.mean()
        pre_ci = stats.t.interval(confidence_level, len(pre_props)-1, loc=pre_mean, scale=stats.sem(pre_props))
        post_mean = post_props.mean()
        post_ci = stats.t.interval(confidence_level, len(post_props)-1, loc=post_mean, scale=stats.sem(post_props))
        
        pre_means.append(pre_mean)
        pre_cis.append([pre_mean - pre_ci[0], pre_ci[1] - pre_mean])  # Store as [lower_err, upper_err]
        post_means.append(post_mean)
        post_cis.append([post_mean - post_ci[0], post_ci[1] - post_mean])

    # Single AND AVG but NOT templateMatch
    # For pre
    st_pre = MotifInSingleTrial_pre[freq]  # (n_subs, trials, time)
    avg_pre = MotifInAVG_pre[freq]         # (n_subs, conds, time)
    tm_pre = templateMatch_pre[freq]       # (n_subs, trials, time)
    avg_pre_broadcast = np.repeat(avg_pre, st_pre.shape[1] // avg_pre.shape[1], axis=1) if avg_pre.shape[1] != st_pre.shape[1] else avg_pre
    mask_pre = (st_pre != -1) & (avg_pre_broadcast != -1) & (tm_pre == -1)
    pre_props = np.array([
        np.count_nonzero(mask_pre[sub]) / mask_pre[sub].size
        for sub in range(n_subs)
    ])
    pre_mean = pre_props.mean()
    pre_ci = stats.t.interval(confidence_level, len(pre_props)-1, loc=pre_mean, scale=stats.sem(pre_props))
    pre_means.append(pre_mean)
    pre_cis.append([pre_mean - pre_ci[0], pre_ci[1] - pre_mean])

    # For post
    st_post = MotifInSingleTrial_post[freq]
    avg_post = MotifInAVG_post[freq]
    tm_post = templateMatch_post[freq]
    avg_post_broadcast = np.repeat(avg_post, st_post.shape[1] // avg_post.shape[1], axis=1) if avg_post.shape[1] != st_post.shape[1] else avg_post
    mask_post = (st_post != -1) & (avg_post_broadcast != -1) & (tm_post == -1)
    post_props = np.array([
        np.count_nonzero(mask_post[sub]) / mask_post[sub].size
        for sub in range(n_subs)
    ])
    post_mean = post_props.mean()
    post_ci = stats.t.interval(confidence_level, len(post_props)-1, loc=post_mean, scale=stats.sem(post_props))
    post_means.append(post_mean)
    post_cis.append([post_mean - post_ci[0], post_ci[1] - post_mean])

    # Plot pre and post bars
    for i, (mean, ci, color, label) in enumerate(zip(pre_means, pre_cis, bar_colors, array_order)):
        ax.bar(x_pre[i], mean, width=bar_width, color=color, 
               yerr=np.array([ci]).T,
               error_kw=dict(capsize=5, capthick=2),
               label=label if freq == 0 else "")
    
    for i, (mean, ci, color) in enumerate(zip(post_means, post_cis, bar_colors)):
        ax.bar(x_post[i], mean, width=bar_width, color=color,
               yerr=np.array([ci]).T,
               error_kw=dict(capsize=5, capthick=2))

    # Set x-axis labels and ticks
    ax.set_xticks(x_base)
    ax.set_xticklabels(array_order)
    
    # Add vertical lines to separate conditions
    for x in x_base:
        ax.axvline(x - group_spacing, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x_base[-1] + group_spacing, color='gray', linestyle=':', alpha=0.3)
    
    # Add text labels for pre and post
    for i, x in enumerate(x_base):
        ax.text(x - group_spacing/2, -0.05, 'Pre', ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(x + group_spacing/2, -0.05, 'Post', ha='center', va='top', transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Proportion not -1 (mean ± {int(confidence_level*100)}% CI)")
    ax.set_title(f"Proportion of valid motif indices\nFrequency: {freq_labels[freq]}")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{figsavefolder}{modality}BarAVGvsSinglePrePostMotifProportions_{freq_labels[freq]}.png", dpi=300)
    plt.savefig(f"{figsavefolder}{modality}BarAVGvsSinglePrePostMotifProportions_{freq_labels[freq]}.svg", format='svg')
    plt.show()

# %%
