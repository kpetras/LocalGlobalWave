#%%
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
folder = '<folder_path>'
avg_folder = '<avg_folder_path>'

# allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
# MotifsFromGA_File = 'Motifs_EEG_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>' 
# avg_figfolder = '<avg_figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 

# allMotifsFile = 'AllCondsMotifsMEG_NoThreshold'
# MotifsFromGA_File = 'Motifs_Mag_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>' 
# avg_figfolder = '<avg_figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 

allMotifsFile = 'AllCondsMotifsGrad_NoThreshold'
MotifsFromGA_File = 'Motifs_Grad_avg_OpticalFlowAfterFilter_Hilbert'
figfolder = '<figfolder_path>' 
avg_figfolder = '<avg_figfolder_path>' 
fileList = glob.glob(os.path.join(folder, "*", "**", "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
oscillationThresholdFlag = False 

# allMotifsFile = 'AllCondsMotifsSimulations_NoThreshold'
# MotifsFromGA_File = 'Motifs_Simulations_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, 'Simulations', 'sub*_Filter_Hilbert_OpticalFlow'))
# oscillationThresholdFlag = False
# 
#Load GA motifs

# load single trial motifs
filePath = fileList[0]

GA_motif_counts = []
allTrialInfo = []
#% average top motifs per subject
with open(folder + allMotifsFile +  '.pickle', 'rb') as handle:
    allmotifs = pickle.load(handle)
with open(avg_folder + MotifsFromGA_File + '.pickle', 'rb') as handle:
    GA_motifs = pickle.load(handle)
#load csv of GA motifs
GA_motif_df = pd.read_csv(f"{avg_figfolder}MotifCountsFull.csv")
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'rb') as handle:
    Motif_GA = pickle.load(handle)
with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'rb') as handle:
    allTrialInfo = pickle.load(handle)    
nSubs=19
conds = ['full stand', 'full trav in', 'full trav out']#order is important here. Needs to match that of the GA motifs
avgCondInfo = np.array(list(itertools.chain.from_iterable([[cond]*nSubs for cond in conds])))
trial_to_cond_map = {i: cond for i, cond in enumerate(avgCondInfo)}

#%%
pixelThreshold = 0.4
merge_threshold = .80
mask = GA_motifs[0][0]['average'] != 0
print("Merges if vector angles are below " + str(np.degrees(np.arccos(merge_threshold))) + " degrees")
minPixels = GA_motifs[0][0]['average'].shape[0]*GA_motifs[0][0]['average'].shape[1]*pixelThreshold
for freqInd in range(2):
        matchcount = 0
        for avgInd, avg_motif in enumerate(GA_motifs[freqInd]):#loop over all avg data derived motifs
            template = avg_motif['average'] # this is what we compare all single trial derived motifs to (if they match, the single trl data "looks like" the avg)
            subjects = [item[0] for item in avg_motif['trial_frames']]  # this is just to check later that the matching single trial motif actually comes from the same subject as the GA motif      

            for singleInd, single_motif in enumerate(Motif_GA[freqInd]): #loop over all single trial derived motifs 
                has_merged = False
                
                normalized_avg_motif = avg_motif['average'][mask] / np.abs(avg_motif['average'][mask])
                normalized_single_motif = single_motif['average'][mask] / np.abs(single_motif['average'][mask])
                cosine_similarity = np.real(normalized_avg_motif * np.conj(normalized_single_motif))
                if np.sum(cosine_similarity >= merge_threshold) >= minPixels:
                    print(f"frequency {freqInd} :  {avgInd} matches {singleInd}")
                    matchcount +=1                    
                    plt.quiver(np.real(avg_motif['average']), np.imag(avg_motif['average']), color='r')
                    plt.axis('equal')
                    plt.title(f"GA motif {avgInd}")
                    plt.show()
                    plt.quiver(np.real(single_motif['average']), np.imag(single_motif['average']), color='b')
                    plt.axis('equal')
                    plt.title(f"Single trial motif {singleInd}")
                    plt.show()
#%%Match GA motifs to single trial UV maps, per subject and condition
epsilon = 1e-6
freqs=[5,10]
templateMatch = np.full([19,2, 480, 750],-1, dtype = int) 
magnitudeThreshold=.1
matching_templates = [[[] for _ in range(2)] for _ in range(nSubs)]
# Because trialframes are named stupid in the GA motifs, we need to find the correct condition
# in GA motifs, the conditions are stacked as subjects resulting in 57 'subjects' (19 subs cond 1, 19 subs cond 2 and 19 subs cond 3)
conds = ['stand', 'trav in', 'trav out']
GAcond_indices = {cond: GA_motif_df[GA_motif_df['Condition'] == cond]['Trial'].unique() for cond in conds}
for sub in range(nSubs):
    filePath = fileList[sub]
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

    SingleTrialcondInfo  = waveData.get_trialInfo()
    # Find Motifs   
    sample_rate = waveData.get_sample_rate()  
    for freqInd in range(waveData.get_data(dataBucketName).shape[0]):
        if 'Simulations' in filePath:
            threshold = .8
            pixelThreshold = .6
            mergeThreshold = .8
        else:
            threshold = .85
            pixelThreshold = .4
            mergeThreshold = .7
        nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd]))
        minFrames = int(np.floor((waveData.get_sample_rate() / freqs[1])))
        matchcount = 0
        templates = []
        #find GA motifs that occur in that sub
        data = waveData.get_data(dataBucketName)[freqInd, :, :, :, :] 
        for avgInd, avg_motif in enumerate(GA_motifs[freqInd]):
            subconds = [values[sub] for values in GAcond_indices.values() if sub < len(values)]
            #check if any of the subconds[sub] are in the trial_frames of the GA motif            
            if any([item[0] in subconds for item in avg_motif['trial_frames']]):
                templates.append(avg_motif)
                #loop over the templates and compare to single trial data. Make density plots for GA motif and GA single Trial matches
        for TemplateInd, templateMotif in enumerate(templates):
            template = templateMotif['average']
            has_match = False 
            for trl, testFrames in enumerate(data):
                for frame in range(testFrames.shape[2]):
                    testFrame = testFrames[:,:,frame]          
                    normalized_template = template /np.abs(template+ epsilon)
                    normalized_testFrame = testFrame /np.abs(testFrame+ epsilon)
                    magnitude_mask = np.abs(testFrame) > magnitudeThreshold
                    cosine_similarity = np.real(normalized_testFrame * np.conj(normalized_template))
                    meets_threshold = (cosine_similarity[magnitude_mask] > threshold).sum() >= minPixels 
                    if meets_threshold:
                        templateMatch[sub, freqInd, trl, frame] = TemplateInd  
                        has_match = True
            if has_match:
                matching_templates[sub][freqInd].append(templateMotif['average'])
#save templateMatch to file
with open(figfolder + 'MatchSingleTrialsToTemplate_MotifsFromAVG_UVmaps.pickle', 'wb') as handle:
    pickle.dump(templateMatch, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(figfolder + 'MatchingTemplates.pickle', 'wb') as handle:
    pickle.dump(matching_templates, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
 #%%_________________________plot______________________________________________
 # 
 # 
 # _____________________________________________________________________ 
# Read templateMatch from file
with open(figfolder + 'MatchSingleTrialsToTemplate_MotifsFromAVG_UVmaps.pickle', 'rb') as handle:
    templateMatch = pickle.load(handle)
allmatchproportions = []
sampling_rate = waveData.get_sample_rate()
time_vector = waveData.get_time()
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['gray', '#480384', '#f28c00','#d67258', '#416ae4', '#378b8c', '#7bc35b'])
bounds = [-1, 0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

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
            cond_indices = {cond: GA_motif_df[GA_motif_df['Condition'] == cond]['Trial'].unique() for cond in conds}
            # Plot            
            trialsinaverage = [cond_indices[cond][sub]]
            filtered_df = GA_motif_df[(GA_motif_df['Trial'].isin(trialsinaverage)) & (GA_motif_df['Frequency'] == freqInd)]
            data_pivot = filtered_df.pivot(index='Trial', columns='Timepoint', values='MotifInd')
            # Convert the pivot table to a NumPy array
            subaveragedata = data_pivot.values            
            # subaverageim = ax.imshow(subaveragedata, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm, extent=[time_vector[0], time_vector[-1], 0, data.shape[0]])
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Subject')
            # ax.set_title(f'Condition: {cond}')
            # fig.colorbar(subaverageim, ax=ax)
            # plt.tight_layout()
            # #plt.savefig(f"{figfolder}MotifCountsAllSubsTrialAverageAllTimes_{freq}.svg", format='svg', dpi=1200)
            # plt.show()

            condtrialsinsingle = [i for i, x in enumerate(SingleTrialcondInfo) if x.replace('full ', '') == cond]                      
            TemplateMatchimage = templateMatch[sub, freqInd, condtrialsinsingle, nTimepointsEdge:-nTimepointsEdge]
            colors = ['grey', '#480384', '#f28c00', '#d67258', '#416ae4', '#378b8c', '#7bc35b'][0:len(np.unique(TemplateMatchimage))]            
            unique_values = np.unique(TemplateMatchimage)
            custom_cmap = ListedColormap(colors)
            
            fig, (ax2, ax1, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [.3, 3, 1, 1]})            
            fig.suptitle(f"Subject: {sub}, Frequency Index: {freqInd}, Condition: {cond}", fontsize=16)
            
            # Need to fake a second row to make pcolormesh work
            motif_image = np.squeeze(np.stack([subaveragedata[:, nTimepointsEdge:-nTimepointsEdge], subaveragedata[:, nTimepointsEdge:-nTimepointsEdge]]))
            unique_motif_values = np.unique(motif_image)
            custom_cmap_motif = ListedColormap(colors[:len(unique_motif_values)])
            cax2 = ax2.pcolormesh(timepoints, np.arange(motif_image.shape[0]), motif_image, cmap=custom_cmap_motif, edgecolors='none', shading='auto', rasterized=True,)            
            ax2.set_title('Motif Sequence of Averaged Data')
            ax2.set_xlabel('Time (s)')
            ax2.set_yticks([])  
            ax2.grid(False)
            
            # All trials by all timepoints
            cax1 = ax1.pcolormesh(timepoints, np.arange(TemplateMatchimage.shape[0]), TemplateMatchimage, cmap=custom_cmap, edgecolors='none', rasterized=True, shading='auto')            
            ax1.grid(False)

            # Densities
            num_trials = TemplateMatchimage.shape[0]
            densities = np.zeros((len(unique_values), len(timepoints)))            
            for i, value in enumerate(unique_values):
                densities[i, :] = np.sum(TemplateMatchimage == value, axis=0) / num_trials
            
            for i, value in enumerate(unique_values):
                if value != -1:
                    ax3.plot(timepoints, densities[i, :], label=f'Value {value}', color=colors[i])
            ax3.set_title('Proportion of Each Value Over Time (Excluding -1)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Proportion')
            ax3.legend()
            ax3.grid(True)

            # Proportion of trials matching the motif sequence of averaged data
            motif_sequence = motif_image[0, :]
            match_proportion = np.zeros(len(timepoints))
            for t in range(len(timepoints)):
                match_proportion[t] = np.sum(TemplateMatchimage[:, t] == motif_sequence[t]) / num_trials
            #keep match_proportions for later
            allmatchproportions[freqInd][sub, condInd, :] = match_proportion
            #make everything where the GA motif is -1 into NaN
            allmatchproportions[freqInd][sub, condInd, motif_sequence == -1] = np.nan
            ax4.plot(timepoints, match_proportion, label='Match Proportion', color='black')
            ax4.set_title('Proportion of Trials Matching Averaged Data Motif Sequence')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Proportion')
            ax4.legend()
            ax4.grid(True)
            
            # fix-axis lims
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(timepoints[0], timepoints[-1])
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  
            plt.subplots_adjust(hspace=0.1)  
            plt.savefig(figfolder + f"AVG_vs_Single_Subject{sub}_Frequency{freqInd}_Condition{condInd}.svg")
            plt.show()

            plt.close(fig)
        # #plot matching templates
        # colors = ['grey', '#480384', '#f28c00', '#d67258', '#416ae4', '#378b8c', '#7bc35b']
        # for templateInd, template in enumerate(matching_templates[sub][freqInd]):
        #     plt.quiver(np.real(template), np.imag(template), color=colors[templateInd+1])
        #     plt.axis('equal')
        #     plt.title(f"Matching Template {sub} {freqInd} {templateInd}")
        #     plt.show()

 
#Do match plots across subjects
condcolors = ['#0d586b', '#9c1f27', '#ba7b02']
for freqInd in range(2):
    nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd])) 
    timepoints = waveData.get_time()[nTimepointsEdge:-(nTimepointsEdge+1)]
    plt.figure(figsize=(12, 8))
    ax = plt.gca()  # Get the current axes
    ax.set_facecolor('white')  # Set the background color to white
    for condInd, cond in enumerate(conds):
        nanmean_proportion = np.nanmean(allmatchproportions[freqInd][:, condInd, :], axis=0)
        nanstd_proportion = np.nanstd(allmatchproportions[freqInd][:, condInd, :], axis=0)
        n = np.sum(~np.isnan(allmatchproportions[freqInd][:, condInd, :]), axis=0)  # Number of non-NaN subjects
        nansem_proportion = nanstd_proportion / np.sqrt(n)  # Standard error of the mean
        plt.plot(timepoints, nanmean_proportion, label=cond, color=condcolors[condInd], linewidth=2)
        plt.fill_between(timepoints, nanmean_proportion - nansem_proportion, nanmean_proportion + nansem_proportion, color=condcolors[condInd], alpha=0.2)
    plt.title(f"Nan-mean Proportion of Trials Matching Averaged Data Motif Sequence (Frequency Index: {freqInd})")
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(False)  # Remove the gray grid
    plt.tight_layout()
    plt.savefig(figfolder + f"NanMean_MatchProportion_Frequency{freqInd}.svg", format='svg', dpi=1200)
    plt.show()



condcolors = ['#0d586b', '#9c1f27', '#ba7b02']

# Loop through frequencies
for freqInd in range(2):
    nTimepointsEdge = int(2 * (waveData.get_sample_rate() / freqs[freqInd])) 
    timepoints = waveData.get_time()[nTimepointsEdge:-(nTimepointsEdge+1)]
    
    # Initialize lists to store pre-stim and post-stim averages and errors
    preStim_averages = []
    postStim_averages = []
    preStim_errors = []
    postStim_errors = []
    
    for condInd, cond in enumerate(conds):
        nanmean_proportion = np.nanmean(allmatchproportions[freqInd][:, condInd, :], axis=0)
        nanstd_proportion = np.nanstd(allmatchproportions[freqInd][:, condInd, :], axis=0)
        n = np.sum(~np.isnan(allmatchproportions[freqInd][:, condInd, :]), axis=0)  # Number of non-NaN subjects
        nansem_proportion = nanstd_proportion / np.sqrt(n)  # Standard error of the mean
        
        # Calculate pre-stim and post-stim averages and errors
        preStim_avg = np.nanmean(nanmean_proportion[timepoints < 0])
        postStim_avg = np.nanmean(nanmean_proportion[timepoints >= 0])
        preStim_error = np.nanmean(nansem_proportion[timepoints < 0])
        postStim_error = np.nanmean(nansem_proportion[timepoints >= 0])
        
        preStim_averages.append(preStim_avg)
        postStim_averages.append(postStim_avg)
        preStim_errors.append(preStim_error)
        postStim_errors.append(postStim_error)
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(conds))
    
    # Plot pre-stim and post-stim bars with error bars
    bars1 = ax.bar(index, preStim_averages, bar_width, yerr=preStim_errors, label='Pre-Stim', color=[condcolors[i] for i in range(len(conds))], capsize=5)
    bars2 = ax.bar(index + bar_width, postStim_averages, bar_width, yerr=postStim_errors, label='Post-Stim', color=[condcolors[i] for i in range(len(conds))], alpha=0.7, capsize=5)
    
    # Add labels, title, and legend
    ax.set_xlabel('Conditions')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Average Proportion of Trials Matching Averaged Data Motif Sequence (Frequency Index: {freqInd})')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(conds)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(figfolder + f"BarGraph_MatchProportion_Frequency{freqInd}.svg", format='svg', dpi=1200)
    plt.show()






# %%
