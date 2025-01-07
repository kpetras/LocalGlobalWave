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
from Modules.SpatialArrangement import SensorLayout as sensors
from Modules.PlottingHelpers import Plotting as plotting
from Modules.Decomposition import Hilbert as hilb
from Modules.Preprocessing import Filter as filt
import pickle
import itertools
from itertools import islice
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import circmean, circstd
import pandas as pd

#%%
'''read dataframe with indeces
loop through wavedata
get velocieties of optical flow along primary motif axis
get phase lag between sensors (and grid points) along primary motif axis'''

folder = '<folder_path>'

# allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# frame_size_m = .38 #meters

# allMotifsFile = 'AllCondsMotifsMEG_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# frame_size_m = .36 #meters

# allMotifsFile = 'AllCondsMotifsGrad_NoThreshold'
# figfolder = '<figfolder_path>' 
# fileList = glob.glob(os.path.join(folder, "*", "**", "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# frame_size_m = .36 #meters

allMotifsFile = 'AllCondsMotifsSimulations_NoThreshold'
figfolder = '<figfolder_path>' 
fileList = glob.glob(os.path.join(folder, 'Simulations', 'sub*_Filter_Hilbert_OpticalFlow'))
oscillationThresholdFlag = False 
frame_size_m = .2 #meters
GA_motif_counts = []
allmotifs = []
allTrialInfo = []
#fileList= fileList[0:1] #[KP] just for testing, remove later!
OFspeeds = np.zeros((len(fileList), 2, 4)) #two freqs, 4 motifs
PhaseSpeeds = np.zeros((len(fileList), 2, 4)) 
#%%______________________________________________________________________
csv_file_path = f"{figfolder}MotifCountsFull.csv"
df = pd.read_csv(csv_file_path)
df = df[df['Condition'] != 'none']
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'rb') as handle:
    GA_sorted = pickle.load(handle)
anglethreshold = 15
velocity_data = []
freqs = [5, 10]
for sub, filePath in enumerate(fileList):
    print("Processing file: " + filePath)
    print(f"Subject {sub}")
    UVBucketName = 'UV_Angle'
    AnimationBucketName = "AnalyticSignal"
    waveData = ImportHelpers.load_wavedata_object(filePath)
    waveData.get_extentGeodesic() #extent of the geodesic distances along the grid
    Mask  = waveData.get_data('Mask')
    #if gradiometer data, merge optical Flow
    if 'Grad' in filePath:
        combined_uv_map = waveData.get_data('UV_Angle_GradX') + waveData.get_data('UV_Angle_GradY')
        CombinedUVBucket = wd.DataBucket(combined_uv_map, "CombinedUV", 'freq_trl_posx_posy_time', waveData.get_channel_names())
        waveData.add_data_bucket(CombinedUVBucket)
        waveData.log_history(["CombinedUV", "VectorSum"])
        UVBucketName = 'CombinedUV'
        AnalyticBucketName = "AnalyticSignalX"
        # waveData.delete_data_bucket('UV_Angle_GradX')
        # waveData.delete_data_bucket('UV_Angle_GradY')
    elif 'Simulations' in filePath:
        UVBucketName = 'UV'
        AnalyticBucketName = "AnalyticSignalInterpolatedMasked"
    else:
        AnalyticBucketName = "AnalyticSignal"
    for freqInd in range(waveData.get_data(UVBucketName).shape[0]):
        for motifInd in (np.unique(df['MotifInd'])[1:5]): #skip first one because its the "none" motif
            #find the relavnt flow vectors
            Motif=GA_sorted[freqInd][motifInd]['average']
            filtered_df = df[(df['MotifInd'] == motifInd) & (df['Subject'] == sub) & (df['Frequency'] == freqInd)]
            #go through the rows of filtereddf 
            IndexedData = []
            AnalyticSignal = []
            trlframelist = []
            for rowInd, row in filtered_df.iterrows():
                trl = row['Trial']
                timepoint = row['Timepoint']                
                IndexedData.append(waveData.get_data(UVBucketName)[freqInd,trl,:,:,timepoint])
                AnalyticSignal.append(waveData.get_data(AnalyticBucketName)[freqInd,trl,:,:,timepoint])
                trlframelist.append((trl,timepoint))
            IndexedData = np.array(IndexedData)
            AnalyticSignal = np.array(AnalyticSignal)
            #check
            if len(IndexedData) == 0:
                continue
            #plot some stuff, comment out for real running because slow
            # waveData.add_data_bucket(wd.DataBucket(np.transpose(AnalyticSignal[0:500,:,:], (1, 2, 0)), "tempPlot", "posx_posy_time", waveData.get_channel_names()))
            # waveData.add_data_bucket(wd.DataBucket(np.transpose(IndexedData[0:500,:,:], (1, 2, 0)), "tempUV", "posx_posy_time", waveData.get_channel_names()))
            # ani = plotting.plot_optical_flow(waveData, 
            #                                 UVBucketName='tempUV',
            #                                 PlottingDataBucketName = 'tempPlot', 
            #                                 plotangle=True,
            #                                 normVectorLength = False) 
            # gif_filename = figfolder +  'opticalFlow_freq' + str(freqInd)+ '_Motif' + str(motifInd) + '_masked.gif'
            # ani.save(gif_filename)
            
            plt.figure()
            plt.quiver(-np.real(Motif), -np.imag(Motif))
            plt.title(f"sub {sub} freq {freqInd} Motif {motifInd}")
            plt.show()

            plt.quiver(-np.mean(np.real(IndexedData), axis=0), np.mean(-np.imag(IndexedData), axis=0))
            plt.title(f"sub {sub} freq {freqInd} IndexedData {motifInd} average")
            plt.show()

            plt.quiver(-np.real(IndexedData[-1]), -np.imag(IndexedData[-1]))
            plt.title(f"sub {sub} freq {freqInd} IndexedData {motifInd}")
            plt.show()

            #get principle angle of motif         
            fig, axs = plt.subplots(1, subplot_kw={'polar': True}, gridspec_kw={'wspace': 0.3})
            direction = np.arctan2(-np.imag(Motif[Mask]), -np.real(Motif[Mask]))#angles in rad
            flattened_directions = direction.flatten()

            counts, bin_edges, _ = axs.hist(flattened_directions, bins=40, density=False, alpha=0.6)
            # sort bins by countand get the ones containing the top 90%
            sorted_indices = np.argsort(counts)[::-1]
            cumulative_counts = np.cumsum(counts[sorted_indices])
            total_counts = cumulative_counts[-1]
            top_bins = sorted_indices[cumulative_counts <= 0.9 * total_counts]

            weighted_sin_sum = 0
            weighted_cos_sum = 0
            total_weight = 0
            # weigh how much each bin counts for the mean by how mnay samples it contains
            for i in top_bins:
                bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
                bin_angles = flattened_directions[(flattened_directions >= bin_start) & (flattened_directions < bin_end)]
                bin_count = counts[i]
                weighted_sin_sum += np.sum(np.sin(bin_angles)) * bin_count
                weighted_cos_sum += np.sum(np.cos(bin_angles)) * bin_count
                total_weight += len(bin_angles) * bin_count

            mean_direction = np.arctan2(weighted_sin_sum / total_weight, weighted_cos_sum / total_weight)
            mean_direction = (mean_direction + 2 * np.pi) % (2 * np.pi)
            axs.plot([mean_direction, mean_direction], [0, max(counts)], color='red', lw=2, label="Weighted Mean Direction")
            axs.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
            print(f"Mean direction (radians): {mean_direction}")
            print(f"Mean direction (degrees): {np.degrees(mean_direction)}")
            
            #velocity
            nframes, posx, posy = IndexedData.shape  
            # Conversion parameters
            frame_rate = waveData.get_sample_rate()  
            frame_interval = 1 / frame_rate  # time between frames in seconds             
            pixel_resolution = Mask.shape[0]
            pixel_size_m = frame_size_m / pixel_resolution  # meters per pixel
            # make meandir into unit vec 
            dir_vector = np.array([np.cos(mean_direction), np.sin(mean_direction)])
            u = -np.real(IndexedData)
            v = -np.imag(IndexedData)

            # Project each (u, v) vector onto the direction vector
            dir_vector = dir_vector / np.linalg.norm(dir_vector)
            projections = u * dir_vector[0] + v * dir_vector[1]  # this is speed in pix/frame
            projections_in_mps = projections * pixel_size_m / frame_interval

            # average speed in meandir over time
            average_speed_per_frame = projections_in_mps[:, Mask].mean(axis=1)
            # Mask so plot doesn'tlook weird
            projections_in_mps[:, ~Mask] = np.nan
            #remove the outliers (that come from frames being stuck together from different trials)
            q1, q3 = np.percentile(average_speed_per_frame, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            filtered_speeds_OF = average_speed_per_frame[(average_speed_per_frame >= lower_bound) & (average_speed_per_frame <= upper_bound)]
            mean_velocity_OF = filtered_speeds_OF.mean()

            plt.figure(figsize=(12, 6))
            # Spatial distribution of speed in the reference direction for the first frame
            plt.subplot(1, 2, 1)
            plt.title("Flow Speed in Reference Direction")
            im = plt.imshow(np.mean(projections_in_mps, axis = 0), cmap='coolwarm', origin='lower')
            plt.colorbar(im, label='Speed in Reference Direction (m/s)')
            plt.xlabel('X position')
            plt.ylabel('Y position')

            # Show reference direction
            center_x, center_y = Mask.shape[0] // 2, Mask.shape[1] // 2
            arrow_length = 5
            plt.arrow(center_x, center_y, arrow_length * dir_vector[0], arrow_length * dir_vector[1], 
                    color='black', width=0.3, head_width=1, head_length=1)
            plt.text(center_x + arrow_length * dir_vector[0] / 2, center_y + arrow_length * dir_vector[1] / 2,
                    f"Ref Dir (θ={mean_direction:.2f} rad)", color='black', fontsize=10)

            # Plot of average speed over time in reference direction with outlier bounds
            plt.subplot(1, 2, 2)
            plt.plot(range(len(average_speed_per_frame)), average_speed_per_frame, color='blue', label="Average Speed")
            plt.axhline(y=lower_bound, color='red', linestyle='--', label="Outlier Bound")
            plt.axhline(y=upper_bound, color='red', linestyle='--')
            plt.title("Flow Speed in Reference Direction (uncorrected)\n"
                                    "Mean Velocity: {:.2f} m/s".format(mean_velocity_OF))
            plt.xlabel("Frame")
            plt.ylabel("Average Speed (m/s)")
            plt.grid(True)

            # Display mean velocity without outliers
            plt.axhline(y=mean_velocity_OF, color='green', linestyle='-', linewidth=1.5, label="Mean w/o Outliers")
            plt.legend()
            plt.tight_layout()
            plt.show() 


            #_________speed from phase__________

            # Apply the mask to the phase data first to ignore unreliable points
            PhaseData = np.angle(AnalyticSignal)            
            masked_phase_data = np.where(Mask, PhaseData, np.nan)  # Set values outside the mask to NaN          
            grad_x, grad_y, grad_z = hf.nan_gradient(masked_phase_data, 1, 1, 1)#gradient along x y and z, controls for wrapping and NaNs
            # Project spatial gradients onto the reference direction
            phase_diffs_along_dir = (grad_x * dir_vector[0] + grad_y * dir_vector[1])

            # for ii in range(15):
            #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            #     # Plot the actual PhaseData with circular colormap
            #     im1 = axes[0].imshow(PhaseData[ii], cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
            #     fig.colorbar(im1, ax=axes[0], label='Phase Data (rad)')
            #     axes[0].set_title(f'Phase Data at Frame {ii}')
            #     axes[0].set_xlabel('X position')
            #     axes[0].set_ylabel('Y position')

            #     # Plot the gradient along the direction of the phase data with circular colormap
            #     im2 = axes[1].imshow(phase_diffs_along_dir[ii], cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
            #     fig.colorbar(im2, ax=axes[1], label='Phase lags between neighboring pixels in Reference Direction (rad)')
            #     axes[1].set_title(f'Phase Lags at Frame {ii}')
            #     axes[1].set_xlabel('X position')
            #     axes[1].set_ylabel('Y position')

            #     plt.tight_layout()
            #     plt.show()

            ##___testplot
            # PhaseData_reshaped = np.expand_dims(PhaseData, axis=0).transpose(0, 2, 3, 1)
            # PhaseData_reshaped = PhaseData_reshaped[:,:,:,:749]
            # waveData.add_data_bucket(wd.DataBucket(PhaseData_reshaped, "PhaseData", "trl_posx_posy_time", waveData.get_channel_names()))
            # ani = plotting.animate_grid_data(waveData, DataBucketName = "PhaseData", dataInd = (0), probepositions=[(0,20),(10,20), (20,20),(30,20),(40,20)], plottype = "real")
            # ani.save('test.gif',  fps = 20)
            ##___endtestplot

            # get spatial frequency from phase lags across space
            masked_phase_diffs_along_dir =  -(np.where(Mask, phase_diffs_along_dir, np.nan)) #negative because we want the phase lag (wave travels in direction of lower phase)
            #average over space to get spatial frequency
            average_phase_diffs_along_dir = circmean(masked_phase_diffs_along_dir, 
                                                     axis=(1, 2), high=np.pi, low=-np.pi, 
                                                     nan_policy='omit')

            SF_cpp = average_phase_diffs_along_dir / (2 * np.pi)  # cycles per pixel
            
            #convert to cycles per meter (here via cycles per image to confirm with simdata)
            SF_cpi = SF_cpp * Mask.shape[0] 
            SF_cpm = SF_cpi / frame_size_m
            TF_Hz = freqs[freqInd]
            V_phase = TF_Hz / SF_cpm #velocity in m/s

            #remove the outliers (that come from frames being stuck together from different trials)
            average_speed_per_frame = V_phase[~np.isnan(V_phase)]
            q1, q3 = np.percentile(average_speed_per_frame, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            filtered_speeds = average_speed_per_frame[(average_speed_per_frame >= lower_bound) & (average_speed_per_frame <= upper_bound)]
            mean_velocity_phase = filtered_speeds.mean()
            
            avg_phase_map = circmean(PhaseData, axis=(0), high=np.pi, low=-np.pi, nan_policy='omit')
            # plt.figure(figsize=(12, 6))
            # # Spatial distribution of speed in the reference direction for the first frame
            # plt.subplot(1, 2, 1)
            # plt.title("average Phase map")
            # im = plt.imshow(avg_phase_map, cmap='coolwarm', origin='lower')
            # plt.colorbar(im)
            # plt.xlabel('X position')
            # plt.ylabel('Y position')

            # Show reference direction
            center_x, center_y = Mask.shape[0] // 2, Mask.shape[1] // 2
            arrow_length = 5
            plt.arrow(center_x, center_y, arrow_length * dir_vector[0], arrow_length * dir_vector[1], 
                    color='black', width=0.3, head_width=1, head_length=1)
            plt.text(center_x + arrow_length * dir_vector[0] / 2, center_y + arrow_length * dir_vector[1] / 2,
                    f"Ref Dir (θ={mean_direction:.2f} rad)", color='black', fontsize=10)

            # # Plot of average speed over time in reference direction with outlier bounds
            # plt.subplot(1, 2, 2)
            # plt.plot(range(len(average_speed_per_frame)), average_speed_per_frame, color='blue', label="Average Speed")
            # plt.axhline(y=lower_bound, color='red', linestyle='--', label="Outlier Bound")
            # plt.axhline(y=upper_bound, color='red', linestyle='--')
            # plt.title("Phase Speed Reference Direction\n"
            #                         "Mean Velocity: {:.2f} m/s".format(mean_velocity_phase))
            # plt.xlabel("Frame")
            # plt.ylim((lower_bound-2, upper_bound+2))
            # plt.ylabel("Average Speed (m/s)")
            # plt.grid(True)
            # plt.axhline(y=mean_velocity_phase, color='green', linestyle='-', linewidth=1.5, label="Mean w/o Outliers")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()  

            # # Plot the average spatial frequency over frames
            # plt.figure(figsize=(10, 5))
            # plt.plot(range(len(SF_cpm)), SF_cpi, color='green', label="Avg Spatial Frequency (cycles/image)")
            # plt.title("Average Spatial Frequency Over Frames")
            # plt.xlabel("Frame")
            # plt.ylabel("Spatial Frequency (cycles/meter)")
            # plt.grid(True)
            # plt.legend()
            # plt.show()


            if 'Simulations' in filePath:
                temporal_frequency = freqs[freqInd]
                #true velocity in meters per second
                actual_SF_cpm = waveData.get_SimInfo()[0]['SpatialFrequency'][0]/frame_size_m
                actual_velocity_m_per_s = temporal_frequency /actual_SF_cpm
                print(f"actual spatial frequency: {waveData.get_SimInfo()[0]['SpatialFrequency'][0]:.2f} cycles per image")
                print(f"estimated spatial frequency from phase: {np.nanmean(SF_cpi):.2f} cycles per image")
                print(f"actual spatial frequency: {actual_SF_cpm:.2f} cycles per meter")
                print(f"estimated spatial frequency from phase: {np.nanmean(SF_cpm):.2f} cycles per meter")
                print(f"observed velocity Optical Flow: {mean_velocity_OF:.2f} m/s")
                print(f"observed velocity Phase: {mean_velocity_phase:.2f} m/s")
                print(f"True velocity: {actual_velocity_m_per_s:.2f} m/s")                
                print(f"Difference between Optical Flow and true velocity: {abs(mean_velocity_OF - actual_velocity_m_per_s):.2f} m/s")
                print(f"Difference between Phase gradient and true velocity: {abs(mean_velocity_phase - actual_velocity_m_per_s):.2f} m/s")
                velocity_data.append({
                    'Subject': sub,
                    'Frequency': freqInd,
                    'Motif': motifInd,
                    'Observed Velocity Optical Flow': mean_velocity_OF,
                    'Observed Velocity Phase': mean_velocity_phase,
                    'True Velocity': actual_velocity_m_per_s
                })
            else:
                
                temporal_frequency = freqs[freqInd]
                print(f"estimated spatial frequency from phase: {np.nanmean(SF_cpi):.2f} cycles per image")
                print(f"estimated spatial frequency from phase: {np.nanmean(SF_cpm):.2f} cycles per meter")
                print(f"observed velocity Optical Flow: {mean_velocity_OF:.2f} m/s")
                print(f"observed velocity Phase: {mean_velocity_phase:.2f} m/s")
                print(f"Difference between Phase gradient and Optical Flow: {abs(mean_velocity_phase - mean_velocity_OF):.2f} m/s")
                velocity_data.append({
                    'Subject': sub,
                    'Frequency': freqInd,
                    'Motif': motifInd,
                    'Observed Velocity Optical Flow': mean_velocity_OF,
                    'Observed Velocity Phase': mean_velocity_phase
                })


velocity_df = pd.DataFrame(velocity_data)
velocity_df.to_csv(f"{figfolder}VelocityData.csv", index=False)
print("done")
velocity_df

            #%%

#Make common data frame from all datatypes

modalities = {
    'EEG': {
        'figfolder': '<figfolder_path>/loglofigures/NoThreshold/',
        'frame_size_m': 0.38
    },
    'MAG': {
        'figfolder': '<figfolder_path>/loglofigures_meg/NoThreshold/',
        'frame_size_m': 0.48
    },
    'GRAD': {
        'figfolder': '<figfolder_path>/loglofigures_grad/NoThreshold/',
        'frame_size_m': 0.48
    },
    'Simulations': {
        'figfolder': '<figfolder_path>/loglofigures_simulations/NoThreshold/',
        'frame_size_m': 0.2
    }
}

dataframes = []
for modality, info in modalities.items():
    # Get the list of files for the current modality
    file = glob.glob(os.path.join(info['figfolder'], '**', "VelocityData.csv"), recursive=True)
    df = pd.read_csv(file[0])
    df['Modality'] = modality
    df['Frame Size (m)'] = info['frame_size_m']
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv('<save_path>/combined_velocity_data.csv', index=False)

#table
#paired t-test between motifs
import pandas as pd
from scipy.stats import ttest_rel

combined_df = pd.read_csv('<save_path>/combined_velocity_data.csv')

grouped = combined_df.groupby(['Modality', 'Frequency'])

t_test_results = []
for (modality, frequency), group in grouped:
    motif_0 = group[group['Motif'] == 0]
    motif_1 = group[group['Motif'] == 1]
    
    if not motif_0.empty and not motif_1.empty:
        merged = pd.merge(motif_0, motif_1, on=['Subject', 'Frequency', 'Modality'], suffixes=('_motif0', '_motif1'))
        
        t_stat_phase, p_value_phase = ttest_rel(merged['Observed Velocity Phase_motif0'], merged['Observed Velocity Phase_motif1'])
        t_stat_of, p_value_of = ttest_rel(merged['Observed Velocity Optical Flow_motif0'], merged['Observed Velocity Optical Flow_motif1'])
        
        mean_phase_motif0 = motif_0['Observed Velocity Phase'].mean()
        std_phase_motif0 = motif_0['Observed Velocity Phase'].std()
        mean_phase_motif1 = motif_1['Observed Velocity Phase'].mean()
        std_phase_motif1 = motif_1['Observed Velocity Phase'].std()
        
        mean_of_motif0 = motif_0['Observed Velocity Optical Flow'].mean()
        std_of_motif0 = motif_0['Observed Velocity Optical Flow'].std()
        mean_of_motif1 = motif_1['Observed Velocity Optical Flow'].mean()
        std_of_motif1 = motif_1['Observed Velocity Optical Flow'].std()
        
        t_test_results.append({
            'Modality': modality,
            'Frequency': frequency,
            'T-Statistic Phase': t_stat_phase,
            'P-Value Phase': p_value_phase,
            'Mean Phase Motif 0': mean_phase_motif0,
            'Std Phase Motif 0': std_phase_motif0,
            'Mean Phase Motif 1': mean_phase_motif1,
            'Std Phase Motif 1': std_phase_motif1,
            'T-Statistic Optical Flow': t_stat_of,
            'P-Value Optical Flow': p_value_of,
            'Mean Optical Flow Motif 0': mean_of_motif0,
            'Std Optical Flow Motif 0': std_of_motif0,
            'Mean Optical Flow Motif 1': mean_of_motif1,
            'Std Optical Flow Motif 1': std_of_motif1
        })

t_test_df = pd.DataFrame(t_test_results)
t_test_df.to_csv('<save_path>/Velocity_motif1_vs_motif2_t_test_results.csv', index=False)






        


# %%
