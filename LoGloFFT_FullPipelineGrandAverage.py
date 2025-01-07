from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import time
import os
import copy
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
def plot_data(subplot, freqs, avg_data, single_chan_data, color, label_prefix, title, chan, show_legend=False):
    plt.subplot(subplot)
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

#%% collect all waveData objects into one
folder = "<folder_path>"
savefolder = "<savefolder_path>"

fileList = glob.glob(os.path.join(folder, "**", "MEG_Mag_WaveData"), recursive=True)
for sub,filePath in enumerate(fileList):
    if sub == 0:
        CatchAllMEG = list()
        CatchAllEEG = list()
        CatchAllGrad = list()
        CatchAllSim = list()
    MEG = ImportHelpers.load_wavedata_object(filePath)
    EEG = ImportHelpers.load_wavedata_object(filePath.replace("MEG_Mag_WaveData", "EEGWaveData"))
    Grad = ImportHelpers.load_wavedata_object(filePath.replace("MEG_Mag_WaveData", "MEG_Grad_WaveData"))
    Sim = ImportHelpers.load_wavedata_object(folder + '/Simulations/sub' + str(sub) + '_Filter_Hilbert_OpticalFlow')
    Sim.delete_data_bucket("SimulatedData")
    Sim.delete_data_bucket("0")
    Sim.delete_data_bucket("1")
    Sim.delete_data_bucket("NBFiltered")
    Sim.delete_data_bucket("AnalyticSignal")
    Sim.delete_data_bucket("AnalyticSignalInterpolated")
    Sim.delete_data_bucket("AnalyticSignalInterpolatedMasked")
    Sim.delete_data_bucket("Mask")
    Sim.delete_data_bucket("UV")


    #remove "fov" trials
    trialInfo = MEG.get_trialInfo()
    fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
    MEG.prune_trials(fovInds)    
    CondInfo = MEG.get_trialInfo()
    hf.average_over_trials(MEG, trialInfo = CondInfo)
    MEG.delete_data_bucket("Mag")

    trialInfo = EEG.get_trialInfo()
    fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
    EEG.prune_trials(fovInds)
    CondInfo = EEG.get_trialInfo()
    hf.average_over_trials(EEG, trialInfo = CondInfo)    
    EEG.delete_data_bucket("EEG")

    trialInfo = Grad.get_trialInfo()
    fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
    Grad.prune_trials(fovInds)
    CondInfo = Grad.get_trialInfo()
    hf.average_over_trials(Grad, trialInfo = CondInfo)
    Grad.delete_data_bucket("Grad")

    CatchAllMEG.append(MEG)
    CatchAllEEG.append(EEG)
    CatchAllGrad.append(Grad)
    CatchAllSim.append(Sim)

#%%merge the waveData objects
#!!!Important: this will simply use the sensor positions from the first object in the list and pretend that all the others have the same!!!
#Do not use for anything source-reconstruction related!!!
#the resulting 'trl' dim is actually subjects
report = mne.Report(verbose=True)
MEGwaveData = hf.merge_wavedata_objects(CatchAllMEG)
EEGwaveData = hf.merge_wavedata_objects(CatchAllEEG)        
GradwaveData = hf.merge_wavedata_objects(CatchAllGrad)  
SimwaveData = hf.merge_wavedata_objects(CatchAllSim)

#filter + Hilbert 
for modality in [  GradwaveData]:
    modname = [name for name, obj in globals().items() if obj is modality][0]
    figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/' + modname[:3]
    #set up a bunch of stuff in the first iteration
    time_vect = modality.get_time()
    windowLength = 1 #seconds
    windowsize = int(np.floor((windowLength*modality.get_sample_rate()))) #in samples
    fftfreqs = np.fft.fftfreq(windowsize, d=time_vect[1]-time_vect[0])#we use 1 second segments 
    fft_freqEdges = hf.bin_edges_from_centers(fftfreqs)
    #we want the first bin to be centered on 1 and the last on 40 Hz
    freqMin = 1 - ((fftfreqs[1]-fftfreqs[0])/2)
    freqMax = 40 + ((fftfreqs[1]-fftfreqs[0])/2)
    nbins = hf.find_nearest(fft_freqEdges, freqMax)[0] - hf.find_nearest(fft_freqEdges, freqMin)[0]
    nChans = modality.DataBuckets[modality.ActiveDataBucket].get_data().shape[1]    
    evoked_dict = {}

    #%FFT spectrum 
    Buckets = list(modality.DataBuckets.keys())
    Bucket = Buckets[0]

    chan= 94
    chan = 66
    fft_pre= np.zeros((nChans,windowsize),dtype=complex)
    x = modality.get_data(Bucket)[:, :, 0:250]
    fft_pre[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
    fft_pre_chan_AVG = np.mean(fft_pre, axis=0)
    fft_pre_singleChan = fft_pre[chan,:]
    
    fft_post= np.zeros((nChans,windowsize),dtype=complex)
    x = modality.get_data(Bucket)[:, :, 500:750]#last second of stimulus
    fft_post[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
    fft_post_chan_AVG = np.mean(fft_post, axis=0)
    fft_post_singleChan = fft_post[chan,:]    
    #% Plot  pre- and post-stim psd
    fig = plt.figure(figsize=(6, 6))
    plot_data(221, fftfreqs[1:40], fft_pre_chan_AVG[1:40], fft_pre_singleChan[1:40], 'darkblue', '', ' Pre-Stimulus',chan = chan, show_legend = False)
    plot_data(222, fftfreqs[1:40], fft_post_chan_AVG[1:40], fft_post_singleChan[1:40],'darkblue', '', ' Post-Stimulus',chan = chan, show_legend = True)
    plt.tight_layout()
    plt.show()  
     
    #find individual alpha from the pre-stim peak
    time_vect = modality.get_time()
    windowLength = 1 #seconds
    f_sample = modality.get_sample_rate()
    #initialize arrays to store the spectra
    from scipy.signal import welch
    dataPreStim = modality.get_data(Bucket)[:, :, 0:250]
    dataPostStim = modality.get_data(Bucket)[:, :, 500:750]

    nTrials, nChans, _ = dataPreStim.shape
    _, Pxx_den = welch(dataPreStim[0, 0, :], fs=f_sample)
    nbins = len(Pxx_den)

    specPreStim = np.zeros((nTrials, nChans, nbins))
    specPostStim = np.zeros((nTrials, nChans, nbins))

    # Compute the power spectra for each trial and channel
    for trial in range(nTrials):
        for chan in range(nChans):
            freqs, Pxx_den = welch(dataPreStim[trial, chan, :], fs=f_sample)
            specPreStim[trial, chan, :] = Pxx_den

            freqs, Pxx_den = welch(dataPostStim[trial, chan, :], fs=f_sample)
            specPostStim[trial, chan, :] = Pxx_den
    # Average the spectra over trials and channels
    avgSpecPreStim = np.mean(specPreStim, axis=(0, 1))
    avgSpecPostStim = np.mean(specPostStim, axis=(0, 1))
    #check the 5 Hz
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [2, 40]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    #fm.report(freqs, avgSpecPostStim, freq_range)
    fm.fit(freqs, avgSpecPostStim, [3, 30])

    bands = Bands({'theta' : [4, 8]})
    # Extract any alpha band peaks from the power spectrum model
    theta = get_band_peak_fm(fm, bands.theta)
    print(theta)

    #check pre-stim alpha
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [2, 40]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    #fm.report(freqs, avgSpecPreStim, freq_range)
    fm.fit(freqs, avgSpecPreStim, [3, 30])

    bands = Bands({'alpha' : [8, 12]})
    #get alpha from pre-stim peak
    alpha = get_band_peak_fm(fm, bands.alpha)
    print(alpha)
    if np.any(np.isnan(alpha)):
        alpha = [10]
    if np.any(np.isnan(theta)):
        theta = 5

    if 'Meg' in modname:
        chanInds = [(28,80),(0,51)]
    elif 'Grad' in modname:
        #something weird happens when combining grad ssensors in multiple data Buckets. Will fix later. 
        # for now, do only once and then do the rest in new waveDataObjects and copy the Buckets over 
        temp = copy.deepcopy(modality) 
        temp2 = copy.deepcopy(modality)    
        hf.combine_grad_sensors(modality,dataBucketName="Grad_average_full stand") 

        hf.combine_grad_sensors(temp,dataBucketName="Grad_average_full trav in")         
        inBucketX = wd.DataBucket(temp.get_data('Grad_average_full trav in_GradX'), 'Grad_average_full trav in_GradX', 'trl_chan_time', modality.get_channel_names())
        modality.add_data_bucket(inBucketX)
        inBucketY = wd.DataBucket(temp.get_data('Grad_average_full trav in_GradY'), 'Grad_average_full trav in_GradY', 'trl_chan_time', modality.get_channel_names())
        modality.add_data_bucket(inBucketY)

        hf.combine_grad_sensors(temp2,dataBucketName="Grad_average_full trav out")         
        inBucketX = wd.DataBucket(temp2.get_data('Grad_average_full trav out_GradX'), 'Grad_average_full trav out_GradX', 'trl_chan_time', modality.get_channel_names())
        modality.add_data_bucket(inBucketX)
        inBucketY = wd.DataBucket(temp2.get_data('Grad_average_full trav out_GradY'), 'Grad_average_full trav out_GradY', 'trl_chan_time', modality.get_channel_names())
        modality.add_data_bucket(inBucketY)

        chanInds = [(28,80),(0,51)]        
    elif 'EEG' in modname:
        chanInds = [(1,72),(38,28)]
    else:
        chanInds= True
    Surface, PolySurface = SensorLayout.create_surface_from_points(modality,
                                                            type='channels',
                                                            num_points=1000)
    
    SensorLayout.distance_along_surface(modality, Surface, tolerance=0.1, get_extent = chanInds, plotting= True)
    SensorLayout.distmat_to_2d_coordinates_Isomap(modality) #can also use MDS here
    
    for Bucket in Buckets:
        modality.set_active_dataBucket(Bucket)  
        if 'Grad' in modname:
            grid_x, grid_y,mask =SensorLayout.interpolate_pos_to_grid(
                modality, 
                dataBucketName = Bucket + "_GradX",
                numGridBins=18,        
                return_mask = True,
                mask_stretching = True)
            grid_x, grid_y, mask =SensorLayout.interpolate_pos_to_grid(
                modality, 
                dataBucketName = Bucket + "_GradY",
                numGridBins=18,
                return_mask = True,
                mask_stretching = True)
        else:                  
            grid_x, grid_y, mask =SensorLayout.interpolate_pos_to_grid(
                modality, 
                dataBucketName = Bucket,
                numGridBins=18,
                return_mask = True,
                mask_stretching = True)

    # make new distMat based on the interpolated grid
    positions = np.dstack((grid_x, grid_y)).reshape(-1, 2)
    distMat = SensorLayout.regularGrid(modality, positions)

    original_data_bucket = Buckets[0]
    interpolated_data_bucket = list(modality.DataBuckets.keys())[-1]
    if 'Grad' in modname:
        original_data_bucket = original_data_bucket + '_GradX'
        interpolated_data_bucket = original_data_bucket + 'Interpolated' 

    OrigInd  = (0,slice(None),300)
    InterpInd =(0,slice(None),slice(None),300)    
    fig = plotting.plot_interpolated_data(modality, original_data_bucket, interpolated_data_bucket,
                                            grid_x, grid_y, OrigInd, InterpInd,  type='')
    
    
    for freqInd in range(2):
        if freqInd == 0:
            freq = 5 #theta[0] we know the freq of interest here
        else:
            freq = alpha[0]
         #% do filter + Hilbert to get complex Timeseries 
        for Bucket in Buckets:
            #this is not pretty, but the optical flow needs to be done separately for X and Y gradients and then combined afterwards
            if 'Grad' in modname:
                #Grad X
                XBucket = Bucket + '_GradXInterpolated'
                filt.filter_narrowband(modality, dataBucketName = XBucket, LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
                hilb.apply_hilbert(modality, dataBucketName = 'NBFiltered')
                SensorLayout.apply_mask(modality, mask, dataBucketName = 'AnalyticSignal', overwrite = True, storeMask = True)
                modality.DataBuckets[XBucket + ' AnalyticSignal_' +  str(freqInd)] =  modality.DataBuckets.pop("AnalyticSignal")
        
                #do optical flow
                tStart = time.time()
                print("OpticalFlow started")
                OpticalFlow.create_uv(modality, 
                        applyGaussianBlur=False, 
                        type = "angle", 
                        Sigma=1, 
                        alpha = 0.1, 
                        nIter = 200, 
                        dataBucketName=XBucket + ' AnalyticSignal_' +  str(freqInd),
                        is_phase = False)
                print('optical flow took: ', time.time()-tStart)
                
                trialToPlot = 0
                modality.DataBuckets[XBucket + ' UV_Angle' +  str(freqInd)] =  modality.DataBuckets.pop("UV")
                modality.set_active_dataBucket(XBucket + ' UV_Angle' +  str(freqInd))
                ani = plotting.plot_optical_flow(modality, 
                                                UVBucketName = XBucket + ' UV_Angle' +  str(freqInd),
                                                PlottingDataBucketName = XBucket + ' AnalyticSignal_' +  str(freqInd), 
                                                dataInds = (trialToPlot, slice(None), slice(None), slice(None)),
                                                plotangle=True,
                                                normVectorLength = True)  
                ani.save(figfolder + '/' + XBucket + 'OpticalFlowAfterAverageFilter_Hilbert' + str(freqInd) + '.gif')
                #same for grad Y                
                YBucket = Bucket + '_GradYInterpolated'
                filt.filter_narrowband(modality, dataBucketName = YBucket, LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
                hilb.apply_hilbert(modality, dataBucketName = 'NBFiltered')
                SensorLayout.apply_mask(modality, mask, dataBucketName = 'AnalyticSignal', overwrite = True, storeMask = True)
                modality.DataBuckets[YBucket + ' AnalyticSignal_' +  str(freqInd)] =  modality.DataBuckets.pop("AnalyticSignal")
        
                #do optical flow
                tStart = time.time()
                print("OpticalFlow started")
                OpticalFlow.create_uv(modality, 
                        applyGaussianBlur=False, 
                        type = "angle", 
                        Sigma=1, 
                        alpha = 0.1, 
                        nIter = 200, 
                        dataBucketName=YBucket + ' AnalyticSignal_' +  str(freqInd),
                        is_phase = False)
                print('optical flow took: ', time.time()-tStart)
                
                trialToPlot = 0
                modality.DataBuckets[YBucket + ' UV_Angle' +  str(freqInd)] =  modality.DataBuckets.pop("UV")
                modality.set_active_dataBucket(YBucket + ' UV_Angle' +  str(freqInd))
                ani = plotting.plot_optical_flow(modality, 
                                                UVBucketName = YBucket + ' UV_Angle' +  str(freqInd),
                                                PlottingDataBucketName = YBucket + ' AnalyticSignal_' +  str(freqInd), 
                                                dataInds = (trialToPlot, slice(None), slice(None), slice(None)),
                                                plotangle=True,
                                                normVectorLength = True)  
                ani.save(figfolder + '/' + YBucket + 'OpticalFlowAfterAverageFilter_Hilbert' + str(freqInd) + '.gif')

            else:

                Bucket = Bucket + 'Interpolated'
                filt.filter_narrowband(modality, dataBucketName = Bucket, LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
                hilb.apply_hilbert(modality, dataBucketName = 'NBFiltered')
                SensorLayout.apply_mask(modality, mask, dataBucketName = 'AnalyticSignal', overwrite = True, storeMask = True)
                modality.DataBuckets[Bucket + ' AnalyticSignal_' +  str(freqInd)] =  modality.DataBuckets.pop("AnalyticSignal")
        
                #do optical flow
                tStart = time.time()
                print("OpticalFlow started")
                OpticalFlow.create_uv(modality, 
                        applyGaussianBlur=False, 
                        type = "angle", 
                        Sigma=1, 
                        alpha = 0.1, 
                        nIter = 200, 
                        dataBucketName=Bucket + ' AnalyticSignal_' +  str(freqInd),
                        is_phase = False)
                print('optical flow took: ', time.time()-tStart)
                
                trialToPlot = 0
                modality.DataBuckets[Bucket + ' UV_Angle' +  str(freqInd)] =  modality.DataBuckets.pop("UV")
                modality.set_active_dataBucket(Bucket + ' UV_Angle' +  str(freqInd))
                # ani = plotting.plot_optical_flow(modality, 
                #                                 UVBucketName = Bucket + ' UV_Angle' +  str(freqInd),
                #                                 PlottingDataBucketName = Bucket + ' AnalyticSignal_' +  str(freqInd), 
                #                                 dataInds = (trialToPlot, slice(None), slice(None), slice(None)),
                #                                 plotangle=True,
                #                                 normVectorLength = True)  
                # ani.save(figfolder + '/' + Bucket + 'OpticalFlowAfterAverageFilter_Hilbert' + str(freqInd) + '.gif')

    modality.save_to_file(savefolder + Bucket[0:4] + 'AlltheStuffAverage_18_OpticalFlowAfterFilter_Hilbert_masked')


    # Define conditions, gradient types, data types, and frequencies
    conds = ['stand', 'trav in', 'trav out']
    GradTypes = ['GradX', 'GradY']
    dataTypes = ['AnalyticSignal_', 'UV_Angle']
    freqs = ['0', '1']

    # Get all buckets
    AllBuckets = list(modality.DataBuckets.keys())

    # Initialize new data buckets
    new_data_buckets = {}

    # Check if modname contains 'Grad'
    is_grad = 'Grad' in modname
    if "MEG" in modname:
        modname = "MagwaveData"

    for dataType in dataTypes:
        data_list = {cond: {gradType: [] for gradType in GradTypes} if is_grad else [] for cond in conds}
        if is_grad:
            for gradType in GradTypes:
                for cond in conds:
                    for freqInd in freqs:
                    
                        Bucket = f"{modname[:4]}_average_full {cond}_{gradType}Interpolated {dataType}{freqInd}"
                        if Bucket in AllBuckets:
                            tempData = modality.get_data(Bucket)
                            data_list[cond][gradType].append(tempData)

                # Concatenate the data for each condition
                for cond in conds:
                        concatenated_data = np.stack(data_list[cond][gradType], axis=0)
                        if dataType == 'UV_Angle':
                            new_bucket_name = f"UV_Angle_{cond}_fromAvgAnalytic{gradType}"
                        else:
                            new_bucket_name = f"{cond}_{dataType}{gradType}"
                        new_data_buckets[new_bucket_name] = concatenated_data

                # Add the new data buckets to the modality
                for bucketName, data in new_data_buckets.items():
                    CombinedBucket = wd.DataBucket(data, bucketName, 'freq_trl_posx_posy_time', modality.get_channel_names())
                    modality.add_data_bucket(CombinedBucket)

        # for all other sensor types
        else:
            for cond in conds:
                for freqInd in freqs:
                        Bucket = f"{modname[:3]}_average_full {cond}Interpolated {dataType}{freqInd}"
                        if Bucket in AllBuckets:
                            tempData = modality.get_data(Bucket)
                            data_list[cond].append(tempData)
            
            # Concatenate the data for each condition
            for cond in conds:
                    concatenated_data = np.stack(data_list[cond], axis=0)
                    if dataType == 'UV_Angle':
                        new_bucket_name = f"UV_Angle_{cond}_fromAvgAnalytic"
                    else:
                        new_bucket_name = f"{cond}_{dataType}"
                    new_data_buckets[new_bucket_name] = concatenated_data
            # Add the new data buckets to the modality
            for bucketName, data in new_data_buckets.items():
                CombinedBucket = wd.DataBucket(data, bucketName, 'freq_trl_posx_posy_time', modality.get_channel_names())
                modality.add_data_bucket(CombinedBucket)

    if is_grad:
        #merge the UV maps across grad types
        data_list = {cond: [] for cond in conds}
        for cond in conds:
            BucketNames = [f"UV_Angle_{cond}_fromAvgAnalytic{gradType}" for gradType in GradTypes]
            data_list[cond] = [modality.get_data(BucketName) for BucketName in BucketNames]
            merged_data  = data_list[cond][0] + data_list[cond][1]
            new_bucket_name = f"UV_Angle_{cond}_fromAvgAnalytic_merged"
            CombinedBucket = wd.DataBucket(merged_data, new_bucket_name, 'freq_trl_posx_posy_time', modality.get_channel_names())
            modality.add_data_bucket(CombinedBucket)
            modality.delete_data_bucket(BucketNames[0])
            modality.delete_data_bucket(BucketNames[1])

    # Delete all the old stuff
    for bucket_name in AllBuckets:
        if bucket_name not in new_data_buckets: 

            if bucket_name != "Mask":
                if "merged" not in bucket_name:
                    modality.delete_data_bucket(bucket_name)


    modality.save_to_file(savefolder + Bucket[0:4] + 'Average_18_OpticalFlowAfterFilter_Hilbert_masked')







#%%