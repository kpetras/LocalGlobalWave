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

# %%
folder = "<folder_path>"
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
# Create a list of WaveData files in the folder
fileName = "MEG_Grad_WaveData"
fileList = glob.glob(os.path.join(folder, "**", fileName), recursive=True)
#fileList= fileList[1:2] #[KP] just for testing, remove later!
for sub,filePath in enumerate(fileList):
    subname = os.path.basename(os.path.dirname(filePath))    
    savefolder = os.path.dirname(filePath)
    print("Processing subject: " + str(sub))
    print("Processing file: " + filePath)
    # the wavedata objects
    waveData = ImportHelpers.load_wavedata_object(filePath)

    #remove "fov" trials
    trialInfo = waveData.get_trialInfo()
    fovInds = [i for i, trial in enumerate(trialInfo) if 'fov' in trial]
    waveData.prune_trials(fovInds)
    #%_______remove! Just for debug________________
    #waveData.DataBuckets['Grad'].set_data(waveData.DataBuckets['Grad'].get_data()[5:50,:,:],'trl_chan_time')
   
    #set up a bunch of stuff in the first iteration
    time_vect = waveData.get_time()
    windowLength = 1 #seconds
    windowsize = int(np.floor((windowLength*waveData.get_sample_rate()))) #in samples
    fftfreqs = np.fft.fftfreq(windowsize, d=time_vect[1]-time_vect[0])#we use 1 second segments 
    fft_freqEdges = hf.bin_edges_from_centers(fftfreqs)
    #we want the first bin to be centered on 1 and the last on 40 Hz
    freqMin = 1 - ((fftfreqs[1]-fftfreqs[0])/2)
    freqMax = 40 + ((fftfreqs[1]-fftfreqs[0])/2)
    nbins = hf.find_nearest(fft_freqEdges, freqMax)[0] - hf.find_nearest(fft_freqEdges, freqMin)[0]
    nChans = waveData.get_data('Grad').shape[1]    
    evoked_dict = {}

    #%FFT spectrum 
    chan= 94
    chan = 66
    fft_pre= np.zeros((nChans,windowsize),dtype=complex)
    x = waveData.get_data('Grad')[:, :, 0:250]
    fft_pre[:,:] = np.mean(np.abs(np.fft.fft(x)/x.shape[-1]) ,0)
    fft_pre_chan_AVG = np.mean(fft_pre, axis=0)
    fft_pre_singleChan = fft_pre[chan,:]
    
    fft_post= np.zeros((nChans,windowsize),dtype=complex)
    x = waveData.get_data('Grad')[:, :, 500:750]#last second of stimulus
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
    time_vect = waveData.get_time()
    windowLength = 1 #seconds
    f_sample = waveData.get_sample_rate()
    #initialize arrays to store the spectra
    from scipy.signal import welch
    dataPreStim = waveData.get_data('Grad')[:, :, 0:250]
    dataPostStim = waveData.get_data('Grad')[:, :, 500:750]

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
    # Extract any alpha band peaks from the power spectrum model
    alpha = get_band_peak_fm(fm, bands.alpha)
    print(alpha)
    if np.any(np.isnan(alpha)):
        alpha = [10]
    if np.any(np.isnan(theta)):
        theta = 5
        
    for freqInd in range(2):
        if freqInd == 0:
            freq = 5 #theta[0] we know the freq of interest here
        else:
            freq = alpha[0]
         #% do filter + Hilbert to get complex Timeseries 
        filt.filter_narrowband(waveData, dataBucketName = "Grad", LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
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
                                                            num_points=1000)
    
    SensorLayout.distance_along_surface(waveData, Surface, tolerance=0.1, get_extent = chanInds, plotting= True)
    SensorLayout.distmat_to_2d_coordinates_Isomap(waveData) #can also use MDS here
    
    #plot topo
    timeInds = 300
    dataInds = (0, slice(None), slice(None))
    plotting.plot_topomap(waveData,dataBucketName = "NBFiltered_GradY", dataInds = dataInds, timeInds= timeInds, trlInd = 0)

    grid_x, grid_y,mask =SensorLayout.interpolate_pos_to_grid(
        waveData, 
        dataBucketName = "NBFiltered_GradX",
        numGridBins=18,        
        return_mask = True,
        mask_stretching = True)
    grid_x, grid_y, mask =SensorLayout.interpolate_pos_to_grid(
        waveData, 
        dataBucketName = "NBFiltered_GradY",
        numGridBins=18,
        return_mask = True,
        mask_stretching = True)

    # make new distMat based on the interpolated grid
    positions = np.dstack((grid_x, grid_y)).reshape(-1, 2)
    distMat = SensorLayout.regularGrid(waveData, positions)
    original_data_bucket = "NBFiltered_GradY"
    interpolated_data_bucket = "NBFiltered_GradYInterpolated"

    OrigInd  = (0,0,slice(None),300)
    InterpInd =(0,0,slice(None),slice(None),300)
    fig = plotting.plot_interpolated_data(waveData, original_data_bucket, interpolated_data_bucket,
                                            grid_x, grid_y, OrigInd, InterpInd,  type='')
    # # get complex timeseries
    hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered_GradXInterpolated")
    waveData.DataBuckets["AnalyticSignalX"] =  waveData.DataBuckets.pop("AnalyticSignal")
    SensorLayout.apply_mask(waveData, mask, dataBucketName = 'AnalyticSignalX', overwrite = True, storeMask = True)
    tStart = time.time()
    print("OpticalFlow started")
    OpticalFlow.create_uv(waveData, 
            applyGaussianBlur=False, 
            dataBucketName = 'AnalyticSignalX',
            type = "angle", 
            Sigma=1, 
            alpha = 0.1, 
            nIter = 200, 
            is_phase = False)
    trialToPlot = 5
    waveData.DataBuckets["UV_Angle_GradX"] =  waveData.DataBuckets.pop("UV")
    print('first optical flow took: ', time.time()-tStart)


    hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered_GradYInterpolated")
    waveData.DataBuckets["AnalyticSignalY"] =  waveData.DataBuckets.pop("AnalyticSignal")
    SensorLayout.apply_mask(waveData, mask, dataBucketName = 'AnalyticSignalY', overwrite = True, storeMask = True)
    tStart = time.time()
    OpticalFlow.create_uv(waveData, 
            applyGaussianBlur=False,
            dataBucketName = 'AnalyticSignalY', 
            type = "angle", 
            Sigma=1, 
            alpha = 0.1, 
            nIter = 100, 
            is_phase = False)
    trialToPlot = 5
    waveData.DataBuckets["UV_Angle_GradY"] =  waveData.DataBuckets.pop("UV")
    print("second OpticalFlow took: ", time.time()-tStart)
    waveData.set_active_dataBucket('AnalyticSignalY')
    # ani = plotting.plot_optical_flow(waveData, 
    #                                 UVBucketName = 'UV_Angle_GradY',
    #                                 PlottingDataBucketName = 'AnalyticSignalY', 
    #                                 dataInds = (0,trialToPlot, slice(None), slice(None), slice(None)),
    #                                 plotangle=True,
    #                                 normVectorLength = True)  
    # ani.save('Grad_OpticalFlowAfterFilter_Hilbert.gif')

    waveData.save_to_file(savefolder + '/Grad_18_OpticalFlowAfterFilter_Hilbert_masked')


    

   

# %%
