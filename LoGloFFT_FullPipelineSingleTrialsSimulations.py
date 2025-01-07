#%%
import copy
from Modules.Utils import WaveData as wd, HelperFuns as hf
from Modules.Utils import ImportHelpers
from Modules.Simulation import SimulationFuns
from Modules.PlottingHelpers import Plotting
from Modules.SpatialArrangement import SensorLayout
from Modules.Decomposition import Hilbert as hilb
from Modules.WaveAnalysis import OpticalFlow
from Modules.Preprocessing import Filter as filt

from mne import report 
import numpy as np
from random import choices
import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy.ndimage
from fooof import FOOOF, FOOOFGroup
from fooof.plts.spectra import plot_spectrum
from fooof.sim.gen import gen_aperiodic
from fooof.objs import fit_fooof_3d, combine_fooofs
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fm, get_band_peak_fg

#%%Simulate some data 
# Requirements Simulated Data:
# - 3 seconds 
# - onset at 1000 ms
# - 5 HZ wave that stays until end of trial but switches direction at 2000ms 
# - Local oscillators are present during the entirity at 10 Hz (Synchonised or random)
# - Also some pink noise
folder = 'pathToStoreYourData/'
figfolder =  'pathToWhereYouWantYourFigures'
report = report.Report()

for sub in range (19):
    SpatialFrequency = .6
    TemporalFrequency  = 5
    SampleRate = 250
    WaveDuration = 50 #in cycles, anything longer than the sim duration just makes that there is no offset
    MatrixSize = 42
    #Construct Low SNR data
    onsetInMs = [1000, 1000, 1000]
    waveOnset = [(onset / (1000/SampleRate) ) / (1 / TemporalFrequency  * SampleRate) for onset in onsetInMs]
    FirstHalfData = SimulationFuns.simulate_signal(
        Type =  "PlaneWave", 
        ntrials = 3, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate, 
        SimDuration= 2, 
        #SimOptions from here on
        TemporalFrequency = TemporalFrequency,
        SpatialFrequency = SpatialFrequency,
        WaveDirection = [90,270,180],
        WaveOnset = waveOnset,
        WaveDuration = WaveDuration,
    )

    onsetInMs = [100, 100, 100]
    waveOnset = [  (onset / (1000/SampleRate) ) / (1 / TemporalFrequency  * SampleRate) for onset in onsetInMs]
    SecondHalfData = SimulationFuns.simulate_signal(
        Type =  "PlaneWave" , 
        ntrials = 3, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate, 
        SimDuration= 1, 
        #SimOptions from here on
        TemporalFrequency = TemporalFrequency,
        SpatialFrequency = SpatialFrequency,
        WaveDirection = [270,90,0],
        WaveOnset = waveOnset,
        WaveDuration = WaveDuration,
    )
    #%combine the two halves
    LoSNRWaveData = SimulationFuns.combine_SimData([FirstHalfData,SecondHalfData], dimension = 'time', SimCondList = None)
    #repeat the trials 2 times each
    LoSNRWaveData.DataBuckets["SimulatedData"]._data = np.repeat(LoSNRWaveData.DataBuckets["SimulatedData"]._data, 2, axis=0)
    LoSNRWaveData.DataBuckets["Mask"]._data = np.repeat(LoSNRWaveData.DataBuckets["Mask"]._data, 2, axis=0)

    #create Copy for High SNR
    HiSNRWaveData = copy.deepcopy(LoSNRWaveData)
    #
    #Create Oscillators
    localOscillatorsRandom = SimulationFuns.simulate_signal(
        Type="LocalOscillation", 
        ntrials = 1, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate, 
        SimDuration= 3, 
        OscillatoryPhase = "Random",
        TemporalFrequency = 10,
        OscillatorProportion = 0.2
    )

    localOscillatorsSynched = SimulationFuns.simulate_signal(
        Type="LocalOscillation", 
        ntrials = 1, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate, 
        SimDuration= 3, 
        OscillatoryPhase = "Synchronized",
        TemporalFrequency = 10,
        OscillatorProportion = 0.2 
    )

    #repeat the trials 5 times each
    localOscillatorsRandom.DataBuckets["SimulatedData"]._data = np.repeat(localOscillatorsRandom.DataBuckets["SimulatedData"]._data, 2, axis=0)
    localOscillatorsRandom.DataBuckets["Mask"]._data = np.repeat(localOscillatorsRandom.DataBuckets["Mask"]._data, 2, axis=0)
    localOscillatorsSynched.DataBuckets["SimulatedData"]._data = np.repeat(localOscillatorsSynched.DataBuckets["SimulatedData"]._data, 2, axis=0)
    localOscillatorsSynched.DataBuckets["Mask"]._data = np.repeat(localOscillatorsSynched.DataBuckets["Mask"]._data, 2, axis=0)
    #Overwrite data with oscillators
    def oscillatorSNRMIX(signal, oscillator, mask, SNR):
        mask = 1 - mask
        SNRmask = mask * SNR
        signal[mask>0] = (oscillator[mask>0] + (signal[mask>0] * SNRmask[mask>0])) / (1 + SNRmask[mask>0])
        return signal

    #Set SNR for mixing with oscillators 
    oscSNR = 1
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[0] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[0], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[0],localOscillatorsSynched.get_data("Mask")[0],oscSNR )
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[1] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[1], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[0],localOscillatorsRandom.get_data("Mask")[0],oscSNR )
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[2] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[2], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[1],localOscillatorsSynched.get_data("Mask")[1],oscSNR )
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[3] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[3], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[1],localOscillatorsRandom.get_data("Mask")[1],oscSNR )
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[4] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[4], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[0],localOscillatorsSynched.get_data("Mask")[0],oscSNR )
    LoSNRWaveData.DataBuckets["SimulatedData"]._data[5] = oscillatorSNRMIX(LoSNRWaveData.DataBuckets["SimulatedData"]._data[5], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[0],localOscillatorsRandom.get_data("Mask")[0],oscSNR )

    HiSNRWaveData.DataBuckets["SimulatedData"]._data[0] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[0], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[0],localOscillatorsSynched.get_data("Mask")[0],oscSNR )
    HiSNRWaveData.DataBuckets["SimulatedData"]._data[1] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[1], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[0],localOscillatorsRandom.get_data("Mask")[0],oscSNR )
    HiSNRWaveData.DataBuckets["SimulatedData"]._data[2] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[2], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[1],localOscillatorsSynched.get_data("Mask")[1],oscSNR )
    HiSNRWaveData.DataBuckets["SimulatedData"]._data[3] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[3], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[1],localOscillatorsRandom.get_data("Mask")[1],oscSNR )
    HiSNRWaveData.DataBuckets["SimulatedData"]._data[4] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[4], 
                                                                        localOscillatorsSynched.get_data("SimulatedData")[0],localOscillatorsSynched.get_data("Mask")[0],oscSNR )
    HiSNRWaveData.DataBuckets["SimulatedData"]._data[5] = oscillatorSNRMIX(HiSNRWaveData.DataBuckets["SimulatedData"]._data[5], 
                                                                        localOscillatorsRandom.get_data("SimulatedData")[0],localOscillatorsRandom.get_data("Mask")[0],oscSNR )

    #Create Noise
    Noise = SimulationFuns.simulate_signal(
        Type="SpatialPinkNoise", 
        ntrials = 6, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate,
        SimDuration= 3)
    SNR = 0.8
    LoSNRWaveData = SimulationFuns.SNRMix(LoSNRWaveData, Noise, SNR)
    LoSNRWaveData._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": MatrixSize}, {"sampleRate": SampleRate}]})


    #Create new Noise
    Noise = SimulationFuns.simulate_signal(
        Type="SpatialPinkNoise", 
        ntrials = 6, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate,
        SimDuration= 3)
    SNR = 1.8
    HiSNRWaveData = SimulationFuns.SNRMix(HiSNRWaveData, Noise, SNR)
    HiSNRWaveData._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": MatrixSize}, {"sampleRate": SampleRate}]})

    #combine the whole thing into a single waveData object
    waveData = SimulationFuns.combine_SimData([HiSNRWaveData,LoSNRWaveData], dimension = 'trl', SimCondList = None)


    hf.squareSpatialPositions(waveData)#make into regular grid
    #Plot before sampling and interpolation (high SNR)
    # for trial in range(8):
    #     # Generate the plot for the current trial
    #     ani = Plotting.animate_grid_data(waveData, DataBucketName="SimulatedData", dataInd=0, probepositions=[(0,15), (5,15), (10,15), (15,15), (20,15), (25,15)])
    #     # Save the plot to a file
    #     plot_file = figfolder + f'trial_{trial}.gif'
    #     ani.save(plot_file)
    #     # Add the plot to the report with the corresponding SimInfo as the caption
    #     report.add_image(plot_file, plot_file, tags='RawData' + str(trial))
    #     # Convert the SimInfo of the current trial to a string and format it as HTML
    #     sim_info_html = "<p>{}</p>".format(waveData.get_SimInfo()[trial])
    #     report.add_html(title="SimInfo", tags='RawData' + str(trial), html=sim_info_html)

    gridshape = waveData.get_data('SimulatedData').shape[1:3]
    # pick some channels and assign a spatial layout to them (this makes no sense at all for real data and
    #is only to demonstrate the interpolation to a reagular grid from 3d positions)
    chanpos = np.load('exampleChanpos.npy')
    waveData.set_channel_positions(chanpos)

    x_normalized = (chanpos[:, 0] - np.min(chanpos[:, 0])) / (np.max(chanpos[:, 0]) - np.min(chanpos[:, 0]))
    y_normalized = (chanpos[:, 1] - np.min(chanpos[:, 1])) / (np.max(chanpos[:, 1]) - np.min(chanpos[:, 1]))
    z_normalized = (chanpos[:, 2] - np.min(chanpos[:, 2])) / (np.max(chanpos[:, 2]) - np.min(chanpos[:, 2]))

    x_indices = np.round(x_normalized * (gridshape[0] - 1)).astype(int)
    y_indices = np.round(y_normalized * (gridshape[1] - 1)).astype(int)
    z_indices = np.round(z_normalized * (gridshape[1] - 1)).astype(int)

    original_data = waveData.get_data('SimulatedData')
    grid_data = np.repeat(original_data[:, :, :, np.newaxis, :], gridshape[1], axis=3)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_indices, y_indices, z_indices)
    # X, Y, Z = np.meshgrid(np.arange(gridshape[0]), np.arange(gridshape[1]), np.arange(gridshape[1]))
    # ax.scatter(X, Y, Z, alpha=0.1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    newdata = grid_data[:, y_indices, x_indices, z_indices, :]
    dataBucket = wd.DataBucket(newdata, "EEGLayout",'trl_chan_time', [] )
    waveData.add_data_bucket(dataBucket) 
    print(waveData)

    time_vect = waveData.get_time()
    windowLength = 1 #seconds
    windowsize = int(np.floor((windowLength*waveData.get_sample_rate()))) #in samples
    fftfreqs = np.fft.fftfreq(windowsize, d=time_vect[1]-time_vect[0])#we use 1 second segments 
    fft_freqEdges = hf.bin_edges_from_centers(fftfreqs)
    #we want the first bin to be centered on 1 and the last on 40 Hz
    freqMin = 1 - ((fftfreqs[1]-fftfreqs[0])/2)
    freqMax = 40 + ((fftfreqs[1]-fftfreqs[0])/2)
    nbins = hf.find_nearest(fft_freqEdges, freqMax)[0] - hf.find_nearest(fft_freqEdges, freqMin)[0]
    nChans = waveData.get_data('EEGLayout').shape[1]    
    evoked_dict = {}

    #find individual alpha from the pre-stim peak
    time_vect = waveData.get_time()
    windowLength = 1 #seconds
    f_sample = waveData.get_sample_rate()
    #initialize arrays to store the spectra
    pre_stim_start = int(0 * f_sample)  # start at 0 seconds
    pre_stim_end = int(1 * f_sample)  # end at 1 second
    post_stim_start = int(2 * f_sample)  # start at 2 seconds
    post_stim_end = int(3 * f_sample)  # end at 3 seconds

    # Extract the data for the desired time ranges
    dataPreStim = waveData.get_data('EEGLayout')[:, :, pre_stim_start:pre_stim_end]
    dataPostStim = waveData.get_data('EEGLayout')[:, :, post_stim_start:post_stim_end]

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
    #PreStim
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [1, 40]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm.report(freqs, avgSpecPreStim, freq_range)
    fm.fit(freqs, avgSpecPreStim, [3, 30])

    bands = Bands({'theta' : [4, 8],'alpha' : [8, 12]})
    # Extract any alpha band peaks from the power spectrum model
    theta = get_band_peak_fm(fm, bands.theta)
    alpha = get_band_peak_fm(fm, bands.alpha)
    print(theta)
    print(alpha)
    # plt.savefig(figfolder + 'FOOOFpreStim.png')
    # # Add the figure to the report
    # report.add_image(figfolder + 'FOOOFpreStim.png', 'FOOOFpreStim.png', caption='PreStim' , tags='FOOOF')

    #PostStim
    fm = FOOOF()
    # Set the frequency range to fit the model
    freq_range = [1, 40]
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm.report(freqs, avgSpecPostStim, freq_range)
    fm.fit(freqs, avgSpecPostStim, [3, 30])

    bands = Bands({'theta' : [4, 8],'alpha' : [8, 12]})
    # Extract any alpha band peaks from the power spectrum model
    theta = get_band_peak_fm(fm, bands.theta)
    alpha = get_band_peak_fm(fm, bands.alpha)
    print(theta)
    print(alpha)
    # plt.savefig(figfolder + 'FOOOFpostStim.png')
    # # Add the figure to the report
    # report.add_image(figfolder + 'FOOOFpostStim.png', 'FOOOFpostStim.png', caption='PostStim' , tags='FOOOF')

    for freqInd in range(2):
        if freqInd == 0:
            freq = 5 #theta[0] we know the freq of interest here
        else:
            freq = alpha[0]
            #% do filter + Hilbert to get complex Timeseries 
        filt.filter_narrowband(waveData, dataBucketName = "EEGLayout", LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
        waveData.DataBuckets[str(freqInd)] =  waveData.DataBuckets.pop("NBFiltered")
    temp = np.stack((waveData.DataBuckets["0"].get_data(), waveData.DataBuckets["1"].get_data()),axis=0)
    waveData.add_data_bucket(wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names()))

    # get complex timeseries
    hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered")
    # plot some timerseries to check
    chan = 10
    freqs = ['theta', 'alpha']
    # Get the number of trials
    num_trials = waveData.get_data("EEGLayout").shape[0]

    # # Loop over the trials
    # for trl in range(num_trials):
    #     # Loop over the frequencies
    #     for freq in range(len(freqs)):
    #         # Create a new figure
    #         fig, axs = plt.subplots(2, figsize=(12, 6))

    #         # Subplot 1: raw data, filtered data, envelope
    #         axs[0].plot(waveData.get_time(), waveData.get_data("EEGLayout")[trl,chan,:], label='Raw data')
    #         axs[0].plot(waveData.get_time(), waveData.get_data("NBFiltered")[freq,trl,chan,:], label=f'{freqs[freq]} Narrow Band filtered')
    #         axs[0].plot(waveData.get_time(), np.abs(waveData.get_data("AnalyticSignal")[freq,trl,chan,:]), label='Envelope')
    #         axs[0].set_title(f'Trial {trl+1}: Raw data, Filtered data, and Envelope')
    #         axs[0].set_xlabel('Time (s)')
    #         axs[0].set_ylabel('Amplitude')
    #         axs[0].legend()

    #         # Subplot 2: instantaneous phase
    #         axs[1].plot(waveData.get_time(), np.angle(waveData.get_data("AnalyticSignal")[freq,trl,chan,:]))
    #         axs[1].set_title(f'Trial {trl+1}: Instantaneous Phase')
    #         axs[1].set_xlabel('Time (s)')
    #         axs[1].set_ylabel('Phase (radians)')

    #         # Adjust the layout and show the plot
    #         plt.tight_layout()
    #         report.add_figure(fig=fig,  title=f'Trial {trl+1}: Filter + Hilbert for {freqs[freq]}', tags='FrequencyDecomposition')
    #         plt.show()

    #
    waveData.set_active_dataBucket('AnalyticSignal')
    chanInds=True
    Surface, PolySurface = SensorLayout.create_surface_from_points(waveData,
                                                            type='channels',
                                                            num_points=1000)

    SensorLayout.distance_along_surface(waveData, Surface, tolerance=0.1,get_extent = chanInds, plotting= True)
    SensorLayout.distmat_to_2d_coordinates_Isomap(waveData) #can also use MDS here
    grid_x, grid_y, mask =SensorLayout.interpolate_pos_to_grid(
        waveData, 
        numGridBins=18,
        return_mask=True, 
        mask_stretching=True
        )

    # make new distMat based on the interpolated grid
    positions = np.dstack((grid_x, grid_y)).reshape(-1, 2)
    distMat = SensorLayout.regularGrid(waveData, positions)
    print(waveData)

    # ani = Plotting.animate_grid_data(waveData,DataBucketName = "AnalyticSignalInterpolated", dataInd = (0,0), probepositions=[(0,15), (5,15), (10,15), (15,15), (20,15), (25,15)])
    # ani.save(figfolder + 'test.gif')
    # report.add_image(figfolder +  'test.gif', 'test.gif', caption='theta Interpolated' , tags='Interpolated')
    # ani = Plotting.animate_grid_data(waveData,DataBucketName = "AnalyticSignalInterpolated", dataInd = (1,0), probepositions=[(0,15), (5,15), (10,15), (15,15), (20,15), (25,15)])
    # ani.save(figfolder + 'test2.gif')
    # report.add_image(figfolder +  'test2.gif', 'test2.gif', caption='alpha Interpolated' , tags='Interpolated')
    #
    SensorLayout.apply_mask(waveData, mask, dataBucketName = 'AnalyticSignalInterpolated', overwrite = False, storeMask=True)
    # 
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(chanpos[:, 0], chanpos[:, 1],chanpos[:, 2], c=newdata[0, :, 50], cmap='viridis')
    # plt.colorbar(sc)
    # plt.show()
    # #plot movie
    # for trl in range(12):
    #     ani = Plotting.animate_grid_data(waveData, DataBucketName="AnalyticSignalInterpolated", dataInd=(0,trl), probepositions=[(0,7), (2,7), (4,7), (6,7), (8,7), (10,7),(12,7), (14,7)])
    #     plot_file = figfolder + f'trial_{trl}.gif'
    #     ani.save(plot_file)
    #     report.add_image(plot_file, plot_file, tags='Interpolated' + str(trl))



    # optical flow (here done on amplitude. Use you favorite method for getting the analytic signal
    # and pass complex values along with angleFlag=True to use phase)
    #then go get some coffee, this still takes a while
    waveData.set_active_dataBucket('AnalyticSignalInterpolatedMasked')
    OpticalFlow.create_uv(waveData, 
            applyGaussianBlur=False, 
            type = "angle", 
            Sigma=1, 
            alpha = 0.1, 
            nIter = 200, 
            is_phase = False)

    SensorLayout.apply_mask(waveData, mask, dataBucketName = 'UV', overwrite = True, storeMask=False, maskValue=0.5)

    # for trl in range(9):
    #     ani = Plotting.plot_optical_flow(waveData, 
    #                                     PlottingDataBucketName = 'AnalyticSignalInterpolatedMasked', 
    #                                     dataInds = (0,trl, slice(None), slice(None), slice(None)),
    #                                     plotangle=True,
    #                                     normVectorLength = False) 
    #     gif_filename = figfolder +  'opticalFlow' + str(trl) + '_masked.gif'
    #     ani.save(gif_filename)
    #     report.add_image(gif_filename, gif_filename, caption='Optical Flow'+str(trl) , tags='OpticalFlowAngle')


    waveData.set_trialInfo(['full trav in','full trav in','full trav out','full trav out', 'full stand', 'full stand', 'full trav in','full trav in','full trav out','full trav out', 'full stand', 'full stand'])
    waveData.set_time(np.linspace(-1, 2, 750)[:-1])
    waveData.save_to_file(folder + 'Simulations/sub' + str(sub) + "_Filter_Hilbert_OpticalFlow")
#%%
