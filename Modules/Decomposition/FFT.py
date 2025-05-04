#%%
from Modules.Utils import WaveData as wd
from Modules.Utils import HelperFuns as hf
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import hann
#%%
def hann_fft(waveData, dataBucketName = "", timeStart = [], timeEnd = [], timeStep = 1, freqStart = 0, freqEnd = -1):
    """
    Computes the fft of the data in the active data bucket.
    The fft is computed along the time dimension of the data array.
    """
    #set defaults
    if timeStart == []:
        timeStart = waveData.get_time()[0]
    if timeEnd == []:
        timeEnd = waveData.get_time()[-1]

    # ensure proper bookkeeping of data dimensions
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , "trl_chan_time")
    
    # Select time range (input is in seconds)
    timeStart = int(timeStart * waveData.get_sample_rate())
    timeEnd = int(timeEnd * waveData.get_sample_rate())
    
    currentData = currentData[..., timeStart:timeEnd:timeStep]
    
    # Compute FFT
    nSamples = currentData.shape[-1]
    hannWindow = hann(nSamples)
    #freq resolution
    freqStep = waveData.get_sample_rate() / nSamples
    #freq range
    if freqEnd == -1:
        freqEnd = waveData.get_sample_rate() / 2
    freqStart = int(freqStart / freqStep)
    freqEnd = int(freqEnd / freqStep)
    freqs = fftfreq(nSamples, 1/waveData.get_sample_rate())[freqStart:freqEnd]
    fft_result = fft(currentData * hannWindow, axis=-1)[:, :, freqStart:freqEnd]
    #normalize
    fft_result = fft_result / nSamples
      

    
    if hasBeenReshaped:
        fft_result = np.reshape(fft_result, (freqs.shape(),*origShape))

    complexDataBucket = wd.DataBucket(fft_result, "FFT", origDimord.replace('time', 'freq'),
                                        waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(complexDataBucket)
    waveData.log_history(["Frequency Decomposition", "FFT", "Time range: " + str(timeStart) + " to " + str(timeEnd) + " in steps of " + str(timeStep), "Frequency range: " + str(freqStart) + " to " + str(freqEnd) + " in steps of " + str(freqStep)])
