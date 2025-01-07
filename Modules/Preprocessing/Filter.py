import matplotlib.pyplot as plt
import mne
from Modules.Utils import WaveData as wd
from Modules.Utils import HelperFuns as hf
from scipy.signal import butter, filtfilt
from scipy.signal import detrend
from scipy.signal import impulse
import numpy as np

def filter_broadband(data, LowCutOff, HighCutOff,  n_jobs=5):
    '''MNE non-causal filter'''
    shape = data.get_active_data().shape
    new_shape = (-1,) + shape[-1:]
    old_dimord = data.DataBuckets[data.ActiveDataBucket].get_dimord()
    data.DataBuckets[data.ActiveDataBucket].reshape(new_shape, data.DataBuckets[data.ActiveDataBucket].get_dimord())  
    NewData = mne.filter.filter_data(data = data.get_active_data(),sfreq = data.get_sample_rate(),l_freq = LowCutOff, h_freq= HighCutOff, n_jobs=n_jobs)
    dataBucket = wd.DataBucket(NewData, "BBFiltered", data.DataBuckets[data.ActiveDataBucket].get_dimord())
    # reshape original data
    data.DataBuckets[data.ActiveDataBucket].reshape(shape, old_dimord)  
    data.add_data_bucket(dataBucket)  
    # reshape last bucket
    data.DataBuckets[data.ActiveDataBucket].reshape(shape, old_dimord)   
    data.log_history(["Broadband Filter", "filt",LowCutOff, HighCutOff])
    return data

def filter_notch(data,  LineNoiseFreq, n_jobs=5):
    '''MNE non-causal filter'''
    shape = data.get_active_data().shape
    new_shape = (-1,) + shape[-1:]
    old_dimord = data.DataBuckets[data.ActiveDataBucket].get_dimord()
    data.DataBuckets[data.ActiveDataBucket].reshape(new_shape, data.DataBuckets[data.ActiveDataBucket].get_dimord())
    NewData = mne.filter.notch_filter(x =data.get_active_data(), Fs=data.get_sample_rate(),freqs = LineNoiseFreq, 
        filter_length = 'auto', n_jobs=n_jobs)
    dataBucket = wd.DataBucket(NewData, "NotchFiltered", data.DataBuckets[data.ActiveDataBucket].get_dimord())
    # reshape original data
    data.DataBuckets[data.ActiveDataBucket].reshape(shape, old_dimord)  
    data.add_data_bucket(dataBucket)
    # reshape last bucket
    data.DataBuckets[data.ActiveDataBucket].reshape(shape, old_dimord)   
    data.log_history(["Notch Filter", "notch", LineNoiseFreq])
    return data


from scipy.signal import firwin

def bandpass(lowcut, highcut, fs, type="IIR", order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if type == "IIR":
        b, a = butter(order, [low, high], btype='band')
        # Calculate the impulse response
        t, h = impulse((b, a))
    elif type == "FIR":
        print("CAUTION!!! Make sure your filter order is correct!\n"
            "For FIR filters, a reasonable order is about 20 times\n"
            "of what you would use for an IIR filter.\n"
            "Current order is: " + str(order) + "\n" )    
        numtaps = order + 1  # Number of taps in the FIR filter
        b = firwin(numtaps, [low, high], pass_zero=False)
        a = [1.0]  # In an FIR filter, the a coefficients are just [1.0]
        # The length of the impulse response is the number of taps
        h = b
    else:
        raise ValueError("Invalid filter type. Must be either 'IIR' or 'FIR'.")

    # Calculate the length of the impulse response
    impulse_response_length = len(h)
    
    return b, a, impulse_response_length


from scipy.signal import lfilter

def filter_narrowband(waveData, dataBucketName = "", LowCutOff=0, HighCutOff=120, type= "IIR", order=5, causal=True):
    '''Scipy zero-phase bandpass filter. Detrends before narrowband filtering.'''
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

    # Detrend the data
    currentData = detrend(currentData)

    b, a, impulse_response_length = bandpass(LowCutOff, HighCutOff, waveData.get_sample_rate(),type=type, order=order)
    print("CAUTION!!! Impulse response length: " + str(impulse_response_length))
    
    # Apply the filter
    if causal:
        # For a causal filter, use lfilter
        NewData = lfilter(b, a, currentData)
    else:
        # For a non-causal filter, use filtfilt
        NewData = filtfilt(b, a, currentData)

    if hasBeenReshaped:
        NewData = np.reshape(NewData, origShape)

    dataBucket = wd.DataBucket(NewData, "NBFiltered", waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord(), waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)    
    waveData.log_history(["Narrowband Filter", "filt", LowCutOff, HighCutOff, "Type: " + type, "Causal: " + str(causal)])