#%%
from certifi import where
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as matlib
from scipy.signal import convolve2d
from importlib import reload
from Modules.Utils import WaveData as wd
from Modules.Utils import HelperFuns as hf

def assertCorrectWaveSettings(Type, ntrials, waveSettings):
    # Check if all required variables are set for the type of wave
    if Type == "PlaneWave" or Type == "TargetWave" :
        assert "TemporalFrequency" in waveSettings.keys(), "TemporalFrequency not set"
        assert "SpatialFrequency" in waveSettings.keys(), "SpatialFrequency not set"
        assert "WaveDirection" in waveSettings.keys(), "WaveDirection not set" 
    if Type == "RotatingWave": 
       assert "TemporalFrequency" in waveSettings.keys(), "TemporalFrequency not set"
       assert "WaveDirection" in waveSettings.keys(), "WaveDirection not set" 
    if Type == "LocalOscillation":
        assert "TemporalFrequency" in waveSettings.keys(), "TemporalFrequency not set"
        assert "OscillatoryPhase" in waveSettings.keys(), "OscillatoryPhase not set" 
    # Check if all required variables are set
    if "NonLinearSkew" in waveSettings.keys():
        assert "NonLinearDegree" in waveSettings.keys() and (Type == "PlaneWave") , "NonLinearSkew set without NonLinearDegree or to wrong type of data"
    if "NonLinearDegree" in waveSettings.keys():
        assert "NonLinearSkew" in waveSettings.keys() and (Type == "PlaneWave") , "NonLinearDegree set without NonLinearSkew or to wrong type of data"
       
    # If array or list is supplied, size must be the same as amount of trials
    for item in waveSettings.items():
        if np.ma.isarray(item[1]):
            assert len(item[1])==ntrials, f"Length of supplied array for \"{item[0]}\" must equal amount of trials"

#%%
def simulate_signal(Type, ntrials, MatrixSize, SampleRate, SimDuration, **waveSettings):
    assertCorrectWaveSettings(Type, ntrials, waveSettings)
    #InitializeDataCubes
    fullData = np.zeros((ntrials,MatrixSize,MatrixSize,int(np.floor(SampleRate*SimDuration))))

    simOptions = []
    # Create SimOptions for each Trial
    for trialNr in range(ntrials):
        currentOptions = {}
        for key, values in waveSettings.items():
            if type(values) is tuple:                    
                currentOptions[key] = np.random.uniform(values[0], values[1])
            elif isinstance(values, np.ndarray) or type(values) is list: 
                currentOptions[key] = values[trialNr]
            else:
                currentOptions[key] = values
        simOptions.append(currentOptions)
    isMaskPresent = "WaveOnset" in waveSettings.keys() or "WaveDuration" in waveSettings.keys() or "OscillatorProportion" in waveSettings.keys()
    if isMaskPresent:
        fullMask = np.zeros((ntrials,MatrixSize,MatrixSize,int(np.floor(SampleRate*SimDuration))))
    
    for TrialNr, SimOption in enumerate(simOptions):
        #Add Cube
        if Type == "None":
            fullData[TrialNr,:,:,:] = initialize_data(MatrixSize, SampleRate, SimDuration)
        if Type == "PlaneWave":
            fullData[TrialNr,:,:,:] = create_plane_wave( MatrixSize, SampleRate, SimDuration,SimOption)
            if isMaskPresent:                               
                fullMask[TrialNr,:,:,:] = create_plane_wave_mask(MatrixSize, SampleRate, SimDuration,SimOption)                
        if Type == "TargetWave":
            fullData[TrialNr,:,:,:] = create_target_wave( MatrixSize, SampleRate, SimDuration,SimOption)
        if Type == "RotatingWave":
            fullData[TrialNr,:,:,:] = create_rotating_wave( MatrixSize, SampleRate, SimDuration,SimOption)
        if Type == "LocalOscillation":
            fullData[TrialNr,:,:,:] = create_local_oscillators( MatrixSize, SampleRate, SimDuration,SimOption)
            if isMaskPresent:
                fullMask[TrialNr,:,:,:] = CreateOscillatorMask( MatrixSize, SampleRate, SimDuration, SimOption)
        if Type == "SpatialPinkNoise":
            fullData[TrialNr,:,:,:] = create_pink_noise( MatrixSize, SampleRate, SimDuration)
        if Type == "WhiteNoise":
            fullData[TrialNr,:,:,:] = create_white_noise( MatrixSize, SampleRate, SimDuration)
        if Type == "StationaryPulse":
            fullData[TrialNr,:,:,:] = create_stationary_pulse(MatrixSize, SampleRate, SimDuration, SimOption)
        if Type == "FrequencyGradient":
            fullData[TrialNr,:,:,:] = create_frequency_gradient(MatrixSize, SampleRate, SimDuration, SimOption)
       
    waveData = create_wavedata(fullData, SampleRate, SimDuration, simOptions)  
    if isMaskPresent: 
        if (len(fullMask.shape)==4):
            fullMask = np.reshape(fullMask,(fullMask.shape[0],fullMask.shape[1]*fullMask.shape[2],fullMask.shape[3]), order='C') 
        dataBucket = wd.DataBucket(fullMask,"Mask", waveData.DataBuckets["SimulatedData"].get_dimord(), waveData.get_channel_names())   
        waveData.add_data_bucket(dataBucket)
    return waveData

def initialize_data(MatrixSize, SampleRate,SimDuration):
    return np.zeros((MatrixSize,MatrixSize,int(np.floor(SimDuration * SampleRate))))  

def create_wavedata(data, SampleRate,SimDuration, simOptions, name = "SimulatedData"):
    #flatten channels
    if (len(data.shape)==4):
        data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2],data.shape[3]), order='C') 
    dimord = "trl_chan_time"
    waveData = wd.WaveData(sampleRate=SampleRate,time=(0,SimDuration))
    x_ = np.linspace(0, int(np.sqrt(data.shape[1]-1)), int(np.sqrt(data.shape[1])))
    y_ = np.linspace(0, int(np.sqrt(data.shape[1]-1)), int(np.sqrt(data.shape[1])))
    grid = np.meshgrid(x_, y_ , indexing='xy')
    waveData.HasRegularLayout = True
    chanpos = np.ones([3,data.shape[1]]).T
    chanpos[:,0:2] = (np.vstack(list(map(np.ravel, grid)))).T  
    waveData.log_history("Created SimulatedData")
    waveData.set_simInfo(simOptions)
    waveData.set_channel_positions(chanpos)
    waveData.set_channel_names([str(s) for s in np.arange(len(chanpos))])
    dataBucket = wd.DataBucket(data,name, dimord, waveData.get_channel_names(), unit="AU")
    waveData.add_data_bucket(dataBucket)
    return waveData

def apply_mask(signal, mask):
    return signal * (1-mask)

def create_stationary_pulse(MatrixSize, SampleRate, SimDuration, SimOptions):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    xsize, ysize, npoints = signalCube.shape    
    grid = get_board(MatrixSize)
    centerX = SimOptions["CenterX"]
    centerY = SimOptions["CenterY"]
    test = gaussian2d(x=grid[0], y=grid[1], x0=centerX, y0=centerY, sigma=SimOptions["Sigma"])     
    signalCube = np.repeat(test[:,:,np.newaxis], npoints, axis=2)
    time_vect = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    signalCubeOut = signalCube * np.sin(2 * np.pi * SimOptions["TemporalFrequency"] * time_vect)
    return signalCubeOut 

def gaussian2d(x, y, x0, y0, sigma):
    """Return the value of a 2D Gaussian function at (x, y) with the given center point (x0, y0) and standard deviation sigma."""
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def create_plane_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration) 
    NonLinearDegree = 0
    NonLinearSkew = 0
    if ("NonLinearDegree" in SimOption.keys()):
        NonLinearDegree = SimOption["NonLinearDegree"]
    if ("NonLinearSkew" in SimOption.keys()):
        NonLinearSkew = SimOption["NonLinearSkew"]

    phase_offset = 0 
    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    orientation = np.deg2rad(360-SimOption["WaveDirection"])       

    #Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
        [np.sin(orientation), np.cos(orientation)]]
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[ii,jj] , Y[ii,jj]]).T)                
            M[ii,jj] = tmp_M[0] 
    
    L = 1/SimOption["SpatialFrequency"] * MatrixSize # from cycles per image to shift per gridstep

    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Adapted from:
            #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py       
            
            time_vect = np.linspace(0, SimDuration , int(SimDuration  * SampleRate ))

            factor = np.sqrt(1 - NonLinearDegree**2)

            num = NonLinearDegree * np.sin(NonLinearSkew) / (1 + factor)            
            num = num + np.sin(2 * np.pi * SimOption["TemporalFrequency"]* time_vect - 2*np.pi/L*M[ii,jj])

            denom = 1 - NonLinearDegree * np.cos(2 * np.pi * SimOption["TemporalFrequency"] \
                    * time_vect + NonLinearSkew - 2*np.pi/L*M[ii,jj])

            signalCube[ii,jj,:] = factor * (num / denom)
            #L*M[ii,jj] is the phaseshift over space where M is a linear 
            # gradient over the grid in the direction determined by orientation
            # And L is the stepsize (determined by spatial frequency)
    return signalCube

def create_plane_wave_mask(MatrixSize, SampleRate, SimDuration,SimOption):
    MaskCube = initialize_data(MatrixSize, SampleRate, SimDuration)  
    orientation = np.deg2rad(360-SimOption["WaveDirection"])      
    grid = get_board(MatrixSize)          
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    #Create Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
            [np.sin(orientation), np.cos(orientation)]]
    for timeSample in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[timeSample,jj] , Y[timeSample,jj]]).T)                
            M[timeSample,jj] = tmp_M[0] 
    
    #M = scale(M,(0,1))
    # Create Time-vector
    temporalChange = SimOption["TemporalFrequency"] / SampleRate         
    spatialChange = SimOption["SpatialFrequency"] /MatrixSize
    stepsize = temporalChange * (1 / spatialChange)
    onsetTimeVector = np.arange(np.min(M),np.max(M)+stepsize,stepsize)            
    #time = time
    onsetStartingIndex = int(np.floor((SimOption["WaveOnset"]/ SimOption["TemporalFrequency"] ) * SampleRate))
    offsetStartingIndex = len(onsetTimeVector) +  onsetStartingIndex + \
                            (int(np.floor(SimOption["WaveDuration"]*     \
                            1/SimOption["TemporalFrequency"] * SampleRate)))

    for timeSample in range(int(np.floor(SimDuration * SampleRate))):
        #Mask ON
        if (timeSample < onsetStartingIndex):
            MaskCube[:,:,timeSample]  = 1.0

        #Signal Onset        
        if (timeSample < (len(onsetTimeVector) + onsetStartingIndex)) and (timeSample >= onsetStartingIndex) :            
            MatrixFunction = np.vectorize(lambda a : 0.0 if a <= onsetTimeVector[timeSample-onsetStartingIndex] else 1.0)
            MaskCube[:,:,timeSample]  = MatrixFunction(M)

        # no mask Sustain
        if (timeSample >= len(onsetTimeVector) + onsetStartingIndex) and (timeSample < offsetStartingIndex):
            MaskCube[:,:,timeSample]  = 0.0

        #Offset
        if (timeSample >= offsetStartingIndex) and (timeSample < offsetStartingIndex + len(onsetTimeVector)):            
            MatrixFunction = np.vectorize(lambda a : 1.0\
                            if a <= onsetTimeVector[timeSample - offsetStartingIndex ] else 0.0)
            MaskCube[:,:,timeSample]  = MatrixFunction(M)
        #Signal Off
        if (timeSample >= offsetStartingIndex + len(onsetTimeVector)):
            MaskCube[:,:,timeSample]  = 1.0
    return MaskCube

def create_frequency_gradient(MatrixSize, SampleRate, SimDuration,SimOption):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration) 
    NonLinearDegree = 0
    NonLinearSkew = 0
    if ("NonLinearDegree" in SimOption.keys()):
        NonLinearDegree = SimOption["NonLinearDegree"]
    if ("NonLinearSkew" in SimOption.keys()):
        NonLinearSkew = SimOption["NonLinearSkew"]

    phase_offset = 0 
    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    orientation = np.deg2rad(360-SimOption["WaveDirection"])       

    #Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
        [np.sin(orientation), np.cos(orientation)]]
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[ii,jj] , Y[ii,jj]]).T)                
            M[ii,jj] = tmp_M[0] 
    FrequencyGradient = hf.scale(M, (SimOption["MinTemporalFrequency"], SimOption["MaxTemporalFrequency"]))
    L = MatrixSize # from cycles per image to shift per gridstep

    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Adapted from:
            #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py       
            
            time_vect = np.linspace(0, SimDuration , int(SimDuration  * SampleRate ))

            factor = np.sqrt(1 - NonLinearDegree**2)

            num = NonLinearDegree * np.sin(NonLinearSkew) / (1 + factor)            
            num = num + np.sin(2 * np.pi * FrequencyGradient[ii,jj]* time_vect)

            denom = 1 - NonLinearDegree * np.cos(2 * np.pi * FrequencyGradient[ii,jj] \
                    * time_vect + NonLinearSkew)

            signalCube[ii,jj,:] = factor * (num / denom)

    return signalCube

def create_target_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)  
    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1] 
    D = np.sqrt((X-SimOption["CenterX"]) **2  + (Y-SimOption["CenterY"])**2)

    # wavelength
    L = 1/SimOption["SpatialFrequency"] * MatrixSize

    # direction of wave
    freq_sign = np.sign(SimOption["WaveDirection"])
    #Time vector
    time_vect = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    #Loop through positions
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Radial wave
            signalCube[ii,jj,:] = np.exp(freq_sign * 1j * ( 2 * np.pi * np.abs(SimOption["TemporalFrequency"]) * \
                time_vect -2 * np.pi / L * D[ii,jj]))
    if SimOption["WaveDirection"] < 0:
        return np.flip(np.real(signalCube), axis=2)
    else:
        return np.real(signalCube)

def create_rotating_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    grid = get_board(MatrixSize)    
    X = grid[0] 
    Y = grid[1] 

    [R,TH] = cart2pol(X,Y)
    # direction of wave
    freq_sign = np.sign(SimOption["TemporalFrequency"])
    direction = np.sign(SimOption["WaveDirection"] )
    #Time vector
    time_vect = np.linspace(0, SimDuration , int(SimDuration *SampleRate ))
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            signalCube[ii,jj,:] = np.exp(freq_sign * 1j * (2*np.pi*np.abs(SimOption["TemporalFrequency"]) * 
                time_vect - direction * TH[ii,jj]))
    return signalCube

def create_local_oscillators(MatrixSize, SampleRate, SimDuration,SimOption):
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    time = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    if SimOption["OscillatoryPhase"] == "Random": 
        for ii in range(MatrixSize):
            for jj in range(MatrixSize):
                #adds sine to initial value in fullstatus 
                signal = np.sin(2*np.pi*SimOption["TemporalFrequency"]* time + \
                    np.random.choice(np.arange(0,2*np.pi),1))
                signalCube[ii,jj,:] = signal
    if SimOption["OscillatoryPhase"] == "Synchronized": 
        for ii in range(MatrixSize):
            for jj in range(MatrixSize):
                #adds sine to initial value in fullstatus 
                signal = np.sin(2*np.pi*SimOption["TemporalFrequency"]* time) #add + Phase offset
                signalCube[ii,jj,:] = signal
    return signalCube

def CreateOscillatorMask(MatrixSize, SampleRate, SimDuration, SimOption):
    # selects which cells will be oscillating
    proportionOfOscillators = SimOption["OscillatorProportion"]
    oscillatorIndeces = np.random.choice(MatrixSize * MatrixSize, int(np.floor((MatrixSize**2) * (1-proportionOfOscillators))),replace=False)
    oscillatorIndeces = np.unravel_index(oscillatorIndeces, (MatrixSize, MatrixSize))  
    oscillatorMask = np.zeros((MatrixSize, MatrixSize, int(SimDuration * SampleRate)))
    oscillatorMask[oscillatorIndeces[0], oscillatorIndeces[1], :] = 1
    return oscillatorMask

def create_white_noise( MatrixSize, SampleRate, SimDuration):
    return np.random.randn(MatrixSize,MatrixSize, int(np.floor(SimDuration*SampleRate)))

def create_pink_noise( MatrixSize, SampleRate, SimDuration):
    signalCube = np.random.randn(MatrixSize,MatrixSize, int(np.floor(SimDuration * SampleRate))) 

    for i in range(int(np.floor(SimDuration * SampleRate))):        
        beta = 2
        u = np.concatenate((np.arange(0,(int(np.floor(MatrixSize)/2)+1),1), np.arange(-(int(np.floor(MatrixSize)/2)-1),0,1)))/MatrixSize
        u = matlib.repmat(u,MatrixSize, 1)
        v = u.T
        SF = (u**2 + v**2)**(beta / 2)
        SF[SF==np.inf] = 0
        # phi=(np.reshape(np.arange(0,1,1/(16*16)),[16,16])).T
        #Take timepoint over space
        phi=scale(signalCube[:,:,i],[0,2*np.pi]).T
        FFT_signal = (SF**.5 *(np.cos(2*np.pi*phi)+1j*np.sin(2*np.pi*phi))).T
        FFT_signal = np.fft.fftshift(FFT_signal)
        FFT_signal[0,0] = 0
        FFT_signal = (FFT_signal * np.exp(1j*(phi)))
        FFT_signal = np.real(np.fft.ifft2(FFT_signal))
        status = scale(FFT_signal)
        signalCube[:,:,i] = status
    return signalCube

def SNRMix(SignalWaveData, NoiseWaveData, SNR, Mask=None):
    if Mask is not None and np.any(Mask):
        SNR = SNR * (1-Mask)
    elif "Mask" in SignalWaveData.DataBuckets.keys():
        SNR = SNR * (1-SignalWaveData.DataBuckets["Mask"].get_data())
    
    signal = SignalWaveData.DataBuckets[list(SignalWaveData.DataBuckets.keys())[0]].get_data()
    noise = NoiseWaveData.get_active_data()
    if isinstance(SNR, (float, int)) or SNR.shape == signal.shape:
        signalCube = (noise + (signal * SNR)) / (1 + SNR)
    else:
        signalCube = (noise + (signal * SNR[:,np.newaxis,np.newaxis])) / (1 + SNR)[:,np.newaxis,np.newaxis]

    wavedata = create_wavedata(signalCube, SignalWaveData.get_sample_rate(),SignalWaveData.get_time()[-1],SignalWaveData.get_SimInfo())
    return wavedata

# utility functions

def createVectorField(board):
    # outputs to matplotlib Quiver
    x = board[0]
    y = board[1]
    u = -y/np.sqrt(x**2 + y**2)
    v = -x/np.sqrt(x**2 + y**2)
    return x,y,u,v

def getProbeColor(index, totalProbes):
    cmap = plt.cm.hsv
    return cmap(index/totalProbes) 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_updated_colors(cmap, status):    
    #fp = open('cmap.pkl', 'rb')
    #cmap = pickle.load(fp)
    #fp.close()
    #cmap = AllOptions["ColorMap"]
    status= status.astype(float)
    #values are between -1 and 1 
    # Don't use "scale()" as this will account for max and min values and change
    # colorscales between frames
    status = (status +1)/2
    facecolors = [cmap(value) for value in status.flatten()]
    return facecolors

def scale(x, out_range=(-1, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def get_board(size):
    #make a grid go from  -7 to 8 in two dimensions
    #one matrix x and one matrix y
    xs = np.arange(0, size) - (np.floor(size /2)-1)
    ys = np.arange(0, size) - (np.floor(size /2)-1)
    board = np.meshgrid(xs, ys)
    return board

def abreu2010(f, nonlin_deg, nonlin_phi, sample_rate, seconds):
    #Adapted from:
    #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py
    r"""Simulate a non-linear waveform using equation 7 in [1]_.

    Parameters
    ----------
    f : float
        Fundamental frequency of generated signal
    nonlin_deg : float
        Degree of non-linearity in generated signal
    nonlin_phi : float
        Skew in non-linearity of generated signal
    sample_rate : float
        The sampling frequency of the generated signal
    seconds : float
        The number of seconds of data to generate

    math::
        u(t) = U_wf \frac{ sin(\omega t) + \frac{r sin \phi}{1+\sqrt{1-r^2}} } {1-r cos(\omega t+ \phi)}

    References
    ----------
    [1] Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010).
       Analytical approximate wave form for asymmetric waves. Coastal Engineering,
       57(7), 656-667. https://doi.org/10.1016/j.coastaleng.2010.02.005
    [2] Drake, T. G., & Calantoni, J. (2001). Discrete particle model for
       sheet flow sediment transport in the nearshore. In Journal of Geophysical
       Research: Oceans (Vol. 106, Issue C9, pp. 19859-19868). American
       Geophysical Union (AGU). https://doi.org/10.1029/2000jc000611

    """
    time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

    factor = np.sqrt(1 - nonlin_deg**2)
    num = nonlin_deg * np.sin(nonlin_phi) / (1 + factor)
    num = num + np.sin(2 * np.pi * f * time_vect)

    denom = 1 - nonlin_deg * np.cos(2 * np.pi * f * time_vect + nonlin_phi)

    return factor * (num / denom)

def combine_SimData(SimDataList, dimension = 'trl', SimCondList = None, dataBucketNames = None):
    """combine multiple SimData objects into one    

    Args:
        SimDataList (list): list of SimData objects
        dimension (str, optional): dimension to concatenate along. Options are 'trl' and 'time'. Defaults to 'trl'.
        SimCondList (list, optional): list of condition names. Defaults to None.

    Returns:
        WaveData: WaveData object containing the combined data
    """
    # Check if there are at least two datasets
    assert len(SimDataList) >= 2, "At least two datasets are required"
    sampleRate = SimDataList[0].get_sample_rate()
    time = SimDataList[0].get_time()
    channel_names = SimDataList[0].get_channel_names()
    channel_positions = SimDataList[0].get_channel_positions()
    dimord = SimDataList[0].DataBuckets[SimDataList[0].ActiveDataBucket].get_dimord()
    if dataBucketNames is None:
        dataBucketNames = SimDataList[0].DataBuckets.keys()

    newdata = [None] * len(dataBucketNames)
    # If SimCondList is not provided, use the index in SimDataList as strings
    if SimCondList is None:
        SimCondList = ['Condition_' + str(i) for i in range(len(SimDataList))]
    #check that all datasets have the same dimensionality
    for SimData in SimDataList:
        for name in dataBucketNames:
            assert SimData.DataBuckets[name].get_dimord() == dimord, "Dimension order must be the same for all datasets"
    if dimension == 'trl':
        # Get the sample rate, time, channel names, and channel positions of the first dataset
        # Check if all datasets have the same sample rate, time, channel names, and channel positions
        for SimData in SimDataList[1:]:
            assert SimData.get_sample_rate() == sampleRate, "Sample Rates are not the same"
            assert np.array_equal(SimData.get_time(), time), "Time is not the same"
            assert SimData.get_channel_names() == channel_names, "Channel names are not the same"
            assert np.array_equal(SimData.get_channel_positions(), channel_positions), "Channel positions are not the same"
        for ind,name in enumerate(dataBucketNames):
            newdata[ind] = np.concatenate([SimData.get_data(name) for SimData in SimDataList], axis=0)
        SimInfo = []
        for SimData, condname in zip(SimDataList, SimCondList):
            sim_info = SimData.get_SimInfo()
            for info in sim_info:
                info['condname'] = condname
            SimInfo += sim_info
    elif dimension == 'time':
        # Check if all datasets have the same sample rate, channel names, and channel positions
        for SimData in SimDataList[1:]:
            assert SimData.get_sample_rate() == sampleRate, "Sample Rates are not the same"
            assert SimData.get_channel_names() == channel_names, "Channel names are not the same"
            assert np.array_equal(SimData.get_channel_positions(), channel_positions), "Channel positions are not the same"
        # Check if all datasets have the same number of trials
        for ind,name in enumerate(dataBucketNames):
            for SimData in SimDataList[1:]:
                assert SimData.get_data(name).shape[0] == SimDataList[0].get_data(name).shape[0], "Number of trials is not the same"
            newdata[ind] = np.concatenate([SimData.get_data(name) for SimData in SimDataList], axis=-1)
        
        # Get all unique keys from the SimInfo objects
        all_keys = set().union(*(SimData.get_SimInfo()[0].keys() for SimData in SimDataList))

        SimInfo = []
        for trial in range(SimDataList[0].get_data(name).shape[0]):
            # Initialize a new dictionary for each trial
            trial_info = {key: [] for key in all_keys}
            for SimData in SimDataList:
                sim_info = SimData.get_SimInfo()[trial]
                # Append the values of the keys in trial_info with the values from the current SimData
                for key in sim_info:
                    trial_info[key].append(sim_info[key])
            # Set 'SwitchTime' to the last time point of the first SimData plus one sample
            trial_info['SwitchTime'] = SimDataList[0].get_time()[-1] + 1/SimDataList[0].get_sample_rate()
            # Set 'condname' to the condition name of the first SimData
            trial_info['condname'] = SimDataList[0].get_SimInfo()[0].get('condname', 'n.a.')
            SimInfo.append(trial_info)
        #update time
        time = np.arange(0+1/sampleRate, (newdata[0].shape[-1]/SimDataList[0].get_sample_rate())+1/sampleRate, 1/sampleRate)
    waveData = wd.WaveData(time=time)
    waveData.set_channel_names(channel_names)
    waveData.set_channel_positions(channel_positions)
    for ind,name in enumerate(dataBucketNames):
        dataBucket = wd.DataBucket(newdata[ind], name,dimord, channel_names)
        waveData.add_data_bucket(dataBucket)
    waveData.set_simInfo(SimInfo)
    waveData.set_sample_rate(sampleRate)
    return waveData
