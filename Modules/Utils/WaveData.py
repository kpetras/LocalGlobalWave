from . import ImportHelpers
import numpy as np
from . import HelperFuns as hf
import re
import pickle

class DataBucket:
    def __init__(self, data, description, dimord, chanNames, unit=""):
        self._data = data
        self._description = description
        self._dimord = dimord
        self._trialInfo = []
        self._chanNames = chanNames
        self._unit = unit
        self._reservedNames = ["time", "chan", "posx", "posy", "trl"]

    def get_channel_names(self):
        return self._chanNames
    
    def get_dimord(self):
        return self._dimord

    def set_dimord(self, dimord):
        self._dimord = dimord

    def get_description(self):
        return self._description

    def set_description(self, description):
        self._description = description
    
    def get_data(self):
        return self._data
    
    def set_data(self, data, dimord):
        assert len(data.shape) == len(dimord.split("_")), "Dimord does not match data dimensions"
        self._dimord = dimord
        self._data = data
        print("Warning: Dangerous move to set data directly buddy, be sure to know what you're doing")
    
    def get_unit(self):
        return self._unit
    
    def set_unit(self, unit):
        self._unit = unit

    def reshape(self, shape, newDimord):
        """Spatial dimensions must be called chan or pos(char) like posx posy etc.."""
        splitDimord = newDimord.split('_')
        assert len(shape) == len(splitDimord), "Dimensions of new shape do not match dimensions of new dimension order"
        self.set_dimord(newDimord)
        self._data = np.reshape(self._data, shape, order="C")
        chanShape =  tuple([shape[i] for i in  [ind for ind, s in enumerate(splitDimord) if s[0:3]== 'pos']])        
        if len(chanShape) > 0 :
            self._chanNames = np.reshape(self._chanNames, chanShape, order="C")

    def assure_consistency():
        return None

class WaveData():
    def __init__(self, chanpos=[], coords2D=[], time = [], sampleRate=0.0):
        """WaveData Generator

        Args:
            filename(str, optional): Full path to datafile. 
            dataSource (str, optional): data source. Options: "MNE", "Simulation"
            chanpos (list, optional): Nx3 Description of channel positions. 
            time (list, optional): Start- and End-time in seconds or a vector of times in seconds
            sampleRate (float, optional): Sampling rate in Hz. 
            dimord (str, optional): Dimension order, string of the format: trl_chan_time; trl_chan_freq_time etc. 
        """
        self.DataBuckets= {}
        self.ActiveDataBucket = ""
        self.HasRegularLayout = False

        self._sampleRate = sampleRate
        self._chanpos = chanpos
        self._simInfo = []
        self._trialInfo = []
        self._history = []
        self._distMat = []
        self._coords2D = []
        self._channames =[]
        
        if len(time)>0:
            if (len(time) < 2):
                raise Exception("Time should contain two or more numbers")
            elif (len(time) == 2):
                self._time = np.arange(time[0]+1/sampleRate, time[-1]+1/sampleRate, 1/sampleRate)
            else:
                self._time = time
        else:
            self._time = time
  

    def __repr__(self):
        out= ""
        for key, dataBucket in self.DataBuckets.items():
            out += "DataBuckets[\"%s\"]| %s | %s \n" % (key , dataBucket.get_dimord(),dataBucket.get_data().shape )
        out += "%s | %s(Hz) \n" % ("Sampling Rate", self._sampleRate)
        if len(self._time>1):
            out += "%s | %s(S) - %s(S) \n" % ("Time", self._time[0], self._time[-1])
        return out  
    
    def append_dataset(self, wavedata, dataBucketName):
        """Appends active bucket of the supplied wavedata-object to databucket with dataBucketName in the current wavedata"""
        data = wavedata.DataBuckets[wavedata.ActiveDataBucket].get_data()
        self.DataBuckets[dataBucketName]._data = np.concatenate([self.DataBuckets[dataBucketName]._data, data], axis=0)
        self._simInfo += wavedata.get_SimInfo()

    def get_data(self, name):
        return self.DataBuckets[name].get_data()

    def get_active_data(self):
        return self.DataBuckets[self.ActiveDataBucket].get_data()

    def add_data_bucket(self, dataBucketName):
        if (self.has_data_bucket(dataBucketName)):
            Warning(f"DataBucket {dataBucketName} already exists, overwriting it")
        name = dataBucketName.get_description()
        self.ActiveDataBucket = name
        self.DataBuckets[name] = (dataBucketName)

    def delete_data_bucket(self, dataBucketName):
        if dataBucketName in self.DataBuckets.keys():
            del self.DataBuckets[dataBucketName]
        else:
            raise NameError("DataBucket does not exist")

    def has_data_bucket(self, bucket_name):
        '''Check if a data bucket with a bucket_name exists in the WaveData object.
            bucket_name : str
            The name of the data bucket to check
            Returns 
            exists : bool
            True if the data bucket exists, False otherwise
        '''
        return bucket_name in self.DataBuckets

    def crop_data(self, *args):
        """_summary_
        Crop Data in time, input: Either: Start (s), Stop (s) OR: BufferDuration(S)
        """
        if len(args) > 1:
            t0,_=hf.find_nearest(self._time, args[0])#Index of start time of interest
            t1,_=hf.find_nearest(self._time, args[1])#Index of end time of interest 
        else:
            nsamples = int(args[0] * self._sampleRate)             
            t0 = 0+nsamples
            t1 = len(self._time)-nsamples
        self._time = self._time[t0:t1]
        dimensions = self.DataBuckets[self.ActiveDataBucket]._dimord.split("_")
        timedim = [ind for ind, item in enumerate(dimensions) if re.search("time", item)]
        self.DataBuckets[self.ActiveDataBucket]._data = self.DataBuckets[self.ActiveDataBucket]._data.take(indices=range(t0,t1), axis = timedim[0])
        self.log_history(["Crop", "Start",t0, "Stop", t1])  
    
    def prune_trials(self, trials_to_remove):
        """Prune trials from the data and trialInfo list.
        Operation is performed on the active data bucket.
        Args:
            trials_to_remove (list): A list of trial indices to remove.
        """
        dimensions = self.DataBuckets[self.ActiveDataBucket]._dimord.split("_")
        trialdim = [ind for ind, item in enumerate(dimensions) if re.search("trl", item)]
        self.DataBuckets[self.ActiveDataBucket]._data = np.delete(self.DataBuckets[self.ActiveDataBucket]._data, trials_to_remove, axis=trialdim[0])
        self._trialInfo = [trial for i, trial in enumerate(self._trialInfo) if i not in trials_to_remove]
        print('Pruned ' + str(len(trials_to_remove)) + ' trials from dataBucket ' + self.ActiveDataBucket)
        print('New data shape: ' + str(self.DataBuckets[self.ActiveDataBucket]._data.shape))

    def log_history(self, log):
        if (not(len(log) >= 2)):
            raise Exception("Input to log requires a list with at least two string items (Full name & shorthand of method)")
        else:
            self._history.append(log)       
    
    def set_channel_positions(self, chanpos):
        if (type(chanpos) == np.ndarray):
            self._chanpos = chanpos
        elif type(chanpos) == str:
            self._chanpos= ImportHelpers.load_channel_positions(chanpos)
        else:
            raise Exception("Incorrect format for channel positions. Supply ND-array or filepath")
        
    def set_time(self, time):
        self._time = time

    def set_channel_names(self, ch_names):
        self._channames = ch_names

    def set_active_dataBucket(self, name):
        self.ActiveDataBucket = name

    def set_sample_rate(self, sampleRate):
        self._sampleRate = sampleRate

    def set_distMat(self, distMat):
        self._distMat = distMat

    def set_2D_coordinates(self, coords):
        self._coords2D = coords

    def set_simInfo(self,simInfo):
        self._simInfo = simInfo

    def set_trialInfo(self, trialInfo):
        self._trialInfo = trialInfo

    def save_to_file(self, filename=""):
        if filename=="":
            filename = "WaveData_" + '_'.join([element[1] for element in self._history])
        
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
    def get_SimInfo(self):
        return self._simInfo

    def get_time(self):
        return self._time

    def get_sample_rate(self):
        return self._sampleRate

    def get_channel_positions(self):
        return np.copy(self._chanpos)

    def get_distMat(self):
        return self._distMat
    
    def get_extentGeodesic(self):
        return self._extentGeodesic

    def get_2d_coordinates(self):
        return self._coords2D

    def get_log_history(self):
        return self._history

    def get_channel_names(self):
        return self._channames

    def get_trialInfo(self):
        return self._trialInfo