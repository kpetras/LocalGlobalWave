import Modules.Utils.HelperFuns as hf
import Modules.Utils.WaveData as wd
from scipy.signal import hilbert
import numpy as np

def apply_hilbert(waveData, dataBucketName =None ):
    """
    Computes the hilbert transform of the data in the active data bucket.
    The hilbert transform is computed along the time dimension of the data array.
    """
    # ensure proper bookkeeping of data dimensions
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    #check if any filtereing has been done before, if so, warn that it will be overwritten
    if not (dataBucketName == "NBFiltered"):
        print("Hilbert Transform to get analytic signal: Make sure data has been filtered to a sufficiently narrow frequency bandwith before applying hilbert")

    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , "trl_chan_time")
    # Compute Hilbert transform
    analytic_signal = hilbert(currentData)
    inst_amplitude = np.abs(analytic_signal)
    inst_phase  = np.unwrap(np.angle(analytic_signal))
    ComplexPhaseData = inst_amplitude * np.exp(1j * inst_phase)
    if hasBeenReshaped:
        ComplexPhaseData = np.reshape(ComplexPhaseData, origShape)

    ComplexPhaseDataBucket = wd.DataBucket(ComplexPhaseData, "AnalyticSignal", origDimord, waveData.get_channel_names())
    waveData.add_data_bucket(ComplexPhaseDataBucket)
    waveData.log_history(["Analytic Signal", "Hilbert"])




