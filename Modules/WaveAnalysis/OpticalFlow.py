from __future__ import division
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib import path
from scipy.ndimage import convolve as filter2
from scipy.ndimage import gaussian_filter, generic_filter
import numpy as np
from Modules.Utils import WaveData as wd, HelperFuns as hf
import pandas as pd
import itertools
import multiprocessing
from joblib import Parallel, delayed
import os
from scipy.sparse import lil_matrix
from scipy.sparse import bmat
from scipy.sparse import diags
import numba
from functools import partial
try:
    from petsc4py import PETSc    
    use_petsc = True
except ImportError:
    print('petsc4py not found, using scipy.sparse.linalg for optical flow calculation. This could take a while....')
    from scipy.sparse.linalg import gmres
    use_petsc = False


def uv_process_trial(trial_data, nframes , nIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase):
    '''Part of the old optical flow calculation. Not used anymore'''
    UV = np.zeros([trial_data.shape[0], trial_data.shape[1], nframes - 1], dtype=complex)
    for i in range(nframes - 1):
        fn1 = trial_data[:, :, i]
        fn2 = trial_data[:, :, i + 1]
        if applyGaussianBlur:
            fn1 = gaussian_filter(fn1, Sigma)
            fn2 = gaussian_filter(fn2, Sigma)
        [u, v] = HS(fn1, fn2, uInitial, vInitial, alpha, kernel, nIter, is_phase)
        UV[:, :, i] = u + 1j * v
    return UV

def create_uv(waveData, applyGaussianBlur=False, type = "real", Sigma=1, alpha = 2, nIter = 100, is_phase = False, dataBucketName = ''): 
    '''Part of the old optical flow calculation. Not used anymore.
        applyGaussianBlur: if True, apply Gaussian blur to images before calculating optical flow
       Sigma: standard deviation of Gaussian blur kernel
       nIter: number of iterations for Horn-Schunck optical flow calculation
       alpha: smoothness weighting parameter, +alpha => smoother optical flow (typically 0<ALPHA<5) see https://github.com/BrainDynamicsUSYD/NeuroPattToolbox
       BETA: nonlinear penalty parameter. Beta close to zero will be more accurate, but slow (ToDo: implement!)
       type: 'angle', 'abs' or 'real' (default: 'real') which basis to use for optical flow calculation
       !!careful!!: function expects complex data 
       If you already have angles you want to use, choose type='real' and pass the angles as data, 
       but set is_phase=True!!!
    '''
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
        
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "trl_posx_posy_time")

    nObs,posx,posy,nframes = currentData.shape
    if type=='angle' and np.iscomplexobj(currentData):
        currentData = np.angle(currentData)
        is_phase = True
    elif type=='abs' and np.iscomplexobj(currentData):
        currentData = np.abs(currentData)
    elif (type=='real') and np.iscomplexobj(currentData):
        currentData = np.real(currentData)        

    # set up initial velocities
    uInitial = np.zeros([posx, posy])
    vInitial = np.zeros([posx, posy])
    # set up averaging kernel for HS function
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]], float)

    if os.name == 'posix':  # Linux
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        result = pool.starmap(uv_process_trial, [(trial_data, nframes , nIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase) for trial_data in currentData])
        allUV = np.array(result)
        pool.close()
        pool.join()
    else:  # Other OS
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(uv_process_trial)(trial_data, nframes , nIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase) for trial_data in currentData)
        allUV = np.array(result)

    if hasBeenReshaped:
        #reshape back to original dimord, take into account that the last dimension has been reduced by 1
        allUV = np.reshape(allUV, oldshape[:-1] + (oldshape[-1] - 1,))       
    dataBucket = wd.DataBucket(allUV, "UV",currentDimord,
                               waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)

def HS(im1, im2, U,V, alpha, kernel, Niter, is_phase):
    """
    Part of the old optical flow calculation. Not used anymore.
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2, is_phase)

    # Iteration to reduce error
    for _ in range(Niter):
        #  Compute local averages of the flow vectors
        uAvg = filter2(U, kernel)
        vAvg = filter2(V, kernel)
        #  common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        #  iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V

def normalize_angle(p):
    return -np.mod(p + np.pi, 2*np.pi) + np.pi

def phase_convolve2d(frame, kernel):
    '''
    part of the old optical flow calculation. Not used anymore.
    Convolve phase values in frame with kernel.
    function from Cobrawap (RRID:SCR_022966). See https://github.com/INM-6/cobrawap/tree/master
     and cite Gutzen, R., De Bonis, G., De Luca, C., Pastorelli, E., Capone, C., 
     Allegra Mascaro, A. L., Resta, F., Manasanch, A., Pavone, 
     F. S., Sanchez-Vives, M. V., Mattia, M., Grün, S., Paolucci, P. S., & Denker, M. (2022). 
     Comparing apples to apples—Using a modular and adaptable analysis pipeline to compare 
     slow cerebral rhythms across heterogeneous datasets. arXiv:2211.08527. 
     https://doi.org/10.48550/arXiv.2211.08527 '''
    dx, dy = kernel.shape
    dimx, dimy = frame.shape
    dframe = np.zeros_like(frame)
    kernel_center = [int((dim-1)/2) for dim in kernel.shape]

    # inverse kernel to mimic behavior or regular convolution algorithm
    k = kernel[::-1, ::-1]
    ci = dx - 1 - kernel_center[0]
    cj = dy - 1 - kernel_center[1]

    # loop over kernel window for each frame site
    for i,j in zip(*np.where(np.isfinite(frame))):
        phase = frame[i,j]
        dphase = np.zeros((dx,dy), dtype=float)

        for di,dj in itertools.product(range(dx), range(dy)):

            # kernelsite != 0, framesite within borders and != nan
            if k[di,dj] and i+di-ci < dimx and j+dj-cj < dimy \
            and np.isfinite(frame[i+di-ci,j+dj-cj]):
                sign = -1*np.sign(k[di,dj])
                # pos = clockwise from phase to frame[..]
                dphase[di,dj] = sign*normalize_angle(phase-frame[i+di-ci,j+dj-cj])

        if dphase.any():
            dframe[i,j] = np.average(dphase, weights=abs(k)) / np.pi
    return dframe

def computeDerivatives(im1, im2, is_phase):
    '''Part of the old optical flow calculation. Not used anymore'''
    #  build kernels for calculating derivatives
    kernelX = np.array([[-1, 1],
                        [-1, 1]]) * .25  # kernel for computing d/dx
    kernelY = np.array([[-1, -1],
                        [1, 1]]) * .25  # kernel for computing d/dy
    kernelT = np.ones((2, 2)) * .25

    if is_phase:
        # Use phase-based method to compute derivatives
        fx = phase_convolve2d(im1, kernelX) + phase_convolve2d(im2, kernelX)
        fy = phase_convolve2d(im1, kernelY) + phase_convolve2d(im2, kernelY)
        ft = normalize_angle(im2 - im1)
    else:
        fx = filter2(im1, kernelX) + filter2(im2, kernelX)
        fy = filter2(im1, kernelY) + filter2(im2, kernelY)
        ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft

def poincare_index(uv):
    [row, col] = uv.shape

    SinkSource = np.zeros(uv.shape, dtype='double')
    Saddle = np.zeros(uv.shape, dtype='double')
    d = np.angle(uv)

    # Generic filter here is replacing NLFIlter in matlab as used by Afrashteh 2017.
    # Difference being that we pad the input-array by mirroring. And they don't.
    PoincareIdx = generic_filter(d, P_index1, footprint=np.ones((2, 2)))

    for i in range(row):
        for j in range(col):
            if PoincareIdx[i, j] > 0.9:
                SinkSource[i, j] = 1
            elif PoincareIdx[i, j] < -0.9:
                Saddle[i, j] = 1
    return SinkSource, Saddle

def P_index1(D):
    D = np.reshape(D, (2, 2))
    s = np.zeros((4))
    tap = np.zeros((4))

    s[0] = D[1, 0]
    s[1] = D[1, 1]
    s[2] = D[0, 1]
    s[3] = D[0, 0]

    for i in range(4):
        if i == 2:
            tap[i] = s[3] - s[2]
        else:
            tap[i] = s[np.mod(i + 1, 4)] - s[i]

        if abs(tap[i]) < np.pi / 2:
            tap[i] = tap[i]
        elif tap[i] <= -np.pi / 2:
            tap[i] = tap[i] + np.pi
        else:
            tap[i] = tap[i] - np.pi

    return sum(tap) / np.pi

def SourceSinkSaddle(delta, tau):
    if delta < 0:
        return 0, 0
    if delta == 0:
        return np.nan, np.nan
    # delta > 0
    if tau == 0:
        return 2, 1
    if tau > 0:
        type = 1
    else:
        type = -1
    if tau * tau < 4 * delta:
        return type, 1
    return type, 0

def makeContours(u, v, Nmin, Lmin_source, Lmax_sink):
    [uy, ux] = np.gradient(u)
    [vy, vx] = np.gradient(v)

    div1 = ux + vy  # Divergence
    sourceContours = []
    sinkContours = []
    # [C,h] = contour(div1);
    contour = plt.contour(div1)

    # gets all contours from plot output
    segments = contour.allsegs
    # go through all levels (plt.contour, by default splits input in 10 bins between max and min and draws contours respectively)
    # level is at which height a contour is drawn
    for ind, level in enumerate(contour.levels):
        for currentContour in segments[ind]:
            # check that there is data
            if len(currentContour) > 1:
                # check if contour is
                # a. Long enough
                # b. is deep enough (bigger than minimum)
                # c. Closes (x and y of first equals x and y of last line)
                if len(currentContour) > Nmin and level > Lmin_source \
                        and currentContour[0][0] == currentContour[-1][0] \
                        and currentContour[0][1] == currentContour[-1][1]:
                    sourceContours.append(currentContour)
                # same, but than check if level is lower than max
                if len(currentContour) > Nmin and level < Lmax_sink \
                        and currentContour[0][0] == currentContour[-1][0] \
                        and currentContour[0][1] == currentContour[-1][1]:
                    sinkContours.append(currentContour)
    return sourceContours, sinkContours

#%%
def calculate_directional_stability(waveData, dataBucketName = "", windowSize=10):
    """ calculates the directional stability of the UV maps.
        Arguments:

        waveData: waveData object
        dataBucketName: name of the dataBucket to use. Defaults to the last active dataBucket
        windowSize: size of the window in samples during which a vector needs to be pointing in the same direction to be considered stable

        Example: 
        Temporal frequency of interest = 10Hz 
        sampling rate = 250Hz
        WindowSize = 50 samples (2 cycles of the Temporal Frequency) """
        
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "trl_posx_posy_time")
    UV_direction = currentData / np.abs(currentData)
    trl, nposx, nposy, nframes = UV_direction.shape
    averageVectors = np.zeros((trl, nposx, nposy, nframes-windowSize), dtype='complex')
    for trialNr in range(trl):
        for frameNr in range(nframes - windowSize): 
            currentUV = UV_direction[trialNr,:, :, frameNr:frameNr + windowSize]
            for x in range(nposx):
                for y in range(nposy):
                    averageVectors[trialNr, x, y, frameNr] = np.sum(currentUV[x, y, :]) / windowSize 
    if hasBeenReshaped:
        #reshape back to original dimord (the -1 in the shape is because the last dimension 
        # has been reduced by the size of the window. passing -1 to reshape makes numpy 
        # figure out the correct size)
        averageVectors = np.reshape(averageVectors, (*oldshape[:-1],-1))
    dataBucket = wd.DataBucket(averageVectors, "Directional_Stability_Timeseries",waveData.DataBuckets[dataBucketName].get_dimord(), waveData.DataBuckets[dataBucketName].get_channel_names() )
    waveData.add_data_bucket(dataBucket)
    
def source_sink_process_trial(thistrialInd, trial_data):
    #initialize arrays
    SourcePoincareJacobian = np.zeros_like(trial_data[:,:,:], dtype=int)
    SinkPoincareJacobian = np.zeros_like(trial_data[:,:,:], dtype=int)
    timepoints = trial_data.shape[2]
    sourceContours = np.empty(timepoints, dtype=object)
    sinkContours = np.empty(timepoints, dtype=object)
    # Initialize dataframes to store confirmed sinks and sources information
    source_df = pd.DataFrame(columns=['trial', 'timepoint', 'posx', 'posy','type'])
    sink_df = pd.DataFrame(columns=['trial', 'timepoint', 'posx', 'posy','type'])
    [uy, ux] = np.gradient(np.real(trial_data[:,:,:]))[:2]
    [vy, vx] = np.gradient(np.imag(trial_data[:,:,:]))[:2]
    for idx in range(timepoints):
        # find critical points
        PoincareSinkSource, PoincareSaddle = poincare_index(trial_data[:, :, idx])
        FixedPointsPoincare = PoincareSinkSource + PoincareSaddle
        [col, row] = (np.where(FixedPointsPoincare.T == 1))
        # Iterate through each detected critical point 
        for f in range(len(row)):
            r = row[f]
            c = col[f]
            # Construct the Jacobian matrix for spatial gradients
            J = np.array([[ux[r, c, idx], uy[r, c, idx]],
                        [vx[r, c, idx], vy[r, c, idx]]])
            # Calculate the determinant and trace of the Jacobian matrix
            delta = np.linalg.det(J)
            tau = np.trace(J)
            # Determine the type of critical point and its stability
            [type, SP] = SourceSinkSaddle(delta, tau)

            if type == 1:
                if SP == 1:
                    SourcePoincareJacobian[r, c, idx] = 1  # for spiral sources
                else:
                    SourcePoincareJacobian[r, c, idx] = 2  # for node sources
            elif type == -1:
                if SP == 1:
                    SinkPoincareJacobian[r, c, idx] = 1  # for spiral sinks
                else:
                    SinkPoincareJacobian[r, c, idx] = 2  # for node sinks
            else:
                SinkPoincareJacobian[r, c, idx] = 0 
                SourcePoincareJacobian[r, c, idx] = 0
                    
    # triple check using gradient of vector field
    # Parameters for source/sink detection
    Nmin = 2  # minimum number of points as the contour size
    Lmin_source = 0.05  # minimum source level
    Lmax_sink = -0.05  # maximum sink level
    Nnested_source = 2  # minimum number of nested sources to verify the most interior source
    Nnested_sink = 2  # minimum number of nested sources to verify the most interior sink

    for it in range(timepoints):
        u = np.real(trial_data[:, :, it])
        v = np.imag(trial_data[:, :, it])
        sourceContours[it], sinkContours[it] = makeContours(
            u, v, Nmin, Lmin_source, Lmax_sink)
    for ii, potentialSourcePoints in enumerate(np.moveaxis(SourcePoincareJacobian, -1, 0)):
        for thisContour in sourceContours[ii]:
            # Make path from current sourcecontour
            p = path.Path(thisContour)
            # find points
            [rS, cS] = np.nonzero(potentialSourcePoints)
            if (len(rS) > 0):
                coordinates = list(zip(rS, cS))
                # check if points are within contour
                isSource = p.contains_points(coordinates)
                if (np.any(isSource)):
                    for sourceInd, (r, c) in enumerate(coordinates):
                        if isSource[sourceInd]:
                            # Add source info to the dataframe
                            source_df = source_df.append({  
                                'trial': thistrialInd,
                                'timepoint': ii,
                                'posx': r,
                                'posy': c,
                                'type': 'source' 
                            }, ignore_index=True) 

    for ii, potentialSinkPoints in enumerate(np.moveaxis(SinkPoincareJacobian, -1, 0)):
        for thisContour in sinkContours[ii]:
            # Make path from current sourcecontour
            p = path.Path(thisContour)
            # find points
            [rS, cS] = np.nonzero(potentialSinkPoints)
            if (len(rS) > 0):
                coordinates = list(zip(rS, cS))
                # check if points are within contour
                isSink = p.contains_points(coordinates)
                if (np.any(isSink)):
                    for sinkInd, (r, c) in enumerate(coordinates):
                        if isSink[sinkInd]:
                            # Add sink info to the dataframe
                            sink_df = sink_df.append({  
                                'trial': thistrialInd,
                                'timepoint': ii,
                                'posx': r,
                                'posy': c,
                                'type': 'sink'
                            }, ignore_index=True)
    plt.close()
    return source_df, sink_df

def find_sources_sinks(waveData, dataBucketName = ""):
    """requires posx, posy channel dimensions. dataBucketName defaults to "UV" 
    Input: waveData: waveData object
    dataBucketName: name of the dataBucket to use. Defaults to "UV"
    Output: Source and Sink DataFrames with columns: trial, timepoint, posx, posy
    identifies sinks, sources, and saddles in UV maps using the Poincare index theorem 
    and Jacobian matrices. Each critical point is checked against contours of the
    divergence of the vector field to confirm its type and stability.    
    """
    # Find Sources and Sinks
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    UV = waveData.DataBuckets[dataBucketName].get_data()
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    oldShape = UV.shape
    hasBeenReshaped, UV =  hf.force_dimord(UV, currentDimord , "trl_posx_posy_time")
    trial, sizeX, sizeY, timepoints = UV.shape

    if os.name == 'posix':  # Linux
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = pool.starmap(source_sink_process_trial, [(thistrialInd, trial_data) for thistrialInd, trial_data in enumerate(UV)])        
        pool.close()
        pool.join()
    else:  # Windows or other OS
        from joblib import Parallel, delayed
        num_cores = os.cpu_count()  # Get number of cores
        result = Parallel(n_jobs=num_cores)(delayed(source_sink_process_trial)
                                            (thistrialInd, trial_data) for thistrialInd, trial_data 
                                            in enumerate(UV))

    source_df = pd.concat([x[0] for x in result])
    sink_df = pd.concat([x[1] for x in result])

    source_df = source_df.drop_duplicates()
    sink_df = sink_df.drop_duplicates()

    if hasBeenReshaped:
        def get_original_indices_from_flat(item, oldShape):
            x, y = oldShape
            x_original = item // y
            y_original = item % y
            return x_original, y_original

        # Iterate through DataFrame entries
        for index, row in sink_df.iterrows():
            # Calculate the original x and y indices
            x_original, y_original = get_original_indices_from_flat(int(row['trial']), oldShape[:2])            
            # Update 'freq' and 'trial' columns
            sink_df.at[index, 'freqBin'] = int(x_original)
            sink_df.at[index, 'trial'] = int(y_original)

        for index, row in source_df.iterrows():
            # Calculate the original x and y indices
            x_original, y_original = get_original_indices_from_flat(int(row['trial']), oldShape[:2])
            
            # Update 'freq' and 'trial' columns
            source_df.at[index, 'freqBin'] = int(x_original)
            source_df.at[index, 'trial'] = int(y_original)

    return source_df, sink_df 


#%% New optical Flow
def get_surround_locs(nrows, ncols):
    dxMatrix = lil_matrix((nrows*ncols, nrows*ncols))
    dyMatrix = dxMatrix.copy()
    lapMatrix = dxMatrix.copy()

    for irow in range(nrows):
        for icol in range(ncols):
            xSurrLocs = np.array([(irow-2)*ncols + icol, (irow-1)*ncols + icol, irow*ncols + icol, (irow+1)*ncols + icol, (irow+2)*ncols + icol])
            ySurrLocs = np.array([irow*ncols + icol-2, irow*ncols + icol-1, irow*ncols + icol, irow*ncols + icol+1, irow*ncols + icol+2])

            lapWeight = np.array([1, 1, 1, 1])

            if irow == 0:
                dxLocs = xSurrLocs[[2,3]]
                dxWeight = np.array([-1, 1])
                lapWeight = np.array([0, 2, 1, 1])
            elif irow == nrows-1:
                dxLocs = xSurrLocs[[1,2]]
                dxWeight = np.array([-1, 1])
                lapWeight = np.array([2, 0, 1, 1])
            elif irow == 1 or irow == nrows-2:
                dxLocs = xSurrLocs[[1,3]]
                dxWeight = np.array([-0.5, 0.5])
            else:
                dxLocs = xSurrLocs[[0,1,3,4]]
                dxWeight = np.array([1/12, -8/12, 8/12, -1/12])

            if icol == 0:
                dyLocs = ySurrLocs[[2,3]]
                dyWeight = np.array([-1, 1])
                lapWeight = lapWeight + np.array([0, 0, -1, 1])
            elif icol == ncols-1:
                dyLocs = ySurrLocs[[1,2]]
                dyWeight = np.array([-1, 1])
                lapWeight = lapWeight + np.array([0, 0, 1, -1])
            elif icol == 1 or icol == ncols-2:
                dyLocs = ySurrLocs[[1,3]]
                dyWeight = np.array([-0.5, 0.5])
            else:
                dyLocs = ySurrLocs[[0,1,3,4]]
                dyWeight = np.array([1/12, -8/12, 8/12, -1/12])

            lapLocs = np.concatenate((xSurrLocs[[1,3]], ySurrLocs[[1,3]]))
            lapLocs = lapLocs[lapWeight!=0]
            lapWeight = lapWeight[lapWeight!=0]

            thisRow = np.ravel_multi_index((irow, icol), (nrows, ncols))
            dxMatrix[thisRow, dxLocs] = dxWeight
            dyMatrix[thisRow, dyLocs] = dyWeight
            lapMatrix[thisRow, lapLocs] = lapWeight

    lapMatrix = lapMatrix + lil_matrix(np.diag(-4*np.ones(nrows*ncols)))
    surroundLocs = {'dx': dxMatrix, 'dy': dyMatrix, 'laplacian': lapMatrix}
    print('Surround Locs Done')
    return surroundLocs

@numba.jit(nopython=True)    
def anglesubtract(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

#@numba.njit
def phasegradient(f, angleFlag):
    nrow = f.shape[0]
    ncol = f.shape[1]

    # Initialize the gradient arrays
    gfx = np.zeros((nrow, ncol), dtype=f.dtype)
    gfy = np.zeros((nrow, ncol), dtype=f.dtype)

    # Compute the gradient in the x direction
    if ncol > 1:
        if angleFlag:
            gfx[:, 0] = anglesubtract(f[:, 1], f[:, 0])  # forward difference at left edge
            gfx[:, -1] = anglesubtract(f[:, -1], f[:, -2])  # forward difference at right edge
        else:
            gfx[:, 0] = f[:, 1] - f[:, 0]  # forward difference at left edge
            gfx[:, -1] = f[:, -1] - f[:, -2]  # forward difference at right edge
    if ncol > 2:
        if angleFlag:
            gfx[:, 1:-1] = anglesubtract(f[:, 2:], f[:, :-2]) / 2  # centered difference at interior points
        else:
            gfx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2  # centered difference at interior points

    # Compute the gradient in the y direction
    if nrow > 1:
        if angleFlag:
            gfy[0, :] = anglesubtract(f[1, :], f[0, :])  # forward difference at top edge
            gfy[-1, :] = anglesubtract(f[-1, :], f[-2, :])  # forward difference at bottom edge
        else:
            gfy[0, :] = f[1, :] - f[0, :]  # forward difference at top edge
            gfy[-1, :] = f[-1, :] - f[-2, :]  # forward difference at bottom edge
    if nrow > 2:
        if angleFlag:
            gfy[1:-1, :] = anglesubtract(f[2:, :], f[:-2, :]) / 2  # centered difference at interior points
        else:
            gfy[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2  # centered difference at interior points

    return gfx, gfy

def opticalFlowStep(Ex, Ey, Et, angleFlag, maxIter=1000, maxChange=0.01, alpha=0.1, is_linear=True, beta=1.0, surroundLocs=None):
    #set up fixed point iteration parameters
    relaxParam = 1.1 #Starting relaxation parameter for fixed point iteration
    relaxDecStep = 0.02 #Step to decrease relaxation parameter every iteration to ensure convergence
    relaxParamMin = 0.2 #Minimum relaxation parameter

    # Initialize optical flow fields
    u = np.zeros_like(Ex)
    v = np.zeros_like(Ey)
    #initialize the smoothness and error
    smoothE = np.inf
    dataE = np.inf
    ncol= Ex.shape[0]
    nrow = Ey.shape[1]
    N = ncol * nrow
    Ones=np.ones((1, N))
    sp_zeros = lil_matrix((N, N))

    # Create a KSP object (Krylov subspace method)
    ksp = PETSc.KSP().create()
    # Organize the discretized optical flow equations into a system of linear equations in the form Ax=b. 
    # Loop over penalty parameters
    for convergenceLoop in range(maxIter):
        lastDataE = dataE
        lastSmoothE = smoothE

        # Compute the first order error in data and smoothness
        dataE = Ex*u + Ey*v + Et
        upx, upy = phasegradient(u, angleFlag)
        vpx, vpy = phasegradient(v, angleFlag)
        smoothE = upx**2 + upy**2 + vpx**2 + vpy**2

        # Compute nonlinear penalty functions
        dataP = 0.5/beta*(beta**2 + dataE**2)**(-1/2)
        smoothP = 0.5/beta*(beta**2 + smoothE)**(-1/2)

        # Check if data and smoothing errors have reached a fixed point
        dataEChange = np.abs(dataE-lastDataE) / np.abs(dataE)
        smoothEChange = np.abs(smoothE-lastSmoothE) / np.abs(smoothE)
        # Exit loop if fixed point has been reached
        if np.max(dataEChange) < maxChange and np.max(smoothEChange) < maxChange:
            break 

        if is_linear:
            # Use original Horn-Schunk equations
            gamma = dataP / alpha
            delta = 4*smoothP
            surroundTerms = surroundLocs["laplacian"] * np.tile(smoothP.flatten(), (N, 1)).T
        else:
            # Use non-linear penalty function for more robust results (but calculation may take more time)
            gamma = dataP / alpha
            delta = 0
            # Surrounding terms are a combination of laplacian and first spatial derivative terms
            psx, psy = phasegradient(smoothP, angleFlag) 
            surroundTerms = surroundLocs["dx"] * (psx.flatten()[:, None] * Ones) + \
                surroundLocs["dy"] * (psy.flatten()[:, None] * Ones) + \
                surroundLocs["laplacian"] * (smoothP.flatten()[:, None] * Ones)
        # Calculate b vector
        b = np.concatenate((gamma.flatten() * Et.flatten() * Ex.flatten(), gamma.flatten() * Et.flatten() * Ey.flatten()))

        # Add diagonal terms
        if np.isscalar(delta):
            delta_array = delta * np.ones_like(Ex.flatten())
        else:
            delta_array = delta.flatten()

        diag_vals = np.concatenate((-delta_array - Ex.flatten()**2 * gamma.flatten(), -delta_array - Ey.flatten()**2 * gamma.flatten()))    
        # Create sparse diagonal matrix
        A = diags(diag_vals)

        # Add off-diagonal terms for ui-vi dependence
        uvDiag = -Ex.flatten() * Ey.flatten() * gamma.flatten()
        p_off_diag = diags([uvDiag, uvDiag], [N, -N])
        A += p_off_diag

        # Add other terms for surrounding locations
        A += bmat([[surroundTerms, sp_zeros], [sp_zeros, surroundTerms]], format='csr')

        # Add a small value along the diagonal to avoid potentially having a singular matrix
        A.setdiag(A.diagonal() + 1e-10)

        # Solve the system of linear equations
        #xexact = spsolve(A, b) #revert to this if you don not want to use petsc
        #alternatively, still using scipy, but with a different solver
        # Solve the system of linear equations
        if use_petsc:
            # Solve the system using PETSc
            A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
            b_petsc = PETSc.Vec().createWithArray(b)
            x_petsc = PETSc.Vec().createWithArray(np.zeros_like(b))
            ksp = PETSc.KSP().create()
            ksp.setOperators(A_petsc)
            ksp.solve(b_petsc, x_petsc)
            xexact = x_petsc.getArray()
        else:
            # Solve the system using scipy
            xexact, info = gmres(A, b)
            if info != 0:
                print("WARNING: Conjugate Gradient did not converge!")

        # Reshape back to grids
        u *= (1-relaxParam)
        u += relaxParam*np.reshape(xexact[:N], (ncol, nrow))
        v *= (1-relaxParam)
        v += relaxParam*np.reshape(xexact[N:], (ncol, nrow))

        # Gradually reduce the relaxation parameter to ensure the fixed point iteration converges
        if relaxParam > relaxParamMin:
            relaxParam = relaxParam - relaxDecStep
           
    return u, v

def process_observation(obs, ExAll, EyAll, temporal_derivatives, angleFlag, maxIter, maxChange, alpha, is_linear, beta, surroundLocs):
    # Initialize arrays for the results
    ivxx_all = np.empty(ExAll.shape)
    ivyy_all = np.empty(EyAll.shape)
    nframes = ExAll.shape[-1]
    # Loop over time
    for t in range(nframes-1):
        ivxx, ivyy = opticalFlowStep(ExAll[:,:,t], 
                                     EyAll[:,:,t],  
                                     temporal_derivatives[:, :, t],
                                     angleFlag, 
                                     maxIter=maxIter,
                                     maxChange=maxChange, 
                                     alpha=alpha, 
                                     is_linear=is_linear,
                                     beta=beta, 
                                     surroundLocs=surroundLocs)
        ivxx_all[:,:,t] = ivxx
        ivyy_all[:,:,t] = ivyy

    return obs, ivxx_all, ivyy_all

#%% main function
def opticalFlow(waveData, dataBucketName=None, angleFlag= 'True', maxIter=1000, maxChange=0.01, alpha=0.1, is_linear = True, beta=10.0, nThreads = 10):
    '''
    Calculates the optical flow of a 3d timeseries. The data is expected to be in the format of a 4D array with the dimensions
    trial, posx, posy, time. Variatinons in the form of e.g., freq,trial, posx, poy, time are accepted. 
    Any extra dimensions preceding trl_posx_posy_time will be flattened into the trial dimension. Original shape will be
    restored after the calculation.
    Returns a new dataBucket with the optical flow data, with DataBucketName = "UV".
    Arguments:
    waveData: waveData object
    dataBucketName: name of the dataBucket to use. Defaults to the last active dataBucket
    angleFlag: if True, optical Flow is calculated on the angles of the complex UV map. If False, the absolute of the UV map is used.
    maxIter: maximum number of iterations for the optical flow calculation
    maxChange: maximum change in the optical flow before the calculation is considered converged
    alpha: smoothness parameter for the optical flow calculation, higher values result in smoother optical flow fields. Should be between .001 and 1.
    Caution: alpha very strongly influences the magnitude of the flow vectors and should be chosen carefully!!! Don't blindly use them as velocity estimates, especially if alpha is high.
    is_linear: if True, the optical flow calculation uses the original Horn-Schunck equations. If False, a non-linear penalty function is used (more robust, but can be very slow)
    beta: parameter for the non-linear penalty function. Only used if is_linear is False 
    nThreads: number of threads to use for the calculation. Default is 10. Set to however many cores you have -1 for optimal performance.
    '''
    
    if dataBucketName is None:
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
        
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "trl_posx_posy_time")

    nObs,posx,posy,nframes = currentData.shape
 
    surroundLocs  = get_surround_locs(posx, posy)
    #surroundLocsLaplacian= surroundLocs['laplacian']
    maxIter = maxIter
    maxChange = maxChange
    alpha = alpha
    velocityX = np.zeros((nObs,posx,posy,nframes-1))
    velocityY = np.zeros((nObs,posx,posy,nframes-1))
    
    if angleFlag:
        if np.iscomplex(currentData).all():
            currentData = np.angle(currentData)
    else:
        currentData = np.abs(currentData)

    # Pre-calculate phase gradients and temporal derivatives
    phase_gradients_x = np.empty((nObs, currentData.shape[1], currentData.shape[2], nframes))
    phase_gradients_y = np.empty((nObs, currentData.shape[1], currentData.shape[2], nframes))
    temporal_derivatives = np.empty((nObs, currentData.shape[1], currentData.shape[2], nframes-1))
    ExAll = np.empty((nObs, currentData.shape[1], currentData.shape[2], nframes-1))
    EyAll = np.empty((nObs, currentData.shape[1], currentData.shape[2], nframes-1))
    for obs in range(nObs):
        for t in range(nframes):
            phase_gradients_x[obs, :, :, t], phase_gradients_y[obs, :, :, t] = phasegradient(currentData[obs, :, :, t], angleFlag)
    for obs in range(nObs):
        #loop over time
        for t in range(nframes-1):
            # Calculate Ex and Ey
            if angleFlag:
                ExAll[obs,:,:,t] = -(phase_gradients_x[obs,:,:,t] + phase_gradients_x[obs,:,:,t+1]) / 2
                EyAll[obs,:,:,t] = -(phase_gradients_y[obs,:,:,t] + phase_gradients_y[obs,:,:,t+1]) / 2
            else:
                ExAll[obs,:,:,t] = (phase_gradients_x[obs,:,:,t] + phase_gradients_x[obs,:,:,t+1]) / 2
                EyAll[obs,:,:,t] = (phase_gradients_y[obs,:,:,t] + phase_gradients_y[obs,:,:,t+1]) / 2
            if np.iscomplex(currentData[obs, :, :, t]).all():
                temporal_derivatives[obs, :, :, t] = anglesubtract(np.angle(currentData[obs, :, :, t]), np.angle(currentData[obs, :, :, t+1]))
            else:
                temporal_derivatives[obs, :, :, t] = anglesubtract(currentData[obs, :, :, t+1], currentData[obs, :, :, t])
    
    # Create a multiprocessing Pool
    print('starting multiprocessing pool')
    with multiprocessing.Pool(nThreads) as pool:
        # Create a partial function with all the arguments except 'obs'
        partial_func = partial(process_observation, 
                            angleFlag=angleFlag, 
                            maxIter=maxIter, 
                            maxChange=maxChange, 
                            alpha=alpha, 
                            is_linear=is_linear, 
                            beta=beta, 
                            surroundLocs=surroundLocs)

        # Map partial_func to all observations
        results = []
        for obs in range(nObs):
            Ex = ExAll[obs]
            Ey = EyAll[obs]
            temporal_derivative = temporal_derivatives[obs]
            result = pool.apply_async(partial_func, args=(obs, Ex, Ey, temporal_derivative))
            results.append(result)

        # Collect the results
        for result in results:
            obs, ivxx_all, ivyy_all = result.get()
            velocityX[obs,:,:,:] = ivxx_all
            velocityY[obs,:,:,:] = ivyy_all
    allUV = velocityX + 1j * velocityY
    #reshape back to original dimord, take into account that the last dimension has been reduced by 1
    allUV = np.reshape(allUV, oldshape[:-1] + (oldshape[-1] - 1,))       
    dataBucket = wd.DataBucket(allUV, "UV",currentDimord,
                               waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)
