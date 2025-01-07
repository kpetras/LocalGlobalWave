import Modules.Utils.HelperFuns as hf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from scipy.interpolate import griddata
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def init():
     plt.style.use("settings.mplstyle")

def getProbeColor(index, totalProbes):
    #cmap = plt.cm.hsv
    cmap = plt.cm.ocean
    return cmap(index/totalProbes) 

# 
def plotfft_zoomed(fft_abs, sfreq, minFreq, maxFreq, title, scale='linear'):
    
    nChan, nTimepoints = fft_abs.shape
    spatialFreqAxis = nChan/2 * np.linspace(-1, 1, nChan)
    tempFreqAxis = np.arange(-sfreq/2, sfreq/2, 1/(nTimepoints/sfreq))
    plotrange = np.where((tempFreqAxis > minFreq) & (tempFreqAxis < maxFreq))
    if scale == 'log':
        fft_abs = np.log10(fft_abs + 1e-12)
        fft_abs = (fft_abs - np.min(fft_abs)) / (np.max(fft_abs) - np.min(fft_abs))

    plt.imshow(fft_abs[:, plotrange[0]], aspect="auto", extent=[tempFreqAxis[plotrange[0]][0], tempFreqAxis[plotrange[0]][-1], spatialFreqAxis[0], spatialFreqAxis[-1]])
    plt.colorbar(label="Power (dB)" if scale == "log" else "Power")
    plt.title("{title} Spatial Freq over Temporal Freq".format(title=title))
    plt.xlabel("Temporal Frequency (Hz)")
    plt.ylabel("Spatial Frequency (channels/Hz)")
    #plt.show()
    return plt  

def plot_imfs(imfs, IMFofInterest, time):
    """Plots the imfs and phase of the IMF of interest
    Parameters
    ----------
    imfs : array
        The imfs to plot. Needs shape (nTimepoints, nIMFs)
    IMFofInterest : int
        The index of the IMF to plot the phase of
    time : array
        The time vector for the imfs
    """
    import emd
    IP = np.angle(imfs[:,IMFofInterest])  
    # remove any imfs that are NaN
    imfs = imfs[:,~np.isnan(imfs[0,:])]
    emd.plotting.plot_imfs(imfs=imfs, time_vect=time, cmap=True, xlabel = 'Time (seconds)')
    f1 = plt.gcf()
    f2 = plt.figure(figsize= [16, 3])
    # Plot Phase
    plt.plot(time,IP)
    plt.title('Phase of IMF of Interest (IMF '+ str(IMFofInterest+1))
    plt.xlabel('Time (seconds)')
    plt.xlim(time[0], time[-1])
    plt.yticks(np.arange(-np.pi, np.pi, step=np.pi/2), [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-np.pi, np.pi, step=np.pi/2)])
    plt.ylim(-np.pi, np.pi)
    plt.ylabel('Phase')
    return f1 , f2
    #plt.subplots_adjust(left=0.4, right=0.99)

def plot_interpolated_data(waveData, original_data_bucket, interpolated_data_bucket, grid_x, grid_y, OrigInd, InterpInd, type = ""):
    """Plots comparison between original and interpolated data. 
       OrigInd is the index into the original dataBucket to plot (usually something like (trl,:,timepoint)), 
       InterpInd is the index into the interpolated dataBucket to plot (usually something like (trl,:,:,timepoint))
    Args:
        waveData: WaveData object
        original_data_bucket: str with name of original data bucket
        interpolated_data_bucket: str with name of interpolated data bucket
        grid_x : interpolated 2d channel x-coordinates
        grid_y : interpolated 2d channel y-coordinates
        trial_idx: which trial to plot. Defaults to 0.
        time_point: which timepoint to plot. Defaults to 500.
        type: "" (default) just plots the data. Options: "phase"/"angle" or "power"/"abs" if data is complex 
    """
    original_data = waveData.get_data(original_data_bucket)[OrigInd]
    interpolated_data = waveData.get_data(interpolated_data_bucket)[InterpInd].ravel()
    norm = None #default colormapping

    if type == "phase" or type == "angle":
        original_data = np.angle(original_data)
        interpolated_data = np.angle(interpolated_data)
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi) #fix range from -pi to pi
    elif type == "power" or type == "abs":
        original_data = np.abs(original_data)
        interpolated_data = np.abs(interpolated_data)
    # if none of the above, just plot the data. If complex this defaults to the real part anyways
    #get 3d positions
    pos_3d = waveData.get_channel_positions()
    # Scale the pos_2d coordinates if needed
    pos_2d = waveData.get_2d_coordinates() 

    #create scatter plot of original 3d channel positions
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # Create scatter plot of original 3D channel positions
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c=original_data, norm=norm)
    plt.colorbar(scatter, label=type)
    plt.title('Original Data')
    ax.set_xlabel('X coordinate (cm)')
    ax.set_ylabel('Y coordinate (cm)')
    ax.set_zlabel('Z coordinate (cm)')
    ax.view_init(elev=90, azim=-90)  # View the plot from the top
    plt.axis('auto')

    # Create scatter plot of 2D projected positions
    ax2 = plt.subplot(1, 3, 2)
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c=original_data, norm=norm)
    plt.colorbar(label=type)
    plt.title('2D Projected Data')
    plt.xlabel('X coordinate (cm)')
    plt.ylabel('Y coordinate (cm)')
    ax2.set_aspect('auto')  # Set the aspect ratio to be equal

    # Create scatter plot of interpolated data
    ax3 = plt.subplot(1, 3, 3)
    plt.scatter(grid_x.ravel(), grid_y.ravel(), c=interpolated_data, norm=norm)
    plt.colorbar(label=type)
    plt.title('Interpolated Data')
    plt.xlabel('X coordinate (cm)')
    plt.ylabel('Y coordinate (cm)')
    ax3.set_aspect('auto')  # Set the aspect ratio to be equal

    plt.tight_layout()
    plt.show()
    return fig

def plot_timeseries_on_surface(Surface, waveData, dataBucketName = " ", indices = (0, 0, None, slice(None), slice(None)), chan_to_highlight = 0 , timepoint =0, plottype = "power"):
    '''Plot topo time series on a surface
    + actual timeseries of a selected channel
    Defaults to plotting power. Set type to "real" to plot real part of the data, "phase" to plot angle
    Parameters
    ----------
    Surface : list
        list containing vertices and faces of the surface
    waveData : WaveData object
    dataBucketName : str
        name of the data bucket to plot
    indices : tuple
        data indeces. IAll exlicit indeces are used as such, None is averaged over and slice(None) stays as is
        example: (0, 0, None, slice(None), slice(None)) will plot mean(data[1,3,:,:,:], axis = 2). the remaining dimensions need to be channels x time
    chan_to_highlight : int
        channel to plot timeseries of
    timepoint : int
        timepoint to plot topo of
    plottype : str
        "power" (default), "real" or "phase"
    '''

    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimlist = dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()

    faces = Surface[1]
    faces = faces.reshape(-1, 3)
    channel_positions = waveData.get_channel_positions()
    time = waveData.get_time()

    # Create base figure
    fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'scene'}], [{'type': 'xy'}]], 
                        subplot_titles=('3D Surface', 'Time Series'), vertical_spacing=0.3)
    
    # Get the channel data. All exlicit indeces are used as such, None is averaged over and slice(None) stays as is
    average_axes = tuple(i for i, index in enumerate(indices) if index is None)
    data = np.mean(data, axis=average_axes, keepdims=True)
    # Create a new set of indices that only includes the dimensions that weren't averaged
    new_indices = tuple(index if isinstance(index, int) else slice(None) for index in indices)
    channel_data = data[new_indices]
    channel_data = channel_data.squeeze()

    # Add traces, one for each slider step
    if plottype == "power":
        channel_data = np.abs(channel_data)
    elif plottype == "real":
        channel_data = np.real(channel_data)
    elif plottype == "phase":
        channel_data = np.angle(channel_data)
    clim = [np.min(channel_data), np.max(channel_data)]
    for timepoint in range(len(time)):
        channel_data_snapshot = channel_data[:, timepoint]
        # Add a trace for the surface
        fig.add_trace(
            go.Mesh3d(
                x=Surface[0][:, 0],
                y=Surface[0][:, 1],
                z=Surface[0][:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgrey',
                opacity=.8,
                visible=False
            ),
            row=1, col=1
        )
        # Add a trace for the channel positions
        fig.add_trace(
            go.Scatter3d(
                x=channel_positions[:, 0],
                y=channel_positions[:, 1],
                z=channel_positions[:, 2],
                mode='markers',
                marker=dict(size=10, color=channel_data_snapshot, 
                            cmin= clim[0], cmax=clim[1],
                            colorscale='RdBu_r', 
                            colorbar=dict(title=plottype, x=-0.07, len=0.7)),
                visible=False
            ),
            row=1, col=1
        )

        # Add a trace to highlight the selected channel
        fig.add_trace(
            go.Scatter3d(
                x=[channel_positions[chan_to_highlight, 0]],
                y=[channel_positions[chan_to_highlight, 1]],
                z=[channel_positions[chan_to_highlight, 2]],
                mode='markers',
                marker=dict(
                    size=12, 
                    color='rgba(0,0,0,0)',  # transparent fill
                    line=dict(color='red', width=5)  # black border
                ),
                visible=False
            ),
            row=1, col=1
        )

    
    # Add the time series trace to the current step
    fig.add_trace(
        go.Scatter(
            x=time,
            y=channel_data[chan_to_highlight, :],
            mode='lines',
            line=dict(color='black', width=2),
            visible=True
        ),
        row=2, col=1
    )

    # Create and add slider
    steps = []
    for i in range(0, len(fig.data)-1, 3):  
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},  # Start by making all traces invisible
                {"title": "Time: " + str(time[i//3])}],  # layout attribute
        )
        # Make the current 3D traces visible
        if i < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i] = True
        if i+1 < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i+1] = True
        if i+2 < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i+2] = True

        # Make the time series trace visible
        step["args"][0]["visible"][-1] = True

        # Update the position of the vertical line with the slider
        step["args"].append({"shapes": [
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=time[i//3],
                y0=0,
                x1=time[i//3],
                y1=1,
                line=dict(
                    color="Red",
                    width=3
                )
            )
        ]})

        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(nticks=4, range=[np.min(Surface[0]),np.max(Surface[0])],),
            yaxis=dict(nticks=4, range=[np.min(Surface[0]),np.max(Surface[0])],),
            zaxis=dict(nticks=4, range=[np.min(Surface[0]),np.max(Surface[0])],),
            aspectmode='cube',
            domain=dict(y=[0.3, 1])  # Adjust the size of the 3D subplot
        ),
        xaxis=dict(domain=[0, 1], anchor='y2'),  # Adjust the size of the 2D subplot
        yaxis=dict(domain=[0, 0.25], anchor='x2'),  # Adjust the size of the 2D subplot
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    fig.show()
    return fig
#
def animate_grid_data(gridData,DataBucketName = "", dataInd = None, probepositions=[(0,0)], plottype = "real"):
    """Plots gridData over time. 
        gridData: waveData object.  
        dataInd: Needs to point to a single trial with shape posx_posy_time, it is either a single int or a tuple. Time ranges can be indicated as
        eg. (0, 0, slice(None), slice(None), [491, 492, 493, 494, 495, 496, 497, 498, 499, ...]) to index some point

    """
    if DataBucketName == "":
        DataBucketName = gridData.ActiveDataBucket
    timevec = gridData.get_time()    

    plotGridSize = (1,2)
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure(figsize=(plotGridSize[1]*8, plotGridSize[0]*8))

    #IMSHOW grid
    ax1 = plt.subplot2grid(plotGridSize, (0, 0), colspan=1, rowspan=1)
    ax1.grid(None)
    plt.set_cmap('copper')  
    #plt.tight_layout()
    ax1.axis('off')
    
    if dataInd is not None:
        if isinstance(dataInd, int):
            dataToPlot = gridData.get_data(DataBucketName)[dataInd, :, :, :]  #just pick trl
        elif isinstance(dataInd, tuple):
            dataToPlot = gridData.get_data(DataBucketName)[dataInd]  
        else:
            raise ValueError("dataInd must be an integer or a tuple")
    else:
        dataToPlot = gridData.get_data(DataBucketName)
    
    # somehow python sometimes shifts dims around. Check and fix
    if dataToPlot.ndim == 3:
        posx, posy, time = dataToPlot.shape if dataToPlot.shape[1] != dataToPlot.shape[2] else dataToPlot.shape[::-1]#this only works if posx==posy. Fix later
        if dataToPlot.shape == (time, posx, posy):  
            dataToPlot = np.transpose(dataToPlot, (1, 2, 0))
            #if that happens, most likely the timevec has changed
            timevec = timevec[dataInd[-1]]
        elif dataToPlot.shape == (posx, posy, time):  
            pass
        else:
            raise ValueError("dataToPlot does not have the right shape.")
    else:
        raise ValueError("dataToPlot should have 3 dimensions after indexing, but got {}".format(dataToPlot.ndim))


    if plottype== 'real':
        dataToPlot = np.real(dataToPlot)
    elif plottype == 'power':
        dataToPlot = np.abs(dataToPlot)
    elif plottype == 'angle': 
        dataToPlot = np.angle(dataToPlot)
    elif plottype == 'isPhase':
        print('data is assumed to already be phase data')
        
    vmin, vmax = np.percentile(dataToPlot, [1, 99])
    img = ax1.imshow(dataToPlot[ :, :, 0],
                    origin='lower', vmin=vmin, vmax=vmax, cmap="copper")

    cbar = plt.colorbar(img)
    cbar.set_label('$\mu$V')
    nFrames = dataToPlot.shape[-1]  
    lengthOfMatrix =  dataToPlot.shape[0] * dataToPlot.shape[1]
    # make all black
    probecolors = []
    allEdgeColors = [(0.0, 0.0, 0.0)for i in range(lengthOfMatrix)]
    for ind, probe in enumerate(probepositions):
        currentColor = getProbeColor(ind, len(probepositions))
        currentRect = plt.Rectangle((probe[1]-0.5, probe[0]-0.5), 1, 1, facecolor='none',edgecolor=currentColor,lw=2)
        probecolors.append(currentRect.get_edgecolor())
        ax1.add_patch(currentRect)

    currentShape = dataToPlot.shape

    nframes = currentShape[2]
    lineseriesdata = np.zeros((len(probepositions), nframes), dtype='float64')
    currentPlot = plt.subplot2grid(
        plotGridSize, (0,1), colspan=1, rowspan=1)
    currentPlot.plot(timevec, lineseriesdata.T,linewidth=3)
    currentPlot.grid(visible = False)
    currentPlot.set_ylabel([])
    currentPlot.set_facecolor("white")

    ylim = np.array([np.min(dataToPlot), np.max(dataToPlot)])

    linedistance = 2
    if plottype == 'angle' or plottype == 'isPhase':
        img = ax1.imshow(dataToPlot[:, :, 0], origin='lower',vmin=-np.pi, vmax=np.pi)
    else:
        img =  ax1.imshow(dataToPlot[:, :, 0], origin='lower',vmin=-1, vmax=1)
    ani = animation.FuncAnimation(plt.gcf(),
                                AnimateFullStatus, fargs=(dataToPlot, timevec, img, ax1, probepositions, lineseriesdata, currentPlot, linedistance, probecolors,ylim),
                                frames=nframes, interval=50)

    return ani

def AnimateFullStatus(frameNR, fullstatus,timevec, img, ax1, probepositions, lineseriesdata, currentPlot, linedistance, probecolors, ylim):
    # plt.figure(figsize=(10,10))
    img.set_data(fullstatus[ :, :, frameNR])
    #update time stamp in title
    ax1.set_title('Time =  ' + str(np.round(timevec[frameNR],3)))
  
    #  lineseriesdata[:][frameNR] = fullstatus[probepositions[:, 0], probepositions[:, 1], frameNR]
    # lineseriesdata[:][frameNR] += np.arange(len(probepositions)) * linedistance
    for ind, position in enumerate(probepositions):
        lineseriesdata[ind][frameNR] = fullstatus[position[0],position[1],frameNR]
        lineseriesdata[ind][frameNR] += ind * linedistance
    currentPlot.cla()
    currentPlot.set_ylim(ylim[0], len(probepositions * linedistance) +ylim[1])
    #currentPlot.set_yticks(np.arange(0,len(probepositions)*linedistance,linedistance),["O" for probe in probepositions])
    #ax1.tick_params(axis='y', colors=['red', 'black'], )  
    currentPlot.yaxis.set_visible(False)
    currentPlot.plot(lineseriesdata.T, linewidth =4)

    # Set x-ticks and x-tick labels at every 10th data point
    currentPlot.set_xticks(np.arange(0, len(timevec), 50))
    currentPlot.set_xticklabels(timevec[::50])
    for ind, line in enumerate(currentPlot.get_lines()):         
        line.set_color("black")
        #line.set_color(probecolors[ind])
        currentPlot.add_patch(plt.Rectangle((-2.5, (ind*linedistance)-0.25), 1, 0.5, facecolor='none',edgecolor=probecolors[ind],lw=8, clip_on=False))
    #currentPlot.get_lines()[3].set_color("red")

def plot_geodesic_distance_on_surface(vertices, faces, sensor_positions, path, chanInds, distance):
    """
    Plot the geodesic distance along the surface, highlighting the path and start/end points.
    
    vertices: The vertices of the surface.
    faces: The faces of the surface (triangular mesh).
    sensor_positions: Positions of the sensors.
    path: 3D coordinates of the points forming the geodesic path. Get from geoalg.geodesicDistance (from oygeodesic package)
    chanInds: Tuple of indices indicating the start and end points of the geodesic distance.
    distance: The geodesic distance value.
    """
    # Create the 3D surface
    surface = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Surface'
    )

    # Highlight the geodesic path
    path_coords = path  # Use path directly since it contains the coordinates
    geodesic_path = go.Scatter3d(
        x=path_coords[:, 0],
        y=path_coords[:, 1],
        z=path_coords[:, 2],
        mode='lines',
        line=dict(color='red', width=4),
        name='Geodesic Path'
    )

    # Highlight the start and end points
    start_end_points = go.Scatter3d(
        x=[vertices[chanInds[0], 0], vertices[chanInds[1], 0]],
        y=[vertices[chanInds[0], 1], vertices[chanInds[1], 1]],
        z=[vertices[chanInds[0], 2], vertices[chanInds[1], 2]],
        mode='markers+text',
        marker=dict(size=8, color=['blue', 'green'], symbol='circle'),
        text=[str(chanInds[0]), str(chanInds[1])],
        textposition='top center',
        name='Start/End Points'
    )

    # Highlight all vertex positions with indices
    vertices_plot = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        text=[str(i) for i in range(len(vertices))],
        textposition='top center',
        name='Vertices'
    )

    # Combine all plots
    plotData = [surface, vertices_plot, geodesic_path, start_end_points]

    # Layout
    layout = go.Layout(
        title=f'Geodesic Path on Surface (Distance: {distance:.2f})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True,
        width=900,
        height=900
    )

    # Create the figure
    fig = go.Figure(data=plotData, layout=layout)
    fig.show()


def plot_topomap(waveData, dataBucketName=None, dataInds=None,timeInds= None, trlInd = None, type = None):
    """Plots a topomap of the data
    Args:
        waveData: WaveData object
        dataBucketName: name of the data bucket to plot
        dataInds: tuple with indices of data to plot e.g.:(freqbin,trial, None, None). data after indexing needs to be posx_posy
    """
    if dataBucketName is None:
        dataBucketName = waveData.ActiveDataBucket
    data = waveData.get_data(dataBucketName)[dataInds]
    if type == "angle":
        data = np.angle(data)
        plt.set_cmap('twilight')
    elif type == "power":
        data = np.abs(data)
    
    pos_2d = waveData.get_2d_coordinates()
    if timeInds is None: #average over time
        data = np.mean(data, axis=-1)
    elif isinstance(timeInds, tuple): #average between timepoints
        data = np.mean(data[:, :, timeInds[0]:timeInds[1]], axis=-1)
    elif isinstance(timeInds, int): #single timepoint
        data = data[:, :, timeInds]

    if trlInd is None:#average over trials
        data = np.mean(data, axis=0)
    else:
        data= data[trlInd]   
        
    # Create a grid to interpolate the data
    grid_x, grid_y = np.mgrid[
        pos_2d[:, 0].min():pos_2d[:, 0].max():100j,
        pos_2d[:, 1].min():pos_2d[:, 1].max():100j
    ]

    # Interpolate the data
    grid_z = griddata(pos_2d, data, (grid_x, grid_y), method='cubic')

    # Plot the topomap
    if type == "angle":
        img = plt.imshow(grid_z.T, extent=(pos_2d[:, 0].min(), pos_2d[:, 0].max(), pos_2d[:, 1].min(), pos_2d[:, 1].max()), origin='lower', vmin=-np.pi, vmax=np.pi)
    else:
        img = plt.imshow(grid_z.T, extent=(pos_2d[:, 0].min(), pos_2d[:, 0].max(), pos_2d[:, 1].min(), pos_2d[:, 1].max()), origin='lower')
    plt.colorbar(img)
 
        

def plot_optical_flow(waveData, PlottingDataBucketName = None, UVBucketName = None, dataInds = None,plotangle = False, normVectorLength=False):
    """Plots the optical flow data
    Args:
        waveData: WaveData object
        PlottingDataBucketName: name of the data bucket to plot. No default. Needs to be set to the data used to calculate the optical flow
        UVBucketName: name of the data bucket with the uv data. Defaults to active data bucket
        dataInds: tuple with indices of data to plot e.g.:(freqbin,trial, None, None). Channels and time need to be None
    """
    if UVBucketName is None:
        UVBucketName = waveData.ActiveDataBucket
    if PlottingDataBucketName is None:
        raise ValueError('Please specify a data bucket to plot')

    # Ensure consistency
    hf.assure_consistency(waveData)
    hf.assure_consistency(waveData, PlottingDataBucketName)

    # Get the data
    UV = np.squeeze(waveData.DataBuckets[UVBucketName].get_data()[dataInds])
    
    if normVectorLength:
        UV = UV/ np.abs(UV)
    if plotangle:
        plotData = np.squeeze((np.angle(waveData.get_data(PlottingDataBucketName))[dataInds]))
        cmap = 'twilight'
    else:
        plotData = np.squeeze((np.real(waveData.get_data(PlottingDataBucketName))[dataInds]))
        cmap = 'copper'

    nFrames = plotData.shape[-1]  # time is the last dimension
    timevec = waveData.get_time()

    def AnimateFullStatus(frameNR, fullstatus,timevec):
        img.set_data(fullstatus[ :, :, frameNR])
        #update time stamp in title
        ax1.set_title('Time =  ' + "{:.2f}".format(timevec[frameNR]))
        barbs.set_UVC(-np.real(UV[ :, :, frameNR]), -np.imag(
            UV[ :, :, frameNR]))

    fig = plt.figure(figsize=(7, 5))
    ax1 = plt.subplot()
    ax1.grid(None)
    vmin, vmax = np.percentile(plotData, [5, 95])
    img = ax1.imshow(plotData[:, :, 0], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    barbs = ax1.quiver(-np.real(UV[:, :, 0]), -np.imag(UV[:, :, 0]))

    if plotangle:
        # Add a small subplot with the ring plot in the upper right corner
        ax2 = fig.add_axes([0.8, 0.7, 0.2, 0.2], polar=True)
        azimuths = np.radians(np.linspace(0, 360, 360))
        zeniths = np.linspace(0.4, 0.7, 30)
        r, theta = np.meshgrid(zeniths, azimuths)
        values = theta
        ax2.pcolormesh(theta, r, values, cmap='twilight')
        ax2.set_rgrids([0.4, 0.7], labels=[], angle=180)
        ax2.set_yticklabels([])
        ax2.grid(color='white')
        radian_multiples = [0, 0.5, 1, 1.5, 2]
        radians = [r * np.pi for r in radian_multiples]
        radian_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', '']
        ax2.set_xticks(radians)
        ax2.set_xticklabels(radian_labels)
    else:
        cbar = plt.colorbar(img)
        cbar.set_label('$\mu$V')

    ani = animation.FuncAnimation(plt.gcf(),
                                AnimateFullStatus, fargs=(plotData, timevec),
                                frames=nFrames-1, interval=100)
    return ani

def plot_optical_flow_polar_scatter(waveData, UVBucketName=None, directionalStabilityBucket=None, dataInds=None, windowSize=100):
    '''Plots a polar scatter plot of the UV data to show direction consistency.
    Args:
        waveData: WaveData object
        dataBucketName: name of the data bucket to plot. No default. Needs to be set to the data used to calculate the optical flow
        directionalStabilityBucket: name of the data bucket with the directional stability data. No default. Run OpticalFlow.calculate_directional_stability first
        dataInds: tuple with indices of data to plot e.g.:(freqbin,trial). Dimensions after indexing should be posx_posy_time
        windowSize: int, number of timepoints to average over for directional stability. 
    '''
   
    if UVBucketName is None:
        raise ValueError('Please specify a data bucket with UV information to plot')
    if directionalStabilityBucket is None:
        raise ValueError('Please specify a data bucket with the directional stability data')
    if dataInds is None:
        raise ValueError('Please specify the data indices to plot')

    # Get the data
    UV = waveData.DataBuckets[UVBucketName].get_data()[dataInds]
    averageVectors = waveData.DataBuckets[directionalStabilityBucket].get_data()[dataInds] 
    UnitVec = UV/ np.abs(UV)
    timevec = waveData.get_time()

    def AnimatePolarScatter(frameNr, UV, AverageVectors, WindowSize, timevec):
        currentUV = UV[:,:, frameNr:frameNr + WindowSize]
        offsetArray = np.stack((np.angle(currentUV).ravel(), np.abs(currentUV).ravel()), axis=1)
        scatterPlot.set_offsets(offsetArray)
        scatterPlot.set_sizes(offsetArray[:, 1] * 30)
        ax.set_title('Time ' + str(timevec[frameNr]))
        if(frameNr >= WindowSize):
            currentAverages = AverageVectors[:, :, (frameNr-WindowSize)+1].ravel()
            for idx, line in enumerate(lines):
                line.set_data([0, np.angle(currentAverages[idx])],
                            [0, np.abs(currentAverages[idx])])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_ylim(0, 1.1)

    # Adjust the spacing to make room for the title
    plt.subplots_adjust(top=0.80)
    dimx, dimy, nFrames = UV.shape
    pad = np.zeros((dimx, dimy, windowSize))
    paddedUnitVec = np.concatenate((pad, UnitVec, pad), axis=-1)

    # Create a color map for the x-direction
    cmap_x = plt.cm.get_cmap('RdBu', dimx)
    colors_x = cmap_x(np.linspace(0, 1, dimx))

    # Create a color map for the y-direction
    cmap_y = plt.cm.get_cmap('PuOr', dimy)
    colors_y = cmap_y(np.linspace(0, 1, dimy))

    # Combine the color maps
    colors = np.zeros((dimx, dimy, 4))
    for i in range(dimx):
        for j in range(dimy):
            colors[i, j, :3] = (colors_x[i, :3] + colors_y[j, :3]) / 2  # Average the RGB values
            colors[i, j, 3] = (colors_x[i, 3] + colors_y[j, 3]) / 2  # Average the alpha values

    # Reshape the colors to a 1D array
    colors_1d = colors.reshape(-1, 4)

    # Repeat the colors for each time point
    allcolors = np.repeat(colors_1d, windowSize, axis=0)

    # Adjust the alpha values over time
    alphasteps = np.linspace(0.1, 1, windowSize)
    alphasteps = np.repeat(alphasteps, dimx * dimy)
    allcolors[:, 3] = alphasteps

    scatterPlot = ax.scatter(np.angle(pad),
                            np.abs(pad), s=20, color=allcolors)
    lines = ax.plot([np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                    np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])],
                    [np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                    np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])], marker='o', linewidth=1.5, markersize=8)
    for idx, line in enumerate(lines):
        line.set_color(colors_1d[idx])

    ani = animation.FuncAnimation(plt.gcf(),
                                AnimatePolarScatter, fargs=(
                                    paddedUnitVec, averageVectors, windowSize, timevec),
                                frames=nFrames-1, interval=100)
    return ani

def plot_streamlines(UV, seedpoints):
    #uv = np.dstack((np.zeros((UV.shape[0], UV.shape[1])), UV))
    nx = UV.shape[0]
    ny = UV.shape[1]
    nz = UV.shape[2]
    u = np.real(UV)
    v = np.imag(UV)
    # origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2) #Puts origin at centre
    origin = (0, 0, 0)
    mesh = pv.ImageData(dims=(nx, ny, nz), spacing=(1, 1, 1), origin=origin)
    vectors = np.zeros((u.shape[0] * u.shape[1], 3))
    for tt in range(UV.shape[2]):
        # Arrange 2d vector-fields in space-time(added 3rd dimension = time)
        newarray = np.stack(
            (np.ravel(u[:, :, tt]) ** 3, np.ravel(v[:, :, tt]) ** 3, np.ones(u[:, :, tt].size))).T
        if tt == 0:
            vectors = newarray
        else:
            vectors = np.vstack((vectors, newarray))
    # Create polydata object
    mesh['vectors'] = vectors
    #create plotters
    pv.set_plot_theme("document")
    pdata = pv.vector_poly_data(mesh.points, vectors)
    sourcepoints = mesh.points[seedpoints.T.ravel()]
    wrappedPoints = pv.wrap(sourcepoints)
    stream = mesh.streamlines_from_source(
        wrappedPoints, 'vectors', integration_direction="forward",
        initial_step_length=0.5, max_step_length=0.5, min_step_length=0.5,
        interpolator_type="cell")

    for cellID in range(stream.n_cells):
        points = stream.get_cell(cellID).points

    # plotting vectors
    cpos = 'xy'
    # pdata.glyph(orient='vectors', scale='mag').plot()
    # pdata.glyph(orient='vectors', scale=False).plot()
    #pl.add_mesh(pdata)
    #pl.show()
    # plot all streamlines
    p = pv.Plotter(off_screen=True)  # Note the off_screen argument
    tube = stream.tube(radius=0.05)
    p.add_mesh(tube)
    return p

def plot_polar_histogram(waveData, DataBucketName, dataInds=None):
    """Plots a polar histogram of the directional stability data
    Args:
        waveData: WaveData object
        DataBucketName: name of the data bucket to plot from. Should be the result of OpticalFlow.calculate_directional_stability
        dataInds: tuple with indices of data to plot e.g.:(freqbin,trial). Dimensions after indexing should be posx_posy_time
    """
    waveData.set_active_dataBucket(DataBucketName)
    # Ensure consistency
    hf.assure_consistency(waveData)

    # Get the data
    Vectors = waveData.DataBuckets[DataBucketName].get_data()[dataInds]

    angles = np.angle(Vectors)
    magnitudes = np.abs(Vectors)
    angles = angles.ravel()
    magnitudes = magnitudes.ravel()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=36, weights=magnitudes, color='b', alpha=0.7)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.set_facecolor('white')
    ax.grid(True, color='black')

    return fig

# def plot_opticalFlow_velocity_profile(waveData, UVBucketName, dataInds=None):
#     """Plots a profile of the optical flow velocity
#     Args:
#         waveData: WaveData object
#         UVBucketName: name of the data bucket with the uv data. Defaults to active data bucket
#         dataInds: tuple with indices of data to plot e.g.:(freqbin,trial). Dimensions after indexing should be posx_posy_time
#     """
#     if UVBucketName is None:
#         UVBucketName = waveData.ActiveDataBucket
#     # Ensure consistency
#     hf.assure_consistency(waveData)

#     # Get the data
#     UV = waveData.DataBuckets[UVBucketName].get_data()[dataInds]


# #     return fig
