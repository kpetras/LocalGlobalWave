#%%
import Modules.Utils.WaveData as wd
import Modules.Utils.HelperFuns as hf
from sklearn.manifold import MDS, Isomap
from sklearn.neighbors import NearestNeighbors
import numpy as np
import gdist
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import Rbf
from scipy.linalg import svd
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from matplotlib.path import Path
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import joblib
import platform
import multiprocessing
import trimesh

#%%

#Sensor spatial arrangement_____________________________
def regularGrid(data, pos):
    distMat = distance_matrix(pos,pos)
    data.set_distMat(distMat)
    data.HasRegularLayout = True
    data.log_history(["Distance matrix", "distmattype","regularGrid"])

def create_surface_from_points(data, type = 'channels', num_points=1000, plotting = False):
    '''Makes a surface from the electrode positions. 
    If type is 'channels', the surface is made from the convex hull of the electrode positions. 
    If type is 'sphere', the surface is made from the convex hull of the electrode positions projected onto a sphere. 
    If type is 'headshape', the surface is made from the headshape points. #Todo: Implement!
    Originally, the surface looked weird and the bottom was flat. the geodesic distances were likely wrong.
    To fix that, I added some points below the electrodes to make a proper bottom. Probaly not the most elegant solution, 
    but it works for now. Keep in mind that this means the surface has more vertices than there are electrodes.'''
    
    positions = data.get_channel_positions()
    centroid = np.mean(positions, axis=0)
    # set radius to mean distance from centroid to positions
    radius = np.mean(np.linalg.norm(positions - centroid, axis=1))        
    if type == 'sphere':
        #project electrodes onto sphere
        # get centroid of positions
        # make more points
        indices = np.arange(0, num_points, dtype=float) + 0.5
        phi = (np.arccos(1 - 2*indices/num_points))
        theta = (np.pi * (1 + 5**0.5) * indices)
        x, y, z = (radius * np.cos(theta) * np.sin(phi) + centroid[0], 
                  radius * np.sin(theta) * np.sin(phi) + centroid[1], 
                  radius * np.cos(phi) + centroid[2])
        sphere_positions = np.column_stack([x, y, z])
        positions = np.concatenate([positions, sphere_positions], axis=0)
    else:
        #the surface from the convex hull of the electrode positions makes a flat bottom and that messes 
        #up the geodesic distance calculation. For lack of a better idea, I just add some random points
        #well below the contacts to prevent this issue. If you have a better idea, please let me know
        bottom_point = centroid.copy()
        bottom_point[2] -= 2*radius  # Adjust this value to make sure it's below the electrodes
        # Calculate the additional four points
        before_point = bottom_point.copy()
        before_point[0] -= radius/4

        after_point = bottom_point.copy()
        after_point[0] += radius/4

        left_point = bottom_point.copy()
        left_point[1] -= radius/4

        right_point = bottom_point.copy()
        right_point[1] += radius/4
        positions = np.concatenate([positions, [bottom_point, before_point, after_point, left_point, right_point]], axis=0)
    #sometimes some electrode position is a bit too far in to make it onto the surface. 
    # so we make a hull defined by the outermost positions and then just move whichever ones are not on it
    # again, if you have a better idea (not too difficult), please let me know  :)  
    try:
        tri = Delaunay(positions)
        # Find original electrodes that are not on the convex hull
        hull = tri.convex_hull
        all_hull_vertices = np.unique(hull.ravel())
        hull_points = positions[all_hull_vertices]
        
        original_electrodes = data.get_channel_positions()
        for i in range(len(original_electrodes)):            
            if i not in all_hull_vertices:
                direction = original_electrodes[i] - centroid
                positions[i] = original_electrodes[i] + direction*0.01  # move tiny bit outward
    except:
        pass  # if this doesn't work, just skip it and hope for the best
    # Create points
    points = vtk.vtkPoints()
    for position in positions:
        points.InsertNextPoint(position)
    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    # Create a surface from the points
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polydata)
    delaunay.Update()
    # Extract the surface
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(delaunay.GetOutputPort())
    surface_filter.Update()
    # Get the surface data
    PolySurface = surface_filter.GetOutput()
    # Get the vertices and faces
    vertices = np.array([PolySurface.GetPoint(i) for i in range(PolySurface.GetNumberOfPoints())])
    faces = np.array([PolySurface.GetCell(i).GetPointIds().GetId(j) for i in range(PolySurface.GetNumberOfCells()) for j in range(3)])

    Surface = [vertices,faces]
    if plotting:
        import plotly.graph_objects as go
        faces = faces.reshape(-1, 3)
        fig = go.Figure(data=[go.Mesh3d(x=vertices[:, 0],
                                        y=vertices[:, 1],
                                        z=vertices[:, 2],
                                        i=faces[:, 0],
                                        j=faces[:, 1],
                                        k=faces[:, 2],
                                        color='lightpink',
                                        opacity=0.50)])
        # Add scatter plot for channel locations
        channel_positions = data.get_channel_positions()
        fig.add_trace(go.Scatter3d(x=channel_positions[:, 0],
                                   y=channel_positions[:, 1],
                                   z=channel_positions[:, 2],
                                   mode='markers',
                                   marker=dict(size=2, color='blue')))
        
        # Add scatter plot for vertex locations
        fig.add_trace(go.Scatter3d(x=vertices[:, 0],
                                   y=vertices[:, 1],
                                   z=vertices[:, 2],
                                   mode='markers',
                                   marker=dict(size=2, color='red')))

        fig.update_layout(scene=dict(xaxis=dict(nticks=4, range=[np.min(vertices),np.max(vertices)],),
                             yaxis=dict(nticks=4, range=[np.min(vertices),np.max(vertices)],),
                             zaxis=dict(nticks=4, range=[np.min(vertices),np.max(vertices)],),
                             aspectmode='cube'),
                             width=700,
                             margin=dict(r=20, l=10, b=10, t=10))
        
        
        fig.show()   

    return Surface, PolySurface
    
def distance_along_surface(data, Surface, tolerance = 0.01, get_extent = False, plotting = False):
    """    Calculate the distance between contacts along a cortical surface.
    Parameters:
    data: waveData object
    Surface (tuple): A tuple containing the vertices and faces of the surface.
    tolerance (float, optional): The maximum distance a channel can be from a vertex to be considered a match. Defaults to 0.01.
    get_extent (bool or list of tuples, optional): If True, calculate the maximum extent along the x and y directions. Check the plot to make sure the correct channels are used!
     Alternatively, provide the indices of the channels to use (the most frontal one, most posterior one, most left, most right). Defaults to False.
    plotting (bool, optional): If True, create a 3D scatter plot of the channel positions. Defaults to False. Only makes sense when get_extent is True. 
    Look at the plot though, because I am not certain this will always work... Might have to specify indices of the channels to use for extent
        
    This calculates the geodesic distance between each pair of channels along the cortical surface using tvb-gdist.
    If `get_extent` is True, also adds extentGeodesic to the waveData (get is with waveData.get_extentGeodesic). Useful to calculate propagation speeds    
    """
    #some very un-elegant changing of data types because the original ones do not work
    channel_positions = np.array(data.get_channel_positions())
    vert = Surface[0].astype(np.float64)
    
    vert = Surface[0].astype(np.float64)
    faces = np.asarray([Surface[1]], dtype='int32').squeeze()
    kdtree = KDTree(vert)

    # Find exact vertex matches
    vertInd = []
    #sometimes we added extra points to the surface to fix the bottom of it (create_surface_from_points). 
    # If the surface has more vertices than there are channels, we ignore the extra ones by looping ove channel positions,
    # instead of over vertices
    for position in channel_positions:
        # Use KD-tree query to find the closest point in the vert
        distance, index = kdtree.query(position)
        if distance < .00001:  # Set your desired tolerance value
            vertInd.append(index)
    if len(vertInd)< len(channel_positions):
        print('not all contacts are assigned. Trying again with closest, rather than exact, match')
        vertInd = []
        for position in channel_positions:
            distance, index = kdtree.query(position, distance_upper_bound=tolerance)
            if np.isinf(distance):
                continue  # No approximate match within the specified tolerance
            vertInd.append(index)
    if len(vertInd)< len(channel_positions):
        raise Exception('not all contacts are assigned. Double check your positions and surface or try increasing tolerance')       

    seed = np.asarray(vertInd,dtype='int32')
    # # calculate distance
    #the next bit might take long. 
    triangles = np.reshape(faces, (int(len(faces)/3), 3))
    distPairs = gdist.distance_matrix_of_selected_points(vert,triangles, seed)

    DistMat = np.zeros([len(seed),len(seed)],dtype = np.float64)   
    #find the closest contact for each contact
    for ind,contact in enumerate(seed):
        #for convenience later on, make contacts x contacts matrix of distance values
        DistMat[ind,:] = distPairs[contact,seed].toarray()



    if get_extent:
        if isinstance(get_extent, list) and all(isinstance(i, tuple) for i in get_extent):
            # Use the provided indices to get the correct values out of DistMat
            max_distance_x = DistMat[get_extent[0][0], get_extent[0][1]]
            max_distance_y = DistMat[get_extent[1][0], get_extent[1][1]]
            min_x_index = get_extent[0][0]
            max_x_index = get_extent[0][1]
            min_y_index = get_extent[1][0]
            max_y_index = get_extent[1][1]
        else:
            # Get indices of channels with maximum and minimum values along x and y directions
            min_x_index = np.argmin(channel_positions[:, 0])
            max_x_index = np.argmax(channel_positions[:, 0])
            min_y_index = np.argmin(channel_positions[:, 1])
            max_y_index = np.argmax(channel_positions[:, 1])

            # Use these indices to get the correct values out of DistMat
            max_distance_x = DistMat[min_x_index, max_x_index]
            max_distance_y = DistMat[min_y_index, max_y_index]

        data._extentGeodesic = (max_distance_x, max_distance_y)

    if plotting:
        # Create a 3D scatter plot of all channel positions
        fig = go.Figure(data=[go.Scatter3d(
            x=channel_positions[:, 0],
            y=channel_positions[:, 1],
            z=channel_positions[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='blue',  # all channel positions are blue
            )
        )])

        # Add the channels used for max_distance_x and max_distance_y
        fig.add_trace(go.Scatter3d(
            x=channel_positions[[min_x_index, max_x_index], 0],
            y=channel_positions[[min_x_index, max_x_index], 1],
            z=channel_positions[[min_x_index, max_x_index], 2],
            mode='markers',
            marker=dict(
                size=6,
                color='red',  # channels for max_distance_x are red
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=channel_positions[[min_y_index, max_y_index], 0],
            y=channel_positions[[min_y_index, max_y_index], 1],
            z=channel_positions[[min_y_index, max_y_index], 2],
            mode='markers',
            marker=dict(
                size=6,
                color='green',  # channels for max_distance_y are green
            )
        ))

        # Print the max distances
        print(f"Max distance along x: {max_distance_x}")
        print(f"Max distance along y: {max_distance_y}")

        fig.show()

    data.set_distMat(DistMat)
    data.HasRegularLayout = False
    data.log_history(["Distance matrix", "distmattype","surfDist"])
    print('distance along surface calculated')



def find_midline_channels(channel_positions, tolerance=0.1):
    """
    Finds the midline channels in a given set of channel positions.
    Parameters:
    channel_positions (numpy.ndarray): A 2D array or 3D of channel positions (only 2dimensions are actually used)
    tolerance (float, optional): maximum distance a channel can be from the midline to be considered a midline channel. Defaults to 0.1.
    Returns:
    tuple: Two numpy arrays containing the indices of the sagittal and coronal midline channels respectively.

    Function mirrors the channel positions over the y-axis and x-axis, then uses a KDTree to find the plane of symmetry.
    Channels close to the midlines (within the given tolerance) are then returned.
    Plots for sanity check. Make sure you look at the plot because I am not certain this will always work...
    """
    # Mirror the sensor positions over the y-axis and x-axis
    mirrored_positions_y = channel_positions.copy()
    mirrored_positions_y[:, 0] = -mirrored_positions_y[:, 0]
    mirrored_positions_x = channel_positions.copy()
    mirrored_positions_x[:, 1] = -mirrored_positions_x[:, 1]

    # Find the plane of symmetry
    kdtree = KDTree(channel_positions)
    _, indices_y = kdtree.query(mirrored_positions_y)
    _, indices_x = kdtree.query(mirrored_positions_x)
    sagittal_plane = np.median(channel_positions[indices_y, 0])
    coronal_plane = np.median(channel_positions[indices_x, 1])

    # Find the channels close to the midlines
    sagittal_channels = np.where(np.abs(channel_positions[:, 0] - sagittal_plane) < tolerance)[0]
    coronal_channels = np.where(np.abs(channel_positions[:, 1] - coronal_plane) < tolerance)[0]
    #plot to check
    fig, ax = plt.subplots()
    ax.scatter(channel_positions[:, 0], channel_positions[:, 1], label='All channels')
    ax.scatter(channel_positions[sagittal_channels, 0], channel_positions[sagittal_channels, 1], label='Sagittal channels', color='red')
    ax.scatter(channel_positions[coronal_channels, 0], channel_positions[coronal_channels, 1], label='Coronal channels', color='green')
    ax.axvline(sagittal_plane, color='red', linestyle='--', label='Sagittal plane')
    ax.axhline(coronal_plane, color='green', linestyle='--', label='Coronal plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()
    return sagittal_channels, coronal_channels


def plot_distance_along_surface(waveData):
    """Plot the distance matrix on the surface with distance as color"""

    DistMat = waveData.get_distMat()
    channel_positions = np.array(waveData.get_channel_positions())  # The 3D positions of the channels
    DistMat = waveData.get_distMat()  # The distance matrix you calculated

    # Create a 3D scatter plot for the points
    scatter = go.Scatter3d(
        x=channel_positions[:, 0],
        y=channel_positions[:, 1],
        z=channel_positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='black',  # Color of the points
        )
    )

    # Create a list to hold the line shapes
    line_shapes = []

    # Create a colormap
    cmap = plt.get_cmap('coolwarm')

    # Normalize the distances to the range [0, 1]
    norm = plt.Normalize(DistMat.min(), DistMat.max())

    # Iterate over the distance matrix and create a line for each pair of points
    for i in range(len(channel_positions)):
        for j in range(i+1, len(channel_positions)):
            if DistMat[i, j] > 0:  # If there is a distance between the points
                # Define the endpoints of the line
                x0, y0, z0 = channel_positions[i]
                x1, y1, z1 = channel_positions[j]

                # Calculate the color based on the distance
                rgba_color = cmap(norm(DistMat[i, j]))

                # Create a line shape
                line = go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(
                        color='rgb'+str(rgba_color[:3]),  # Use the rgb color
                        width=2
                    )
                )
                line_shapes.append(line)

    # Combine the scatter plot and line shapes
    plotData = [scatter] + line_shapes
    min_distance = DistMat.min()
    max_distance = DistMat.max()
    coolwarm = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
    # Create a dummy scatter plot with a color scale and a colorbar
    dummy_scatter = go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None],  # Two points
        mode='markers',
        marker=dict(
            size=0,  # No markers
            color=[min_distance, max_distance],  # Color range
            colorscale=coolwarm,  # The same color scale used for the lines
            colorbar=dict(title='Distance')  # The colorbar
        )
    )

    # Add the dummy scatter plot to the data
    plotData = [scatter, dummy_scatter] + line_shapes

    # Define the layout of the plot
    layout = go.Layout(
        title='3D EEG Sensor Positions and Connections',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=False,
        width=800,  # Set the width of the figure
        height=800  # Set the height of the figure
    )

    # Create the figure
    fig = go.Figure(data=plotData, layout=layout)

    # Show the figure
    fig.show()

def project_to_sphere(sensor_positions, radius=1, plot=True):
    """
    Project the 3D sensor positions onto a sphere of a given radius and optionally plot the result.

    Parameters:
    sensor_positions (np.array): An array of sensor positions in 3D space.
    radius (float): The radius of the sphere onto which the sensors will be projected.
    plot (bool): If True, plot the projected positions.

    Returns:
    np.array: An array of projected sensor positions on the sphere.
    """
    projected_positions = np.zeros_like(sensor_positions)

    for i, pos in enumerate(sensor_positions):
        norm = np.linalg.norm(pos)
        if norm == 0:
            continue
        projected_positions[i] = radius * pos / norm

    if plot:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        # Draw the sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        z = radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="b", alpha=0.1)

        # Plot the original positions
        ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2], color="r", label="Original")
        # Plot the projected positions
        ax.scatter(projected_positions[:, 0], projected_positions[:, 1], projected_positions[:, 2], color="g", label="Projected")

        ax.legend()
        plt.show()

    return projected_positions

def calculate_Euclidean_distance(waveData):
    # Get the 2D coordinates
    coords = waveData.get_2d_coordinates()

    # Calculate the distance matrix
    dist_matrix = np.sqrt(((coords[:, None] - coords) ** 2).sum(axis=2))

    return dist_matrix

def distmat_to_2d_coordinates_MDS(waveData):
    #Do multidimensional scaling to translate distMat into 2D (arbitrary) cartesian coordinates, while preserving relative distances
    mds = MDS(random_state=0, dissimilarity='precomputed')
    # Get the embeddings
    X_transform = mds.fit_transform(waveData.get_distMat())
    # scale and rotate the 2d result to match the 3d 
    X= np.copy(X_transform)
    Y = waveData.get_channel_positions()[:, :2]
    #center and normalize distances (just to make data easier to deal with, gets rescaled below)
    # Translate all the points to their centroids
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    # Scale the points to have root mean square distance from the centroid of 1
    X /= np.sqrt((X ** 2).sum(axis=1)).mean()
    Y /= np.sqrt((Y ** 2).sum(axis=1)).mean()
    # Compute SVD
    U, _, Vt = svd(np.dot(X.T, Y))
    # The rotation matrix R
    R = np.dot(U, Vt)
    X_transform = np.dot(X, R)
    # Compute the scaling factor
    max_geodesic_distance = np.max(waveData.get_distMat())
    distance_in_transformed_space = np.max(X_transform[:,0]) - np.min(X_transform[:,0])
    scaling_factor = max_geodesic_distance / distance_in_transformed_space
    # Apply the scaling factor
    X_transform *= scaling_factor
    waveData.set_2D_coordinates(X_transform)
    waveData.set_2D_coordinates(X_transform)
    waveData.log_history(["distmat_to_2d_coordinates", "projectionmethod","MDS"])

def distmat_to_2d_coordinates_Isomap(waveData):
    # Do Isomap to translate distMat into 2D (arbitrary) cartesian coordinates, while preserving relative distances
    iso = Isomap(n_components=2, metric='precomputed')
    # Get the embeddings
    X_transform = iso.fit_transform(waveData.get_distMat())
    # Rotate the 2d result to match the 3d 
    X = np.copy(X_transform)
    Y = waveData.get_channel_positions()[:, :2]
    # Center the coordinates
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    # Compute SVD
    U, _, Vt = svd(np.dot(X.T, Y))
    # The rotation matrix R
    R = np.dot(U, Vt)
    X_transform = np.dot(X, R)
    waveData.set_2D_coordinates(X_transform)
    waveData.log_history(["distmat_to_2d_coordinates", "projectionmethod","Isomap"])

def is_regular_grid_2d(distMat, tolerance = 0.001):
    """
    Check if a grid of positions is regular.
    Parameters:
        grid (list of tuples): a grid of positions
        Tolerance : relative to distance
    Returns:
        bool: True if the grid is regular, False otherwise
    """
    # Get the number of rows and columns in the grid
    if len(distMat) != len(distMat[0]):
        return False
    size = int(np.sqrt(len(distMat)))
    adj_dist = distMat[0][1]
    for i in range(size):
        for j in range(size):
            if abs(i-j) == 1:
                if not np.isclose(distMat[i][j],adj_dist,atol=0, rtol=tolerance):
                    return False
    return True

def interpolate_pos_to_grid_process_trial(k, data, indices,distances, grid_x_shape, idw_power):
    grid_z = np.empty((grid_x_shape[0] * grid_x_shape[1], data.shape[2]), dtype=data.dtype)
    for j in range(data.shape[2]):
        z = data[k, indices, j]
        grid_z[:, j] = idw_interpolation(z, distances, idw_power=idw_power)
    return grid_z.reshape((grid_x_shape[0], grid_x_shape[1], data.shape[2]))

def interpolate_pos_to_grid(waveData, numGridBins=10, dataBucketName = "", return_mask= False, mask_stretching = False):
    '''Interpolate positions to a regular grid
    Parameters 
    ----------
    waveData : WaveData object
    numGridBins : int
        how many bins to tile the space along the x-coordinate (y-coordinate is scaled accordingly)
 
        Returns
        -------
        new_positions : array
        adds a new data bucket to waveData called InterpolatedData.
        InterpolatedData is the original data, interpolated to the regular grid xy'''


    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    #temporarily reshape data to trl_chan_time
    data = waveData.DataBuckets[dataBucketName].get_data()
    oldshape = data.shape
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    currentDims = currentDimord.split("_")
    desiredDimord = "trl_chan_time"
    desiredDims = desiredDimord.split("_")
    hasBeenReshaped, data =  hf.force_dimord(data, currentDimord , desiredDimord)
     
    if waveData.has_data_bucket(dataBucketName + "Interpolated"):
        print("WARNING: InterpolatedData already exists, replacing it...")

    # Get the 2D coordinates
    pos_2d = waveData.get_2d_coordinates()

    # Get the min and max for x and y
    x_min, y_min = np.min(pos_2d, axis=0)
    x_max, y_max = np.max(pos_2d, axis=0)

    # Calculate the range and step size
    range_x = x_max - x_min
    range_y = y_max - y_min
    step_size = max(range_x, range_y) / (numGridBins - 1)
    if mask_stretching:
        stretch = step_size * 2 
    else:
        stretch = 0
    # Calculate the new min and max for x and y to create a square grid
    x_min_new = x_min - (step_size * numGridBins - range_x) / 2
    x_max_new = x_min_new + step_size * numGridBins
    y_min_new = y_min - (step_size * numGridBins - range_y) / 2
    y_max_new = y_min_new + step_size * numGridBins

    # Create the grid
    x_range = np.linspace(x_min_new, x_max_new, numGridBins)
    y_range = np.linspace(y_min_new, y_max_new, numGridBins)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    # Plot the original positions
    plt.figure(figsize=(10, 10))
    plt.scatter(pos_2d[:,0], pos_2d[:,1], color='blue', label='Original positions')

    # Plot the grid points
    plt.scatter(grid_x, grid_y, color='red', s=2, label='Grid points')

    plt.legend()
    plt.show()

    # Calculate the convex hull of the 2D positions
    hull = ConvexHull(pos_2d)
    hull_path = Path(pos_2d[hull.vertices])

    # Create a mask for grid points inside the convex hull
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    inside_mask = hull_path.contains_points(grid_points, radius=stretch)
    print('stretch: ' + str(stretch))

    # Reshape the mask to the grid shape
    inside_mask_grid = inside_mask.reshape(grid_x.shape)

    # Plot the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(inside_mask_grid, extent=(x_min_new, x_max_new, y_min_new, y_max_new), origin='lower', cmap='gray', alpha=0.5)
    plt.plot(pos_2d[hull.vertices, 0], pos_2d[hull.vertices, 1], 'r--', lw=2)
    plt.plot(pos_2d[hull.vertices[0], 0], pos_2d[hull.vertices[0], 1], 'ro')
    plt.scatter(pos_2d[:,0], pos_2d[:,1], color='blue', label='Original positions')
    plt.legend()
    plt.show()

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(pos_2d)
    distances, indices = nbrs.kneighbors(grid_points)
    
    if platform.system() == 'Linux':
        with multiprocessing.Pool() as pool:
            all_grid_z = pool.starmap(interpolate_pos_to_grid_process_trial, 
                                      [(k, data, indices,distances, grid_x.shape, 2) 
                                        for k in range(data.shape[0])])
    else:
        all_grid_z = joblib.Parallel(n_jobs=joblib.cpu_count())(
                                        joblib.delayed(interpolate_pos_to_grid_process_trial)
                                        (k, data, indices, distances, grid_x.shape, 2) 
                                        for k in range(data.shape[0]))
    all_grid_z = np.stack(all_grid_z, axis=0)

    # make fake channames for the interpolated data
    channames = []
    for i in range(all_grid_z.shape[1]):
        for j in range(all_grid_z.shape[2]):
            channames.append(str(i) + "_" + str(j))
    if hasBeenReshaped:
        trl_dim = currentDims.index("trl")
        new_trl_dim = desiredDims.index("trl")
        all_grid_z = np.reshape(all_grid_z, (*oldshape[:trl_dim+1],*all_grid_z.shape[new_trl_dim+1:]))
        new_Dimord = ("_").join(currentDims[:trl_dim+1]) + "_" + ("_").join(desiredDims[new_trl_dim+1:])
    else:
        new_Dimord = desiredDimord
    new_Dimord = new_Dimord.replace("chan", "posx_posy")
    InterpolatedData = wd.DataBucket(all_grid_z, dataBucketName + "Interpolated", new_Dimord, channames)
    waveData.add_data_bucket(InterpolatedData)

    if return_mask:
        return grid_x, grid_y, inside_mask_grid
    else:
        return grid_x, grid_y

def idw_interpolation(z, distances, idw_power=2):
    # Calculate weights using the distances and normalize them
    weights = 1.0 / distances**idw_power
    weights /= weights.sum(axis=1, keepdims=True)
    # Multiply each weight by corresponding z value and sum them to get the interpolated values
    interpolated_z = np.sum(weights * z, axis=1)

    if np.iscomplexobj(z):
        interpolated_z = interpolated_z.astype(np.complex128)

    return interpolated_z

def best_fit_sphere(coord_3d):
    A = np.hstack((2 * coord_3d, np.ones((coord_3d.shape[0], 1))))
    f = np.sum(coord_3d ** 2, axis=1)
    C, residuals, rank, singval = np.linalg.lstsq(A, f, rcond=None)

    # The center of the sphere:
    center = C[:3]
    # The radius of the sphere:
    radius = np.sqrt(np.sum(C[:3] ** 2) + C[3])
    return center, radius

def apply_mask(waveData, mask, dataBucketName, overwrite = True, maskValue = 0., storeMask = False):
    ''' Apply a spatial mask to the data in a data bucket. Sets all values outside the mask to maskValue'''
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    #temporarily reshape data to trl_chan_time
    data = waveData.DataBuckets[dataBucketName].get_data()
    oldshape = data.shape
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    currentDims = currentDimord.split("_")
    if len(mask.shape)==2:
        desiredDimord = "trl_posx_posy_time"
    else:
        desiredDimord = "trl_chan_time"
    desiredDims = desiredDimord.split("_")
    hasBeenReshaped, data =  hf.force_dimord(data, currentDimord , desiredDimord)
    data[:,~mask,:] = maskValue
    if hasBeenReshaped:
        trl_dim = currentDims.index("trl")
        new_trl_dim = desiredDims.index("trl")
        data = np.reshape(data, oldshape)
    if overwrite:
        waveData.DataBuckets[dataBucketName]._data = data
    else:
        maskedData = wd.DataBucket(data, dataBucketName + "Masked", currentDimord, waveData.get_channel_names())
        waveData.add_data_bucket(maskedData)
    if storeMask:
        maskBucket = wd.DataBucket(mask, "Mask", "_".join(desiredDims[1:-1]), waveData.get_channel_names())
        waveData.add_data_bucket(maskBucket)
    waveData.set_active_dataBucket(dataBucketName)#just to make sure that mask does not become the active dataBucket

def interpolate_spherical_spline_2d(waveData, resolution=10, scalePos=1000, function='multiquadric', n_jobs=-1):
    '''
    [KP] Something doesn't seem quite right about this. Fix it later.
    Interpolate positions to a regular grid using spherical spline interpolation, and
    return data on a 2D grid. This function uses parallel processing.
    Parameters 
    ----------
    waveData : WaveData object
    scalePos : float
        scaling factor to convert from whatever unit pos has to mm (default assumes pos is in meters)
    resolution : int
        the resolution of the grid in mm
    function : str
        the radial basis function, multiquadric by default
    n_jobs : int
        the number of jobs to run in parallel. -1 means using all processors.
    '''
    pos_3d = waveData.get_channel_positions() * scalePos
    data = waveData.get_active_data()
    
    # Calculate the best fit sphere center and radius from the channel positions
    center, radius = best_fit_sphere(pos_3d)

    # Shift electrode positions to center around the best fit sphere center
    pos_3d -= center

    r, theta, phi = cart2sph(pos_3d[:,0], pos_3d[:,1], pos_3d[:,2])

    theta_range = np.arange(np.min(theta), np.max(theta), np.radians(resolution))
    phi_range = np.arange(np.min(phi), np.max(phi), np.radians(resolution))

    grid_theta, grid_phi = np.meshgrid(theta_range, phi_range)

    # Convert to Cartesian coordinates, taking into account the radius of the best fit sphere
    grid_x, grid_y, grid_z = sph2cart(np.ones_like(grid_theta) * radius, grid_theta, grid_phi)

    interpolated_data = np.empty((data.shape[0], grid_theta.shape[0], grid_theta.shape[1], data.shape[2]))
    for j in range(data.shape[2]):
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(interpolate_time_point)(pos_3d, data[k,:,j], function, grid_x, grid_y, grid_z) for k in range(data.shape[0]))
        interpolated_data[:,:,:,j] = np.array(results)

    channames = [f'{i}_{j}' for i in range(grid_z.shape[1]) for j in range(grid_z.shape[2])]
    InterpolatedData = wd.DataBucket(interpolated_data,"InterpolatedDataSphere","trial_posx_posy_time",channames)
    
    # if waveData.has_data_bucket("InterpolatedData"):
    #     print("WARNING: InterpolatedData already exists, replacing it")
    waveData.add_data_bucket(InterpolatedData)

def interpolate_time_point(pos_3d, data_point, function, grid_x, grid_y, grid_z):
    rbf = Rbf(pos_3d[:,0], pos_3d[:,1], pos_3d[:,2], data_point, function=function)
    return rbf(grid_x, grid_y, grid_z)

def fake_grid(waveData, surface):
    import vtk
    import numpy as np

    # 'surface' contains the vtkPolyData surface

    # Generate texture coordinates
    texture_coords = vtk.vtkFloatArray()
    texture_coords.SetNumberOfComponents(2)
    texture_coords.SetName("Texture Coordinates")

    # Compute texture coordinates using spherical mapping
    texture_coords.SetNumberOfTuples(surface.GetNumberOfPoints())
    for i in range(surface.GetNumberOfPoints()):
        point = surface.GetPoint(i)
        theta = np.arctan2(point[1], point[0])  # Compute azimuthal angle
        phi = np.arccos(point[2] / np.linalg.norm(point))  # Compute polar angle

        u = (theta + np.pi) / (2 * np.pi)  # Normalize azimuthal angle to [0, 1]
        v = phi / np.pi  # Normalize polar angle to [0, 1]

        texture_coords.SetTuple2(i, u, v)

    # Assign texture coordinates to the surface
    surface.GetPointData().SetTCoords(texture_coords)

    # Now you have the UV mapping coordinates for each point on the surface

    # Access the texture coordinates
    texture_coords_array = surface.GetPointData().GetTCoords()

    # Get the unwrapped surface vertices and faces
    unwrapped_vertices = np.array([surface.GetPoint(i) for i in range(surface.GetNumberOfPoints())])
    unwrapped_faces = np.array([surface.GetCell(i).GetPointIds().GetId(j) for i in range(surface.GetNumberOfCells()) for j in range(3)])

    # Store the unwrapped surface in a variable
    unwrapped_surface = [unwrapped_vertices, unwrapped_faces]

    return unwrapped_surface  

#some helper functions 
   
def cart2sph(x, y, z):
    '''
    Convert Cartesian coordinates to spherical coordinates.
    Assumes that x, y, z are all arrays of the same length.
    Returns three arrays: azimuthal angle, polar angle, and radius.
    '''
    azimuth = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(z / radius)
    return azimuth, polar, radius

def sph2cart(azimuth, polar, radius):
    '''
    Convert spherical coordinates to Cartesian coordinates.
    Assumes that azimuth, polar, radius are all arrays of the same length.
    Returns three arrays: x, y, z.
    '''
    x = radius * np.sin(polar) * np.cos(azimuth)
    y = radius * np.sin(polar) * np.sin(azimuth)
    z = radius * np.cos(polar)
    return x, y, z



