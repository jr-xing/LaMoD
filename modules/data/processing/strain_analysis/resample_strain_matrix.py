import numpy as np
from scipy.interpolate import RegularGridInterpolator

def resample_strain_matrix_2d(strain_matrix, original_frame_timestamps, query_frame_timestamps):
    """
    Perform 2D interpolation on the strain matrix to resample it according to new frame timestamps,
    taking into account both spatial (sector indices) and temporal (frames) dimensions.
    
    Args:
    strain_matrix (np.ndarray): 2D numpy array of shape (n_sectors, n_frames).
    original_frame_timestamps (np.ndarray): Original timestamps corresponding to each frame (n_frames,).
    query_frame_timestamps (np.ndarray): New timestamps to interpolate to (n_query_frames,).
    
    Returns:
    np.ndarray: Resampled 2D numpy array of shape (n_sectors, n_query_frames).
    """
    # Get number of sectors
    n_sectors = strain_matrix.shape[0]
    
    # Generate sector indices as row indices
    sectors = np.arange(n_sectors)
    
    # Create a meshgrid for the original data coordinates
    sector_grid, time_grid = np.meshgrid(sectors, original_frame_timestamps, indexing='ij')
    
    # Create the interpolator function, allowing extrapolation
    interpolator = RegularGridInterpolator(
        (sectors, original_frame_timestamps), strain_matrix, 
        method='linear', bounds_error=False, fill_value=None
    )
    
    # Create a meshgrid for the query points
    query_sector_grid, query_time_grid = np.meshgrid(sectors, query_frame_timestamps, indexing='ij')
    
    # Stack the query grids for evaluation
    query_points = np.vstack((query_sector_grid.ravel(), query_time_grid.ravel())).T
    
    # Perform the interpolation
    resampled_values = interpolator(query_points).reshape(n_sectors, len(query_frame_timestamps))
    
    return resampled_values
