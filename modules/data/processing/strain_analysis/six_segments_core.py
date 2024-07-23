import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage.draw import polygon
from matplotlib import pyplot as plt
def six_segments_core(mask, basis, segper, origin, insertion, enable_offset=False, offset_degree=None):
    """
    Compute strain given displacement.
    
    Parameters:
    - mask: input mask to define segments
    - basis: number of base segments
    - segper: segments per basis
    - origin: origin point
    - insertion: insertion point
    
    Returns:
    - output: dict containing basis names, basis ID, and segment ID
    """
    
    # Define basis segment names
    if basis == 4:
        names = ['Anterior', 'Septal', 'Inferior', 'Lateral']
        offset = 90 * (np.pi / 180) if offset_degree is None else offset_degree * (np.pi / 180)
    elif basis == 6:
        names = ['Anterior', 'Anteroseptal', 'Inferoseptal', 'Inferior', 'Inferolateral', 'Anterolateral']
        offset = 60 * (np.pi / 180) if offset_degree is None else offset_degree * (np.pi / 180)
    else:
        names = [f'Seg{i}' for i in range(1, basis)]
        offset = (360/basis) * (np.pi / 180) if offset_degree is None else offset_degree * (np.pi / 180)
    Nseg = basis * segper
    
    # Initialize values
    Isz = mask.shape
    
    # Select anterior RV insertion point
    x0, y0 = insertion
    
    # Angle representing lateral extent of anterior segment
    # t0, r = np.arctan2(y0 - origin[1], x0 - origin[0])
    dx = x0 - origin[0]
    dy = y0 - origin[1]
    t0 = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)*10
    
    if enable_offset:
        t0 += offset
        # compute the location of shifted insertion point
        x0_shifted = origin[0] + r * np.cos(t0) / 10
        y0_shifted = origin[1] + r * np.sin(t0) / 10
        # update the insertion point
        insertion_shifted = np.array((x0_shifted, y0_shifted))
        # print(f'{r=}, {t0=}, {origin=}, {insertion=}, {insertion_shifted=}')
        # print(f"Insertion point shifted from {insertion} to: {insertion_shifted}")
    else:
        insertion_shifted = insertion
    
    # Additional counter-clockwise angles
    theta = t0 - np.linspace(0, 2*np.pi, Nseg+1)
    # theta_degree = np.degrees(theta) % 360
    # print(theta_degree)
    # theta = t0 - np.linspace(0, 2*np.pi, Nseg+1)
    
    # Cartesian coords representing angles
    x, y = r * np.cos(theta) + origin[0], r * np.sin(theta) + origin[1]
    # x, y = r*origin[0], r*origin[1]
    # x, y = origin[0], origin[1]
    
    # Label points
    segid = np.zeros(Isz, dtype='single')
    for k in range(Nseg):
        # Generate polygon points for the segment
        # print(origin[1], y[k], y[k+1], origin[1])
        rr, cc = polygon([origin[1], y[k], y[k+1], origin[1]],
                         [origin[0], x[k], x[k+1], origin[0]],
                         shape=Isz)
        
        # Create a temporary mask for the current polygon
        temp_mask = np.zeros(Isz, dtype=bool)
        temp_mask[rr, cc] = True
        
        # Apply the original mask to the temporary mask
        # This ensures we only update segid within the areas where the original mask is True
        temp_mask = np.logical_and(temp_mask, mask)
        
        # Update segid within the masked area
        segid[temp_mask] = k + 1
    
    basid = np.ceil(segid / segper)
    
    # Output
    output = {
        'insertion_shifted': insertion_shifted,
        'BasisNames': names,
        'BasisID': basid,
        'SegmentID': segid
    }
    
    return output

# # Example usage:
# mask = np.zeros((100, 100), dtype=bool)  # Example mask
# basis = 6  # or 4, depending on your segmentation scheme
# segper = 1  # number of segments per basis
# origin = (50, 50)  # Example origin
# insertion = (70, 50)  # Example insertion

# # Compute
# output = six_segments_core(mask, basis, segper, origin, insertion)
# print(output)
from scipy.interpolate import griddata
# from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
import numpy as np
def fillnan(data):
    # smartly fill nan by 2D interpolation
    # DONT from scipy.interpolate import interp2d
    # if data contains no nan, return data
    if not np.isnan(data).any():
        return data

    # create a meshgrid
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    xx, yy = np.meshgrid(x, y)
    # get the valid data
    x1 = xx[~np.isnan(data)]
    y1 = yy[~np.isnan(data)]
    newarr = data[~np.isnan(data)]
    # interpolate
    # newarr = interp2d(x1, y1, newarr, kind='linear')(xx, yy)
    newarr = griddata((x1, y1), newarr, (xx, yy), method='linear')
    # newarr = RectBivariateSpline(x1, y1, newarr)(x, y)
    return newarr
from skimage.morphology import dilation, disk, erosion
import numpy as np
from scipy.ndimage import label

def separate_rings(mask, structure=None):
    """
    Separates two non-intersecting concentric rings in a 2D boolean array by identifying
    connected components and assuming the smaller labeled area corresponds to the inner ring.
    
    Parameters:
        mask (numpy.ndarray): A 2D boolean array with two non-intersecting rings.
        structure (numpy.ndarray): A 2D array defining the neighborhood of each element.
        
    Returns:
        tuple: A tuple containing two 2D boolean arrays. The first array contains
               the inner/smaller ring, and the second contains the outer/larger ring.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=int)

    # Label connected components in the mask
    labeled_array, num_features = label(mask, structure)
    
    # Find the indices of the regions
    indices_1 = (labeled_array == 1)
    indices_2 = (labeled_array == 2)
    
    # Determine which is inner and which is outer by counting the true elements
    inner_ring = indices_1 if np.sum(indices_1) < np.sum(indices_2) else indices_2
    outer_ring = indices_2 if inner_ring is indices_1 else indices_1
    
    return inner_ring, outer_ring
def get_various_segmental_strain_data(strain_images, strain_mask, origin, insertion, enable_offset=[False, False, False], offset_degrees=[0,0,0], fillnanmat=True, layerid_map=None, generate_layerid_map=True):
    """
    Compute strain given displacement.
    layerid_map:
        0 - background
        1 - endocardium layer
        2 - middle layer
        3 - epicardium layer
    """
    if isinstance(enable_offset, bool):
        enable_offset = [enable_offset]*3
    segs4 = six_segments_core(
        mask = strain_mask,
        basis = 4,
        segper = 1,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[0],
        offset_degree = offset_degrees[0]
    )    
    segs6 = six_segments_core(
        mask = strain_mask,
        basis = 6,
        segper = 1,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[1],
        offset_degree = offset_degrees[1]
    )
    segs18 = six_segments_core(
        mask = strain_mask,
        basis = 6,
        segper = 3,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[2],
        offset_degree = offset_degrees[2]
    )
    
    Nfr = strain_images.shape[-1]
    # Nfr = 17
    CCs4 = np.zeros((int(max(segs4['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs4['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs4['SegmentID'] == seg_id, :].mean(axis=0)
        CCs4[seg_id-1, :] = CC_of_seg

    CCs6 = np.zeros((int(max(segs6['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs6['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs6['SegmentID'] == seg_id, :].mean(axis=0)
        CCs6[seg_id-1, :] = CC_of_seg

    CCs18 = np.zeros((int(max(segs18['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs18['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs18['SegmentID'] == seg_id, :].mean(axis=0)
        CCs18[seg_id-1, :] = CC_of_seg

    # fill nan if exist
    if fillnanmat:
        CCs4 = fillnan(CCs4)
        CCs6 = fillnan(CCs6)
        CCs18 = fillnan(CCs18)

    segmental_strain_dict = {
        'origin': origin,
        'insertion': insertion,
        'CCs4': CCs4,
        'CCs6': CCs6,
        'CCs18': CCs18,
        'Seg4': segs4,
        'Seg6': segs6,
        'Seg18': segs18,
    }

    layerid_name_pairs = [
        (1, 'endo'),
        (2, 'mid'),
        (3, 'epi')
    ]
    if layerid_map is not None or generate_layerid_map:
        if layerid_map is not None and generate_layerid_map:
            raise ValueError('layerid_map and generate_layerid_map cannot be both True')
        if generate_layerid_map:
            # strain_mask_dilated = dilation(strain_mask, disk(2))
            strain_mask_eroded = erosion(strain_mask, disk(1))
            inner_ring, outer_ring = separate_rings(np.logical_xor(strain_mask, strain_mask_eroded))
            # plt.figure();plt.imshow(strain_mask)
            # plt.figure();plt.imshow(strain_mask_eroded)
            # plt.figure();plt.imshow(inner_ring)
            # plt.figure();plt.imshow(outer_ring)
            layerid_map = np.zeros_like(strain_mask, dtype=int)
            layerid_map[strain_mask>0] = 2
            layerid_map[inner_ring>0] = 1
            layerid_map[outer_ring>0] = 3
        segmental_strain_dict['layerid_map'] = layerid_map
        for (layerid, layername) in layerid_name_pairs:
            for (seg_info_name, seg_info) in zip(['4', '6', '18'], [segs4, segs6, segs18]):
                curr_strain_matrix = np.zeros((int(max(seg_info['SegmentID'].flatten())), Nfr))
                for seg_id in range(1, int(max(seg_info['SegmentID'].flatten()))+1):
                    curr_segmental_layer_mask = np.logical_and(seg_info['SegmentID'] == seg_id, layerid_map == layerid)
                    curr_CC_of_seg = strain_images[curr_segmental_layer_mask, :].mean(axis=0)
                    curr_strain_matrix[seg_id-1, :] = curr_CC_of_seg
                segmental_strain_dict[f'CCs{seg_info_name}_{layername}'] = curr_strain_matrix
                # layer_mask = layerid_map == layerid
                # seg_info['SegmentID'][~layer_mask] = 0
                # seg_info['BasisID'][~layer_mask] = 0


    return segmental_strain_dict

def get_various_segmental_strain_data_backupo(strain_images, strain_mask, origin, insertion, enable_offset=[False, False, False], offset_degrees=[0,0,0], fillnanmat=True):
    """
    Compute strain given displacement.

    """
    if isinstance(enable_offset, bool):
        enable_offset = [enable_offset]*3
    segs4 = six_segments_core(
        mask = strain_mask,
        basis = 4,
        segper = 1,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[0],
        offset_degree = offset_degrees[0]
    )    
    segs6 = six_segments_core(
        mask = strain_mask,
        basis = 6,
        segper = 1,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[1],
        offset_degree = offset_degrees[1]
    )
    segs18 = six_segments_core(
        mask = strain_mask,
        basis = 6,
        segper = 3,
        origin = origin,
        insertion = insertion,
        enable_offset = enable_offset[2],
        offset_degree = offset_degrees[2]
    )
    
    Nfr = strain_images.shape[-1]
    # Nfr = 17
    CCs4 = np.zeros((int(max(segs4['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs4['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs4['SegmentID'] == seg_id, :].mean(axis=0)
        CCs4[seg_id-1, :] = CC_of_seg

    CCs6 = np.zeros((int(max(segs6['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs6['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs6['SegmentID'] == seg_id, :].mean(axis=0)
        CCs6[seg_id-1, :] = CC_of_seg

    CCs18 = np.zeros((int(max(segs18['SegmentID'].flatten())), Nfr))
    for seg_id in range(1, int(max(segs18['SegmentID'].flatten()))+1):
        CC_of_seg = strain_images[segs18['SegmentID'] == seg_id, :].mean(axis=0)
        CCs18[seg_id-1, :] = CC_of_seg

    # fill nan if exist
    if fillnanmat:
        CCs4 = fillnan(CCs4)
        CCs6 = fillnan(CCs6)
        CCs18 = fillnan(CCs18)


    return {
        'origin': origin,
        'insertion': insertion,
        'CCs4': CCs4,
        'CCs6': CCs6,
        'CCs18': CCs18,
        'Seg4': segs4,
        'Seg6': segs6,
        'Seg18': segs18,        
    }