import numpy as np
def detect_end_systolic_from_mask_volume(mask_volume, criterion='area', verbose=False):
    """
    Detect the frame of end systolic from the mask volume.

    Args:
        mask_volume (np.ndarray): Myocardium binary mask volume. Should have shape (H, W, n_frames)
        verbose (bool, optional): Whether to print additional information. Default is False.

    Returns:
        int: Frame index of end systolic.    
    """

    # Detect the bounding box of the myocardium mask at each frames
    myo_bboxes = []
    H, W, n_frames = mask_volume.shape
    for frame_idx in range(n_frames):
        mask = mask_volume[:, :, frame_idx]
        if np.sum(mask) == 0:
            myo_bboxes.append((0, 0, 0, 0))
        else:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            myo_bboxes.append((rmin, rmax, cmin, cmax))

    if criterion == 'area':
        # Determine the area of the bounding box at each frame
        myo_areas = [(rmax - rmin) * (cmax - cmin) for rmin, rmax, cmin, cmax in myo_bboxes]
        myo_areas = [area / myo_areas[0] for area in myo_areas]
        if verbose:
            print(f"Myocardium areas: {myo_areas}")

        # Find the frame with the smallest area
        end_systolic_idx = np.argmin(myo_areas)
        if verbose:
            print(f"End systolic frame index: {end_systolic_idx}")
    elif criterion == 'min_x_max':
        # Determine the maximum x-coordinate of the bounding box at each frame
        myo_max_xs = [cmax for rmin, rmax, cmin, cmax in myo_bboxes]
        if verbose:
            print(f"Myocardium max x-coordinates: {myo_max_xs}")

        # Find the frame with the smallest max x-coordinate
        end_systolic_idx = np.argmin(myo_max_xs)
        if verbose:
            print(f"End systolic frame index: {end_systolic_idx}")
    elif criterion == 'min_x_max_y_max_sum':
        # Determine the sum of the maximum x-coordinate and maximum y-coordinate of the bounding box at each frame
        myo_max_xs_ys = [cmax + rmax for rmin, rmax, cmin, cmax in myo_bboxes]
        if verbose:
            print(f"Myocardium max x-coordinates + max y-coordinates: {myo_max_xs_ys}")

        # Find the frame with the smallest sum of max x-coordinate and max y-coordinate
        end_systolic_idx = np.argmin(myo_max_xs_ys)
        if verbose:
            print(f"End systolic frame index: {end_systolic_idx}")
    else:
        raise ValueError(f"Invalid criterion: {criterion}")
    
    return end_systolic_idx