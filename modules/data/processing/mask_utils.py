import numpy as np
def crop_image_given_bbox(image, bbox, oversize_action='zero_padding'):
    """
    Crop an image given a bounding box. If the bbox exceeds the image boundary, handle it based on oversize_action.

    :param image: 2D numpy ndarray with shape (H, W)
    :param bbox: Tuple of (ymin, ymax, xmin, xmax)
    :param oversize_action: Action to take if bbox exceeds image boundary. Supports 'zero_padding'.
    :return: Cropped image
    """
    # Ensure the bounding box is within the image dimensions
    H, W = image.shape[:2]
    ymin, ymax, xmin, xmax = bbox
    ymin, xmin = max(0, ymin), max(0, xmin)
    ymax, xmax = min(H, ymax), min(W, xmax)
    
    # Crop the image based on the bounding box
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    if oversize_action == 'zero_padding':
        # Check if padding is needed
        pad_top = max(0, -bbox[0])
        pad_bottom = max(0, bbox[1] - H)
        pad_left = max(0, -bbox[2])
        pad_right = max(0, bbox[3] - W)
        
        if pad_bottom > 0 or pad_right > 0 or pad_left > 0 or pad_top > 0:
            # Add zero padding to the cropped image
            if image.ndim == 2:
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            elif image.ndim == 3:
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
    
    return cropped_image