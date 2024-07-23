import numpy as np
from modules.data.processing.contour_utils import rescale_contours_seq, generate_binary_masks_from_contour_seq
from modules.data.processing.contour_utils import rescale_multiple_contours_seq, generate_binary_mask_from_two_contours_seq


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

def rescale_DENSEanalysis_dict_myocardium_contours_and_masks(DENSEanalysis_datum_dict, target_image_shape=(128,128), centering=True, rescale_center=False):
    """
    Rescales the myocardium contours in a SuiteHeart dictionary.

    Args:
        DENSEanalysis_datum_dict (dict): A DENSEanalysis datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        dict: A SuiteHeart dictionary with rescaled myocardium contours.
    """
    # Use the DENSE slice PixelSpacing to rescale the contours
    # So the spatial resolution of the rescaled contours will be 1x1
    DENSE_slice_PixelSpacing = DENSEanalysis_datum_dict['DENSEInfo']['PixelSpacing']
    lv_endo_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][:,0]
    lv_epi_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][:,1]
    lv_endo_contours_rescaled = rescale_contours_seq(
        [c-1 for c in lv_endo_contours], 
        original_spacing=DENSE_slice_PixelSpacing,
        target_spacing=(1,1),
        rescale_center=rescale_center)
    lv_epi_contours_rescaled = rescale_contours_seq(
        [c-1 for c in lv_epi_contours],
        original_spacing=DENSE_slice_PixelSpacing, 
        target_spacing=(1,1),
        rescale_center=rescale_center)
    
    target_image_shape2 = [d*2 for d in target_image_shape]
    H_cropped, W_cropped = target_image_shape
    # myo_center = np.round(lv_endo_contours_rescaled[0].mean(axis=0))
    
    myo_center = np.round(lv_endo_contours_rescaled[0].mean(axis=0)-1)

    # print(f'{myo_center=}, {lv_endo_contours_rescaled[0].shape=}')
    # myo_center = np.array([myo_center[1], myo_center[0]])
    # myo_center = lv_endo_contours_rescaled[0].mean(axis=0) - 1
    # print(f'{myo_center=}')
    lv_endo_masks_rescaled2 = generate_binary_masks_from_contour_seq(
        lv_endo_contours_rescaled, 
        image_shape=target_image_shape, centering=centering, ori_center=myo_center)
    lv_epi_masks_rescaled2 = generate_binary_masks_from_contour_seq(
        lv_epi_contours_rescaled, 
        image_shape=target_image_shape, centering=centering, ori_center=myo_center)
    lv_endo_masks_rescaled = crop_image_given_bbox(lv_endo_masks_rescaled2, [int(myo_center[1])-H_cropped//2, int(myo_center[1])-H_cropped//2+H_cropped, int(myo_center[0])-W_cropped//2, int(myo_center[0])-W_cropped//2+W_cropped])
    lv_epi_masks_rescaled = crop_image_given_bbox(lv_epi_masks_rescaled2, [int(myo_center[1])-H_cropped//2, int(myo_center[1])-H_cropped//2+H_cropped, int(myo_center[0])-W_cropped//2, int(myo_center[0])-W_cropped//2+W_cropped])
    rescaled_myocardium_masks = np.logical_xor(lv_endo_masks_rescaled, lv_epi_masks_rescaled)
    # rescaled_myocardium_masks = np.logical_xor(lv_endo_masks_rescaled2, lv_epi_masks_rescaled2)
    return lv_endo_contours_rescaled, lv_epi_contours_rescaled, rescaled_myocardium_masks




from skimage.transform import resize
import numpy as np
def get_DENSEanalysis_dict_resized_and_cropped_images(DENSEanalysis_datum_dict, target_image_shape=(128,128)):
    """
    Get the resized and cropped images from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        np.ndarray: A numpy array containing the resized and cropped images.
    """
    images = DENSEanalysis_datum_dict['ImageInfo']['Mag'] # H, W, T
    lv_endo_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][0,0]
    # lv_epi_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][:,1]

    H, W, n_frames = images.shape
    images_rescaled = np.zeros_like(images)
    images_rescaled_cropped = np.zeros((target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)

    lv_endo_center = np.mean(lv_endo_contours, axis=0) - 1
    # print(f'{lv_endo_center=}')
    pixelSpacing = DENSEanalysis_datum_dict['DENSEInfo']['PixelSpacing']

    H_rescaled = int(H * pixelSpacing[0])
    W_rescaled = int(W * pixelSpacing[1])
    # lv_endo_center_rescaled = lv_endo_center * pixelSpacing
    lv_endo_center_rescaled = np.round(lv_endo_center * pixelSpacing)
    # print(f'{lv_endo_center_rescaled=}')
    # print(f'{lv_endo_center_rescaled=}')
    # Get the rescaled images
    for frame_idx in range(n_frames):
        images_rescaled = resize(images[...,frame_idx], 
            (H_rescaled, W_rescaled), 
            anti_aliasing=True)
    
        # Crop the rescaled images
        H_crop, W_crop = target_image_shape
        H_crop_start = int(lv_endo_center_rescaled[1] - H_crop/2)
        W_crop_start = int(lv_endo_center_rescaled[0] - W_crop/2)
        images_rescaled_cropped[..., frame_idx] = crop_image_given_bbox(images_rescaled, [H_crop_start, H_crop_start+H_crop, W_crop_start, W_crop_start+W_crop])
    # print(f'{images.shape=}')
    # print(f'{images_rescaled.shape=}')
    return images_rescaled_cropped

def get_DENSEanalysis_dict_cropped_images(DENSEanalysis_datum_dict, target_image_shape=(128,128)):
    images = DENSEanalysis_datum_dict['ImageInfo']['Mag'] # H, W, T
    lv_endo_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][0,0]
    H, W, n_frames = images.shape
    # lv_endo_center = np.round(np.mean(lv_endo_contours, axis=0))
    lv_endo_center = np.mean(lv_endo_contours, axis=0) - 1
    # print(f'{lv_endo_center=}')

    H_cropped, W_cropped = target_image_shape

    images_cropped = np.zeros((target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)
    cropping_bbox = [int(lv_endo_center[1])-H_cropped//2, int(lv_endo_center[1])-H_cropped//2+H_cropped, int(lv_endo_center[0])-W_cropped//2, int(lv_endo_center[0])-W_cropped//2+W_cropped]
    for frame_idx in range(n_frames):
        images_cropped[..., frame_idx] = crop_image_given_bbox(images[..., frame_idx],
            cropping_bbox)
        
    return images_cropped, cropping_bbox

def get_DENSEanalysis_dict_cropped_masks(DENSEanalysis_datum_dict, ori_image_shape=(128,128), target_image_shape=(48, 48), centering=True):
    
    lv_endo_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][:,0]
    lv_epi_contours = DENSEanalysis_datum_dict['ROIInfo']['Contour'][:,1]
    
    # target_image_shape2 = [d*1 for d in target_image_shape]
    H_cropped, W_cropped = target_image_shape
    myo_center = np.round(lv_endo_contours[0].mean(axis=0)) 
    # myo_center = np.round(lv_endo_contours[0].mean(axis=0)-1) 
    myo_center = lv_endo_contours[0].mean(axis=0) - 1
    # print(f'{myo_center=}')
    lv_endo_masks = generate_binary_masks_from_contour_seq(
        [c-1 for c in lv_endo_contours], 
        image_shape=ori_image_shape, centering=centering, ori_center=myo_center)
    lv_epi_masks = generate_binary_masks_from_contour_seq(
        [c-1 for c in lv_epi_contours], 
        image_shape=ori_image_shape, centering=centering, ori_center=myo_center)
    cropping_bbox = [int(myo_center[1])-H_cropped//2, int(myo_center[1])-H_cropped//2+H_cropped, int(myo_center[0])-W_cropped//2, int(myo_center[0])-W_cropped//2+W_cropped]
    lv_endo_masks_cropped = crop_image_given_bbox(lv_endo_masks, cropping_bbox)
    lv_epi_masks_cropped = crop_image_given_bbox(lv_epi_masks, cropping_bbox)
    
    myocardium_masks = np.logical_xor(lv_endo_masks_cropped, lv_epi_masks_cropped)
    
    return myocardium_masks, cropping_bbox