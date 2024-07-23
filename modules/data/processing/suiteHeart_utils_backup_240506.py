from modules.data.processing.contour_utils import rescale_contours_seq, generate_binary_masks_from_contour_seq
# def generate_binary_mask_from_two_SuiteHeart_contours_seq(contours, image_shape, implementation='skimg', centering=True):
#     """
#     Generates a binary mask from a sequence of two 2D contours.
    
#     Parameters:
#     contours (ndarray): A numpy array of shape (n_frames, n_contours) representing multiple sequences of contours.
#         Each element is a numpy array of shape (n_points, 2) representing a contour.
#         n_contours should be 2, otherwise the later contours will be ignored.
#     image_shape (tuple): 2-tuple representing the shape of the image.
    
#     Returns:
#     numpy.ndarray: Binary mask with shape (image_shape).
#     """
#     # mask = np.zeros(image_shape, dtype=np.uint8)
#     mask0 = generate_binary_masks_from_contour_seq(contours[:,0], image_shape, implementation, centering=centering)
#     mask1 = generate_binary_masks_from_contour_seq(contours[:,1], image_shape, implementation, centering=centering)
#     mask = np.logical_xor(mask0, mask1)
#     return mask
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

def get_SuiteHeart_dict_myocardium_rescaled_contours_and_masks(SuiteHeart_datum_dict, target_image_shape=(128,128), centering=True, check_slice_indices = None):
    """
    Get the myocardium contours and masks from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        tuple: A tuple containing the rescaled myocardium contours and masks.                
    """
    if check_slice_indices is None:
        check_slice_indices = list(range(lv_endo_n_slices))
    n_save_slices = len(check_slice_indices)

    lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    lv_epi_contours = SuiteHeart_datum_dict['lv_epi']
    lv_endo_n_frames, lv_endo_n_slices = lv_endo_contours.shape
    lv_epi_n_frames, lv_epi_n_slices = lv_epi_contours.shape

    lv_endo_contours_rescaled = np.zeros((lv_endo_n_frames, n_save_slices), dtype=object)
    lv_epi_contours_rescaled = np.zeros((lv_endo_n_frames, n_save_slices), dtype=object)
    
    lv_endo_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    lv_epi_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_epi_n_frames), dtype=np.uint8)
    myo_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)

    
    # Generate rescaled contours and masks
    H_cropped, W_cropped = target_image_shape
    for save_slice_idx, slice_idx in enumerate(check_slice_indices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            # lv_endo_contours_rescaled = lv_endo_contours[:, slice_idx].contour
            # lv_epi_contours_rescaled = lv_epi_contours[:, slice_idx].contour
            pass
        else:
            pixelSpacing = SuiteHeart_datum_dict['pixel_size'][0, slice_idx]
            # pixelSpacing = [0.5,0.5]
            # pixelSpacing = [1,1]
            # lv_endo_contours_rescaled_init_center = lv_endo_contours[0, slice_idx].contour.mean(axis=0)
            lv_endo_contours_rescaled[:, save_slice_idx] = rescale_contours_seq(
                [c.contour-1 for c in lv_endo_contours[:, slice_idx]], 
                original_spacing = pixelSpacing,
                target_spacing=(1,1),
                center=None)
            lv_epi_contours_rescaled[:, save_slice_idx] = rescale_contours_seq(
                [c.contour-1 for c in lv_epi_contours[:, slice_idx]],
                original_spacing = pixelSpacing, 
                target_spacing=(1,1),
                center=None)
            # print(f'{lv_endo_contours_rescaled=}')
            target_image_shape2 = [d*2 for d in target_image_shape]
            slice_myo_center = lv_endo_contours_rescaled[:, save_slice_idx][0].mean(axis=0) - 1
            slice_lv_endo_masks_rescaled = generate_binary_masks_from_contour_seq(
                lv_endo_contours_rescaled[:, save_slice_idx], 
                image_shape=target_image_shape2,centering=centering, ori_center=slice_myo_center)
            slice_lv_epi_masks_rescaled = generate_binary_masks_from_contour_seq(
                lv_epi_contours_rescaled[:, save_slice_idx], 
                image_shape=target_image_shape2,centering=centering, ori_center=slice_myo_center)
            # lv_endo_masks_rescaled[save_slice_idx] = slice_lv_endo_masks_rescaled[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            # lv_epi_masks_rescaled[save_slice_idx] = slice_lv_epi_masks_rescaled[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            lv_endo_masks_rescaled[save_slice_idx] = crop_image_given_bbox(slice_lv_endo_masks_rescaled, [int(slice_myo_center[1])-H_cropped//2,int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2,int(slice_myo_center[0])-W_cropped//2+W_cropped])
            lv_epi_masks_rescaled[save_slice_idx] = crop_image_given_bbox(slice_lv_epi_masks_rescaled, [int(slice_myo_center[1])-H_cropped//2,int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2,int(slice_myo_center[0])-W_cropped//2+W_cropped])
            myo_masks_rescaled[save_slice_idx] = np.logical_xor(lv_endo_masks_rescaled[save_slice_idx], lv_epi_masks_rescaled[save_slice_idx])
        
    
    # return lv_endo_contours_rescaled, lv_epi_contours_rescaled, lv_endo_masks_rescaled, lv_epi_masks_rescaled, myo_masks_rescaled
    return lv_endo_contours_rescaled, lv_epi_contours_rescaled, myo_masks_rescaled

from skimage.transform import resize
import numpy as np
def get_SuiteHeart_dict_resized_and_cropped_images(SuiteHeart_datum_dict, target_image_shape=(128,128)):
    """
    Get the resized and cropped images from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        np.ndarray: A numpy array containing the resized and cropped images.
    """
    images = SuiteHeart_datum_dict['raw_image']    
    lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    lv_epi_contours = SuiteHeart_datum_dict['lv_epi']

    n_frames, n_slices = images.shape
    images_rescaled = np.zeros_like(images)
    images_rescaled_cropped = np.zeros((n_slices, target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)
    for slice_idx in range(n_slices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            pass
        else:
            # Scaling parameters
            H, W = images[0, slice_idx].shape
            lv_endo_center = np.mean(lv_endo_contours[0, slice_idx].contour, axis=0) - 1
            pixelSpacing = SuiteHeart_datum_dict['pixel_size'][0, slice_idx]
            # pixelSpacing = [1,1]

            H_rescaled = int(H * pixelSpacing[0])
            W_rescaled = int(W * pixelSpacing[1])
            lv_endo_center_rescaled = lv_endo_center * pixelSpacing
            # lv_endo_center_rescaled = np.round(lv_endo_center * pixelSpacing)

            # Get the rescaled images
            for frame_idx in range(n_frames):
                images_rescaled[frame_idx, slice_idx] = resize(images[frame_idx, slice_idx], 
                    (H_rescaled, W_rescaled), 
                    anti_aliasing=True)

            # Crop the rescaled images
            for frame_idx in range(n_frames):
                # H_rescaled, W_rescaled = images_rescaled[frame_idx, slice_idx].shape
                H_crop, W_crop = target_image_shape
                H_crop_start = int(lv_endo_center_rescaled[1] - H_crop/2)
                W_crop_start = int(lv_endo_center_rescaled[0] - W_crop/2)
                images_rescaled_cropped[slice_idx, :, :, frame_idx] = images_rescaled[frame_idx, slice_idx][H_crop_start:H_crop_start+H_crop, W_crop_start:W_crop_start+W_crop]

    return images_rescaled_cropped


def get_SuiteHeart_dict_cropped_images(SuiteHeart_datum_dict, target_image_shape=(128,128)):
    
    images = SuiteHeart_datum_dict['raw_image']    
    lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    lv_epi_contours = SuiteHeart_datum_dict['lv_epi']

    n_frames, n_slices = images.shape
    H_cropped, W_cropped = target_image_shape
    myo_bboxes = []
    images_cropped = np.zeros((n_slices, target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)
    for slice_idx in range(n_slices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            myo_bboxes.append([])
        else:
            # Scaling parameters
            lv_endo_center = np.mean(lv_endo_contours[0, slice_idx].contour, axis=0) - 1
            # print(f'cine-{lv_endo_center=}')
            slice_myo_bbox = [int(lv_endo_center[1])-H_cropped//2,int(lv_endo_center[1])-H_cropped//2+H_cropped, int(lv_endo_center[0])-W_cropped//2,int(lv_endo_center[0])-W_cropped//2+W_cropped]
            myo_bboxes.append(slice_myo_bbox)
            # Crop the rescaled images
            for frame_idx in range(n_frames):
                # H_rescaled, W_rescaled = images_rescaled[frame_idx, slice_idx].shape
                
                # H_crop_start = int(lv_endo_center[1] - H_crop/2)
                # W_crop_start = int(lv_endo_center[0] - W_crop/2)
                # images_cropped[slice_idx, :, :, frame_idx] = images[frame_idx, slice_idx][H_crop_start:H_crop_start+H_crop, W_crop_start:W_crop_start+W_crop]
                # images_cropped[slice_idx, :, :, frame_idx] = images[frame_idx, slice_idx][int(lv_endo_center[1])-H_cropped//2:int(lv_endo_center[1])-H_cropped//2+H_cropped, int(lv_endo_center[0])-W_cropped//2:int(lv_endo_center[0])-W_cropped//2+W_cropped]
                images_cropped[slice_idx, :, :, frame_idx] = crop_image_given_bbox(images[frame_idx, slice_idx], slice_myo_bbox)

    return images_cropped, myo_bboxes


def get_SuiteHeart_dict_myocardium_masks(SuiteHeart_datum_dict, ori_image_shape=(128,128), target_image_shape=(128,128), centering=True, check_slice_indices = None):
    """
    Get the myocardium contours and masks from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        tuple: A tuple containing the rescaled myocardium contours and masks.                
    """
    if check_slice_indices is None:
        check_slice_indices = list(range(lv_endo_n_slices))
    n_save_slices = len(check_slice_indices)
    lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    lv_epi_contours = SuiteHeart_datum_dict['lv_epi']
    lv_endo_n_frames, lv_endo_n_slices = lv_endo_contours.shape
    lv_epi_n_frames, lv_epi_n_slices = lv_epi_contours.shape
    
    lv_endo_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    lv_epi_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    myo_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)

    H_cropped, W_cropped = target_image_shape
    # Generate rescaled contours and masks
    
    myo_bboxes = []
    for save_slice_idx, slice_idx in enumerate(check_slice_indices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_epi_contours[0, slice_idx], np.ndarray) and lv_epi_contours[0, slice_idx].size < 1):
            # lv_endo_contours_rescaled = lv_endo_contours[:, slice_idx].contour
            # lv_epi_contours_rescaled = lv_epi_contours[:, slice_idx].contour
            pass
        else:
            
            # target_image_shape2 = [d*1 for d in target_image_shape]
            slice_myo_center = lv_endo_contours[:, slice_idx][0].contour.mean(axis=0) - 1
            slice_myo_bbox = [int(slice_myo_center[1])-H_cropped//2,int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2,int(slice_myo_center[0])-W_cropped//2+W_cropped]
            slice_lv_endo_masks = generate_binary_masks_from_contour_seq(
                [c.contour-1 for c in lv_endo_contours[:, slice_idx]], 
                image_shape=ori_image_shape, centering=centering, ori_center=slice_myo_center)
            slice_lv_epi_masks = generate_binary_masks_from_contour_seq(
                [c.contour-1 for c in lv_epi_contours[:, slice_idx]], 
                image_shape=ori_image_shape, centering=centering, ori_center=slice_myo_center)
            # lv_endo_masks[save_slice_idx] = slice_lv_endo_masks[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            # lv_epi_masks[save_slice_idx] = slice_lv_epi_masks[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            lv_endo_masks[save_slice_idx] = crop_image_given_bbox(slice_lv_endo_masks, slice_myo_bbox)
            lv_epi_masks[save_slice_idx] = crop_image_given_bbox(slice_lv_epi_masks, slice_myo_bbox)
            myo_masks[save_slice_idx] = np.logical_xor(lv_endo_masks[save_slice_idx], lv_epi_masks[save_slice_idx])
            myo_bboxes.append(slice_myo_bbox)
        
    
    # return lv_endo_contours_rescaled, lv_epi_contours_rescaled, lv_endo_masks_rescaled, lv_epi_masks_rescaled, myo_masks_rescaled
    return myo_masks, myo_bboxes