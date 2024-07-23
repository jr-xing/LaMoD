from modules.data.processing.contour_utils import rescale_contours_seq, generate_binary_masks_from_contour_seq
import matplotlib.pyplot as plt
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

def list_to_2D_ndarray(list_data):
    """
    Convert a list of data into a 2D numpy ndarray.
    Args:
        list_data (list): A list of data.
    Returns:
        np.ndarray: A 2D numpy ndarray containing the data.
    """        
    data = np.zeros((len(list_data), 1), dtype=object)
    for i in range(len(list_data)):
        data[i, 0] = list_data[i]
    return data


def get_contour(datum):
    if type(datum) is dict:
        return datum['contour']
    else:
        return datum.contour

def get_SuiteHeart_dict_myocardium_rescaled_contours_and_masks(SuiteHeart_datum_dict, target_image_shape=(128,128), centering=False, check_slice_indices=None, rescale_method='pixel_spacing', rescale_ratio=0.6):
    """
    Get the myocardium contours and masks from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        tuple: A tuple containing the rescaled myocardium contours and masks.                
    """
    

    # lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    # lv_epi_contours = SuiteHeart_datum_dict['lv_epi']
    lv_endo_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_endo']) if type(SuiteHeart_datum_dict['lv_endo']) is list else SuiteHeart_datum_dict['lv_endo']
    lv_epi_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_epi']) if type(SuiteHeart_datum_dict['lv_epi']) is list else SuiteHeart_datum_dict['lv_epi']
    lv_rv_insertion_points = list_to_2D_ndarray(SuiteHeart_datum_dict['rv_insertion']) if type(SuiteHeart_datum_dict['rv_insertion']) is list else SuiteHeart_datum_dict['rv_insertion']
    # mat_has_single_slice_flag = False
    if type(SuiteHeart_datum_dict['lv_endo']) is list:
        mat_has_single_slice_flag = True
    else:
        mat_has_single_slice_flag = False
    # if len(lv_endo_contours.shape) == 1 and len(lv_epi_contours.shape) == 1:    
    #     images = np.expand_dims(images, axis=1)
    #     lv_endo_contours = np.expand_dims(lv_endo_contours, axis=1)
    #     lv_epi_contours = np.expand_dims(lv_epi_contours, axis=1)
    #     mat_has_single_slice_flag = True
    # elif len(lv_endo_contours.shape) == 2 and len(lv_epi_contours.shape) == 2:
    #     mat_has_single_slice_flag = False
    # else:
    #     raise ValueError('The input SuiteHeart_datum_dict has invalid shapes.')
    
    lv_endo_n_frames, lv_endo_n_slices = lv_endo_contours.shape
    lv_epi_n_frames, lv_epi_n_slices = lv_epi_contours.shape

    if check_slice_indices is None:
        check_slice_indices = list(range(lv_endo_n_slices))
    n_save_slices = len(check_slice_indices)

    lv_endo_contours_rescaled = np.zeros((lv_endo_n_frames, n_save_slices), dtype=object)
    lv_epi_contours_rescaled = np.zeros((lv_endo_n_frames, n_save_slices), dtype=object)
    
    lv_endo_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    lv_epi_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_epi_n_frames), dtype=np.uint8)
    myo_masks_rescaled = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)

    # rv_insertion_points_rescaled = np.zeros_like(lv_rv_insertion_points)
    rv_insertion_points_rescaled = np.zeros(lv_endo_n_slices, dtype=object)
    lv_epi_contours_rescaled_cropped = np.zeros_like(lv_epi_contours_rescaled)
    lv_endo_contours_rescaled_cropped = np.zeros_like(lv_epi_contours_rescaled)

    # Generate rescaled contours and masks
    H_cropped, W_cropped = target_image_shape
    for save_slice_idx, slice_idx in enumerate(check_slice_indices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            # lv_endo_contours_rescaled = lv_endo_contours[:, slice_idx].contour
            # lv_epi_contours_rescaled = lv_epi_contours[:, slice_idx].contour
            pass
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], np.ndarray) and lv_endo_contours[fidx, slice_idx].size < 1 for fidx in range(lv_endo_n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], np.ndarray) and lv_epi_contours[fidx, slice_idx].size < 1 for fidx in range(lv_epi_n_frames)]):
            pass
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], list) and len(lv_endo_contours[fidx, slice_idx]) < 1 for fidx in range(lv_endo_n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], list) and len(lv_epi_contours[fidx, slice_idx]) < 1 for fidx in range(lv_epi_n_frames)]):
            pass
        else:
            if rescale_method == 'pixel_spacing':
                pixelSpacing = SuiteHeart_datum_dict['pixel_size'][slice_idx] if type(SuiteHeart_datum_dict['pixel_size']) is list else SuiteHeart_datum_dict['pixel_size'][0, slice_idx]
                scale_factors = pixelSpacing
            elif rescale_method == 'fill':
                height_ratio = target_image_shape[0] / (np.max(get_contour(lv_endo_contours[0, slice_idx])-1, axis=0)[1] - np.min(get_contour(lv_endo_contours[0, slice_idx])-1, axis=0)[1])# / target_image_shape[0]
                width_ratio = target_image_shape[1] / (np.max(get_contour(lv_endo_contours[0, slice_idx])-1, axis=0)[0] - np.min(get_contour(lv_endo_contours[0, slice_idx])-1, axis=0)[0])# / target_image_shape[1]
                # if type(lv_endo_contours[0,0]) == dict:
                #     height_ratio = target_image_shape[0] / (np.max(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[1] - np.min(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[1])# / target_image_shape[0]
                #     width_ratio = target_image_shape[1] / (np.max(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[0] - np.min(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[0])# / target_image_shape[1]
                # else:
                #     height_ratio = target_image_shape[0] /  (np.max(lv_endo_contours[0, slice_idx].contour-1, axis=0)[1] - np.min(lv_endo_contours[0, slice_idx].contour-1, axis=0)[1])
                #     width_ratio = target_image_shape[1] / (np.max(lv_endo_contours[0, slice_idx].contour-1, axis=0)[0] - np.min(lv_endo_contours[0, slice_idx].contour-1, axis=0)[0])
                # height_ratio = (np.max(lv_endo_contours[0, slice_idx][0]-1, axis=0)[:,1] - np.min(lv_endo_contours[:, slice_idx][0]-1, axis=0)[:,1]) / target_image_shape[0]
                # width_ratio = (np.max(lv_endo_contours[:, slice_idx]-1, axis=0)[:,0] - np.min(lv_endo_contours[:, slice_idx]-1, axis=0)[:,0]) / target_image_shape[1]
                # fill_ratio = 0.7
                # scale_factors = [height_ratio*fill_ratio, width_ratio*fill_ratio]
                # scale_factors = [width_ratio*fill_ratio, height_ratio*fill_ratio]
                scale_factors = np.array([min(height_ratio, width_ratio)*rescale_ratio, min(height_ratio, width_ratio)*rescale_ratio])
                # print(f{scale_factors=})
            elif rescale_method == 'rescale':
                scale_factors = np.array([rescale_ratio, rescale_ratio])
            else:
                raise ValueError('Invalid rescale_method.')
            rv_insertion_point = get_contour(lv_rv_insertion_points[0, slice_idx][0])
            # pixelSpacing = [0.5,0.5]
            # pixelSpacing = [1,1]
            # lv_endo_contours_rescaled_init_center = lv_endo_contours[0, slice_idx].contour.mean(axis=0)
            if type(lv_endo_contours[0,0]) == dict:            
                slice_lv_endo_contours_list = [c['contour']-1 for c in lv_endo_contours[:, slice_idx]]
                slice_lv_epi_contours_list = [c['contour']-1 for c in lv_epi_contours[:, slice_idx]]
            else:
                slice_lv_endo_contours_list = [c.contour-1 for c in lv_endo_contours[:, slice_idx]]
                slice_lv_epi_contours_list = [c.contour-1 for c in lv_epi_contours[:, slice_idx]]                

            lv_endo_contours_rescaled[:, save_slice_idx] = rescale_contours_seq(
                slice_lv_endo_contours_list, 
                original_spacing = scale_factors,
                target_spacing=(1,1),
                center=None,
                rescale_center=True)
            lv_epi_contours_rescaled[:, save_slice_idx] = rescale_contours_seq(
                slice_lv_epi_contours_list,
                original_spacing = scale_factors, 
                target_spacing=(1,1),
                center=None,
                rescale_center=True)
            #print(f'{slice_lv_endo_contours_list[0].mean(axis=0)=}')
            #print(f'{slice_lv_epi_contours_list[0].mean(axis=0)=}')
            #print(f'{lv_endo_contours_rescaled[0,save_slice_idx].mean(axis=0)=}')
            #print(f'{lv_endo_contours_rescaled[0,save_slice_idx].mean(axis=0)=}')

            #print(f'{slice_lv_endo_contours_list[0].shape=}')
            #print(f'{slice_lv_epi_contours_list[0].shape=}')
            # target_image_shape2 = [d*2 for d in target_image_shape]
            target_image_shape2 = [256, 256]
            slice_myo_endo_center = np.floor(lv_endo_contours_rescaled[:, save_slice_idx][0].mean(axis=0))
            slice_myo_epi_center = np.floor(lv_epi_contours_rescaled[:, save_slice_idx][0].mean(axis=0))
            #print(f'{slice_myo_endo_center=}, {slice_myo_epi_center=}')
            slice_myo_center = slice_myo_endo_center
            slice_lv_endo_masks_rescaled = generate_binary_masks_from_contour_seq(
                lv_endo_contours_rescaled[:, save_slice_idx], 
                image_shape=target_image_shape2,centering=centering, ori_center=slice_myo_endo_center)
            slice_lv_epi_masks_rescaled = generate_binary_masks_from_contour_seq(
                lv_epi_contours_rescaled[:, save_slice_idx], 
                image_shape=target_image_shape2,centering=centering, ori_center=slice_myo_epi_center)
            # fig, axs = plt.subplots(1,2, figsize=(4,2))
            # axs[0].plot(slice_lv_endo_contours_list[0][:,0],slice_lv_endo_contours_list[0][:,1], 'r')
            # axs[0].plot(slice_lv_epi_contours_list[0][:,0],slice_lv_epi_contours_list[0][:,1], 'b')
            # axs[0].plot(slice_lv_endo_contours_list[0].mean(axis=0)[0],slice_lv_endo_contours_list[0].mean(axis=0)[1], 'rx')
            # axs[0].plot(slice_lv_epi_contours_list[0].mean(axis=0)[0],slice_lv_epi_contours_list[0].mean(axis=0)[1], 'bx')
            # axs[1].plot(lv_endo_contours_rescaled[0,save_slice_idx][:,0],lv_endo_contours_rescaled[0,save_slice_idx][:,1], 'r')
            # axs[1].plot(lv_epi_contours_rescaled[0,save_slice_idx][:,0],lv_epi_contours_rescaled[0,save_slice_idx][:,1], 'b')
            # axs[1].plot(lv_endo_contours_rescaled[0,save_slice_idx].mean(axis=0)[0],lv_endo_contours_rescaled[0,save_slice_idx].mean(axis=0)[1], 'rx')
            # axs[1].plot(lv_epi_contours_rescaled[0,save_slice_idx].mean(axis=0)[0],lv_epi_contours_rescaled[0,save_slice_idx].mean(axis=0)[1], 'bx')
            
            #axs[0].set_ylim(80, 150);axs[0].set_xlim(80, 180)
            #axs[1].set_ylim(80, 150);axs[1].set_xlim(80, 180)
            #fig.suptitle(f'{save_slice_idx=}')
            # lv_endo_masks_rescaled[save_slice_idx] = slice_lv_endo_masks_rescaled[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            # lv_epi_masks_rescaled[save_slice_idx] = slice_lv_epi_masks_rescaled[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            H_crop_start = int(slice_myo_center[1] - H_cropped/2)
            W_crop_start = int(slice_myo_center[0] - W_cropped/2)
            # rv_insertion_point_rescaled = (rv_insertion_point - slice_myo_center) * scale_factors + slice_myo_center
            rv_insertion_point_rescaled = (rv_insertion_point - get_contour(lv_endo_contours[0,save_slice_idx]).mean(axis=0)) * scale_factors + slice_myo_center
            # print(f'{slice_myo_center=}, {get_contour(lv_endo_contours_rescaled[0, save_slice_idx][0]).mean(axis=0)=}')
            for frame_idx in range(lv_endo_n_frames):
                # for slice_idx in range(lv_endo_n_slices):
                lv_epi_contours_rescaled_cropped[frame_idx, save_slice_idx] = \
                    lv_epi_contours_rescaled[frame_idx, save_slice_idx] - np.array([W_crop_start, H_crop_start])
                    #np.array([W_crop_start, H_crop_start]) + lv_endo_contours_rescaled[frame_idx, save_slice_idx] - slice_myo_center* scale_factors
                lv_endo_contours_rescaled_cropped[frame_idx, save_slice_idx] = \
                    lv_endo_contours_rescaled[frame_idx, save_slice_idx] - np.array([W_crop_start, H_crop_start])
                    #np.array([W_crop_start, H_crop_start]) + lv_epi_contours_rescaled[frame_idx, save_slice_idx] - slice_myo_center* scale_factors
            rv_insertion_points_rescaled[save_slice_idx] = rv_insertion_point_rescaled - np.array([W_crop_start, H_crop_start])
            lv_endo_masks_rescaled[save_slice_idx] = crop_image_given_bbox(slice_lv_endo_masks_rescaled, [H_crop_start, H_crop_start+H_cropped, W_crop_start, W_crop_start+W_cropped])
            lv_epi_masks_rescaled[save_slice_idx] = crop_image_given_bbox(slice_lv_epi_masks_rescaled, [H_crop_start, H_crop_start+H_cropped, W_crop_start, W_crop_start+W_cropped])
            myo_masks_rescaled[save_slice_idx] = np.logical_xor(lv_endo_masks_rescaled[save_slice_idx], lv_epi_masks_rescaled[save_slice_idx])
        
    if mat_has_single_slice_flag == True:
        lv_endo_contours_rescaled = lv_endo_contours_rescaled[:,0]
        lv_epi_contours_rescaled = lv_epi_contours_rescaled[:,0]
        lv_endo_contours_rescaled_cropped = lv_endo_contours_rescaled_cropped[:,0]
        lv_epi_contours_rescaled_cropped = lv_epi_contours_rescaled_cropped[:,0]
        lv_endo_masks_rescaled = lv_endo_masks_rescaled[0]
        lv_epi_masks_rescaled = lv_epi_masks_rescaled[0]
        myo_masks_rescaled = myo_masks_rescaled[0]
        rv_insertion_points_rescaled = rv_insertion_points_rescaled[0]
    # return lv_endo_contours_rescaled, lv_epi_contours_rescaled, lv_endo_masks_rescaled, lv_epi_masks_rescaled, myo_masks_rescaled
    # return lv_endo_contours_rescaled, lv_epi_contours_rescaled, myo_masks_rescaled, rv_insertion_points_rescaled
    return lv_endo_contours_rescaled_cropped, lv_epi_contours_rescaled_cropped, myo_masks_rescaled, rv_insertion_points_rescaled

from skimage.transform import resize
import numpy as np
def get_SuiteHeart_dict_resized_and_cropped_images(SuiteHeart_datum_dict, target_image_shape=(128,128), rescale_method='pixel_spacing', rescale_ratio=0.6):
    """
    Get the resized and cropped images from a SuiteHeart dictionary.

    Args:
        SuiteHeart_datum_dict (dict): A SuiteHeart datum dictionary.
        target_image_shape (tuple, optional): The target shape of the image. Defaults to (128, 128).

    Returns:
        np.ndarray: A numpy array containing the resized and cropped images.
    """
    # images = SuiteHeart_datum_dict['raw_image']    
    # lv_endo_contours = SuiteHeart_datum_dict['lv_endo']    
    # lv_epi_contours = SuiteHeart_datum_dict['lv_epi']
    images = list_to_2D_ndarray(SuiteHeart_datum_dict['raw_image']) if type(SuiteHeart_datum_dict['raw_image']) is list else SuiteHeart_datum_dict['raw_image']
    lv_endo_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_endo']) if type(SuiteHeart_datum_dict['lv_endo']) is list else SuiteHeart_datum_dict['lv_endo']
    lv_epi_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_epi']) if type(SuiteHeart_datum_dict['lv_epi']) is list else SuiteHeart_datum_dict['lv_epi']

    if type(SuiteHeart_datum_dict['lv_endo']) is list:
        mat_has_single_slice_flag = True
    else:
        mat_has_single_slice_flag = False
    # mat_has_single_slice_flag = False
    # if len(images.shape) == 1 and len(lv_endo_contours.shape) == 1 and len(lv_epi_contours.shape) == 1:
    #     images = np.expand_dims(images, axis=1)
    #     lv_endo_contours = np.expand_dims(lv_endo_contours, axis=1)
    #     lv_epi_contours = np.expand_dims(lv_epi_contours, axis=1)
    #     mat_has_single_slice_flag = True
    # elif len(images.shape) == 2 and len(lv_endo_contours.shape) == 2 and len(lv_epi_contours.shape) == 2:
    #     mat_has_single_slice_flag = False
    # else:
    #     raise ValueError('The input SuiteHeart_datum_dict has invalid shapes.')
        
    n_frames, n_slices = images.shape
    images_rescaled = np.zeros_like(images)
    images_rescaled_cropped = np.zeros((n_slices, target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)    
    # rv_insertion_points_rescaled = []
    for slice_idx in range(n_slices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            pass
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], np.ndarray) and lv_endo_contours[fidx, slice_idx].size < 1 for fidx in range(n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], np.ndarray) and lv_epi_contours[fidx, slice_idx].size < 1 for fidx in range(n_frames)]):
            pass
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], list) and len(lv_endo_contours[fidx, slice_idx]) < 1 for fidx in range(n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], list) and len(lv_epi_contours[fidx, slice_idx]) < 1 for fidx in range(n_frames)]):
            pass
        else:
            # Scaling parameters
            H, W = images[0, slice_idx].shape
            if type(lv_endo_contours[0,0]) == dict:
                lv_endo_center = np.mean(lv_endo_contours[0, slice_idx]['contour'], axis=0) - 1
                
            else:
                lv_endo_center = np.mean(lv_endo_contours[0, slice_idx].contour, axis=0) - 1
            
            if rescale_method == 'pixel_spacing':
                pixelSpacing = SuiteHeart_datum_dict['pixel_size'][slice_idx] if type(SuiteHeart_datum_dict['pixel_size']) is list else SuiteHeart_datum_dict['pixel_size'][0, slice_idx]
                scale_factors = pixelSpacing
            elif rescale_method == 'fill':
                if type(lv_endo_contours[0,0]) == dict:
                    height_ratio = target_image_shape[0] / (np.max(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[1] - np.min(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[1])# / target_image_shape[0]
                    width_ratio = target_image_shape[1] / (np.max(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[0] - np.min(lv_endo_contours[0, slice_idx]['contour']-1, axis=0)[0])# / target_image_shape[1]
                else:                    
                    height_ratio = target_image_shape[0] /  (np.max(lv_endo_contours[0, slice_idx].contour-1, axis=0)[1] - np.min(lv_endo_contours[0, slice_idx].contour-1, axis=0)[1])
                    width_ratio = target_image_shape[1] / (np.max(lv_endo_contours[0, slice_idx].contour-1, axis=0)[0] - np.min(lv_endo_contours[0, slice_idx].contour-1, axis=0)[0])
                # height_ratio = (np.max(lv_endo_contours[:, slice_idx].contour[:,1]) - np.min(lv_endo_contours[:, slice_idx].contour[:,1])) / target_image_shape[0]
                # width_ratio = (np.max(lv_endo_contours[:, slice_idx].contour[:,0]) - np.min(lv_endo_contours[:, slice_idx].contour[:,0])) / target_image_shape[1]
                # fill_ratio = 0.7
                # scale_factors = [height_ratio*fill_ratio, width_ratio*fill_ratio]
                scale_factors = [min(height_ratio, width_ratio)*rescale_ratio, min(height_ratio, width_ratio)*rescale_ratio]
                # print(f'{scale_factors=}')
            elif rescale_method == 'rescale':
                scale_factors = [rescale_ratio, rescale_ratio]
            else:
                raise ValueError('Invalid rescale_method.')

            H_rescaled = int(H * scale_factors[0])
            W_rescaled = int(W * scale_factors[1])
            lv_endo_center_rescaled = np.floor(lv_endo_center * scale_factors)

            # Get the rescaled images
            for frame_idx in range(n_frames):
                images_rescaled[frame_idx, slice_idx] = resize(images[frame_idx, slice_idx], 
                    (H_rescaled, W_rescaled), 
                    anti_aliasing=True)

            # Crop the rescaled images
            # print(f'{lv_endo_center_rescaled=}')
            for frame_idx in range(n_frames):
                # H_rescaled, W_rescaled = images_rescaled[frame_idx, slice_idx].shape
                H_crop, W_crop = target_image_shape
                H_crop_start = int(lv_endo_center_rescaled[1] - H_crop/2)
                W_crop_start = int(lv_endo_center_rescaled[0] - W_crop/2)
                images_rescaled_cropped[slice_idx, :, :, frame_idx] = images_rescaled[frame_idx, slice_idx][H_crop_start:H_crop_start+H_crop, W_crop_start:W_crop_start+W_crop]
    if mat_has_single_slice_flag == True:
        images_rescaled_cropped = images_rescaled_cropped[0]
        
    return images_rescaled_cropped


def get_SuiteHeart_dict_cropped_images(SuiteHeart_datum_dict, target_image_shape=(128,128)):
    
    # images = np.array(SuiteHeart_datum_dict['raw_image'])
    # lv_endo_contours = np.array(SuiteHeart_datum_dict['lv_endo'])
    # lv_epi_contours = np.array(SuiteHeart_datum_dict['lv_epi'])
    images = list_to_2D_ndarray(SuiteHeart_datum_dict['raw_image']) if type(SuiteHeart_datum_dict['raw_image']) is list else SuiteHeart_datum_dict['raw_image']
    lv_endo_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_endo']) if type(SuiteHeart_datum_dict['lv_endo']) is list else SuiteHeart_datum_dict['lv_endo']
    lv_epi_contours = list_to_2D_ndarray(SuiteHeart_datum_dict['lv_epi']) if type(SuiteHeart_datum_dict['lv_epi']) is list else SuiteHeart_datum_dict['lv_epi']

    # mat_has_single_slice_flag = False
    # if len(images.shape) == 1 and len(lv_endo_contours.shape) == 1 and len(lv_epi_contours.shape) == 1:
    #     images = np.expand_dims(images, axis=1)
    #     lv_endo_contours = np.expand_dims(lv_endo_contours, axis=1)
    #     lv_epi_contours = np.expand_dims(lv_epi_contours, axis=1)
    #     mat_has_single_slice_flag = True
    # elif len(images.shape) == 2 and len(lv_endo_contours.shape) == 2 and len(lv_epi_contours.shape) == 2:
    #     mat_has_single_slice_flag = False
    # else:
    #     raise ValueError('The input SuiteHeart_datum_dict has invalid shapes.')
    if type(SuiteHeart_datum_dict['lv_endo']) is list:
        mat_has_single_slice_flag = True
    else:
        mat_has_single_slice_flag = False

    n_frames, n_slices = images.shape
    H_cropped, W_cropped = target_image_shape
    myo_bboxes = []
    images_cropped = np.zeros((n_slices, target_image_shape[0], target_image_shape[1], n_frames), dtype=np.float32)
    for slice_idx in range(n_slices):
        # if the beginning frame has no contour
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1):
            myo_bboxes.append([])
        # if any of the frames has no contour
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], np.ndarray) and lv_endo_contours[fidx, slice_idx].size < 1 for fidx in range(n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], np.ndarray) and lv_epi_contours[fidx, slice_idx].size < 1 for fidx in range(n_frames)]):
            myo_bboxes.append([])
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], list) and len(lv_endo_contours[fidx, slice_idx]) < 1 for fidx in range(n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], list) and len(lv_epi_contours[fidx, slice_idx]) < 1 for fidx in range(n_frames)]):
            myo_bboxes.append([])
        else:
            # Scaling parameters
            if type(lv_endo_contours[0,0]) == dict:
                lv_endo_center = np.mean(lv_endo_contours[0, slice_idx]['contour'], axis=0) - 1
            else:
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
    
    if mat_has_single_slice_flag == True:
        images_cropped = images_cropped[0]
        myo_bboxes = myo_bboxes[0]

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
    
    lv_endo_contours = np.array(SuiteHeart_datum_dict['lv_endo'])
    lv_epi_contours = np.array(SuiteHeart_datum_dict['lv_epi'])
    
    
    mat_has_single_slice_flag = False
    if len(lv_endo_contours.shape) == 1 and len(lv_epi_contours.shape) == 1:
        lv_endo_contours = np.expand_dims(lv_endo_contours, axis=1)
        lv_epi_contours = np.expand_dims(lv_epi_contours, axis=1)
        rv_insertion_points = np.expand_dims(np.array([c[0]['contour'] for c in SuiteHeart_datum_dict['rv_insertion']]),axis=1)
        mat_has_single_slice_flag = True
    elif len(lv_endo_contours.shape) == 2 and len(lv_epi_contours.shape) == 2:
        mat_has_single_slice_flag = False
        rv_insertion_points = np.zeros_like(SuiteHeart_datum_dict['rv_insertion'])
        for frame_idx in range(rv_insertion_points.shape[0]):
            for slice_idx in range(rv_insertion_points.shape[1]):
                pre_rv_insertion = SuiteHeart_datum_dict['rv_insertion'][frame_idx, slice_idx]
                if len(pre_rv_insertion) > 0:
                    rv_insertion_points[frame_idx, slice_idx] = get_contour(SuiteHeart_datum_dict['rv_insertion'][frame_idx, slice_idx][0])
    else:
        raise ValueError('The input SuiteHeart_datum_dict has invalid shapes.')
    lv_endo_n_frames, lv_endo_n_slices = lv_endo_contours.shape
    lv_epi_n_frames, lv_epi_n_slices = lv_epi_contours.shape

    if check_slice_indices is None:
        check_slice_indices = list(range(lv_endo_n_slices))
    n_save_slices = len(check_slice_indices)
    
    lv_endo_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    lv_epi_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)
    myo_masks = np.zeros((n_save_slices, target_image_shape[0], target_image_shape[1], lv_endo_n_frames), dtype=np.uint8)

    lv_epi_contours_cropped = np.zeros((lv_endo_n_frames, lv_endo_n_slices), dtype=object)
    lv_endo_contours_cropped = np.zeros((lv_endo_n_frames, lv_endo_n_slices), dtype=object)

    H_cropped, W_cropped = target_image_shape
    # Generate rescaled contours and masks
    
    myo_bboxes = []
    rv_insertion_points_cropped = np.zeros((n_save_slices, 2), dtype=np.float32)
    for save_slice_idx, slice_idx in enumerate(check_slice_indices):
        if (isinstance(lv_endo_contours[0, slice_idx], np.ndarray) and lv_endo_contours[0, slice_idx].size < 1) or \
            (isinstance(lv_epi_contours[0, slice_idx], np.ndarray) and lv_epi_contours[0, slice_idx].size < 1):
            # lv_endo_contours_rescaled = lv_endo_contours[:, slice_idx].contour
            # lv_epi_contours_rescaled = lv_epi_contours[:, slice_idx].contour
            myo_bboxes.append([])
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], np.ndarray) and lv_endo_contours[fidx, slice_idx].size < 1 for fidx in range(lv_endo_n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], np.ndarray) and lv_epi_contours[fidx, slice_idx].size < 1 for fidx in range(lv_epi_n_frames)]):
            myo_bboxes.append([])
        elif any([isinstance(lv_endo_contours[fidx, slice_idx], list) and len(lv_endo_contours[fidx, slice_idx]) < 1 for fidx in range(lv_endo_n_frames)]) or \
            any([isinstance(lv_epi_contours[fidx, slice_idx], list) and len(lv_epi_contours[fidx, slice_idx]) < 1 for fidx in range(lv_epi_n_frames)]):
            myo_bboxes.append([])
        else:
            
            # target_image_shape2 = [d*1 for d in target_image_shape]
            if type(lv_endo_contours[0,0]) == dict:
                slice_myo_center = lv_endo_contours[:, slice_idx][0]['contour'].mean(axis=0) - 1
            else:
                slice_myo_center = lv_endo_contours[:, slice_idx][0].contour.mean(axis=0) - 1
            slice_myo_bbox = [int(slice_myo_center[1])-H_cropped//2,int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2,int(slice_myo_center[0])-W_cropped//2+W_cropped]
            debugging = 1
            if type(lv_endo_contours[0,0]) == dict:
                slice_lv_endo_contours_list = [c['contour']-1 for c in lv_endo_contours[:, slice_idx]]
                slice_lv_epi_contours_list = [c['contour']-1 for c in lv_epi_contours[:, slice_idx]]
            else:
                debugging = 1
                slice_lv_endo_contours_list = [c.contour-1 for c in lv_endo_contours[:, slice_idx]]
                slice_lv_epi_contours_list = [c.contour-1 for c in lv_epi_contours[:, slice_idx]]
            slice_lv_endo_masks = generate_binary_masks_from_contour_seq(
                slice_lv_endo_contours_list, 
                image_shape=ori_image_shape, centering=centering, ori_center=slice_myo_center)
            slice_lv_epi_masks = generate_binary_masks_from_contour_seq(
                slice_lv_epi_contours_list, 
                image_shape=ori_image_shape, centering=centering, ori_center=slice_myo_center)
            
            rv_insertion_point = rv_insertion_points[0, slice_idx]
            rv_insertion_point_cropped = rv_insertion_point - np.array([slice_myo_bbox[2], slice_myo_bbox[0]])
            rv_insertion_points_cropped[save_slice_idx] = rv_insertion_point_cropped

            for frame_idx in range(lv_endo_n_frames):
                lv_epi_contours_cropped[frame_idx, save_slice_idx] = \
                    get_contour(lv_epi_contours[frame_idx, slice_idx]) - np.array([slice_myo_bbox[2], slice_myo_bbox[0]])
                lv_endo_contours_cropped[frame_idx, save_slice_idx] = \
                    get_contour(lv_endo_contours[frame_idx, slice_idx]) - np.array([slice_myo_bbox[2], slice_myo_bbox[0]])

            # lv_endo_masks[save_slice_idx] = slice_lv_endo_masks[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            # lv_epi_masks[save_slice_idx] = slice_lv_epi_masks[int(slice_myo_center[1])-H_cropped//2:int(slice_myo_center[1])-H_cropped//2+H_cropped, int(slice_myo_center[0])-W_cropped//2:int(slice_myo_center[0])-W_cropped//2+W_cropped]
            lv_endo_masks[save_slice_idx] = crop_image_given_bbox(slice_lv_endo_masks, slice_myo_bbox)
            lv_epi_masks[save_slice_idx] = crop_image_given_bbox(slice_lv_epi_masks, slice_myo_bbox)
            myo_masks[save_slice_idx] = np.logical_xor(lv_endo_masks[save_slice_idx], lv_epi_masks[save_slice_idx])
            myo_bboxes.append(slice_myo_bbox)
    
    if mat_has_single_slice_flag == True:
        lv_endo_masks = lv_endo_masks[0]
        lv_epi_masks = lv_epi_masks[0]
        myo_masks = myo_masks[0]
        myo_bboxes = myo_bboxes[0]
        rv_insertion_points_cropped = rv_insertion_points_cropped[0]
        lv_endo_contours_cropped = lv_endo_contours_cropped[:,0]
        lv_epi_contours_cropped = lv_epi_contours_cropped[:,0]
    
    # return lv_endo_contours_rescaled, lv_epi_contours_rescaled, lv_endo_masks_rescaled, lv_epi_masks_rescaled, myo_masks_rescaled
    # return myo_masks, myo_bboxes
    return lv_endo_contours_cropped, lv_epi_contours_cropped, myo_masks, myo_bboxes, rv_insertion_points_cropped



import pandas as pd
import numpy as np

def is_header(row):
    """Check if the row can be considered a header (more than one non-empty cell and not a divider)."""
    non_empty_cells = row.dropna()
    if len(non_empty_cells) <= 1:
        return False
    # Assuming dividers/headers do not contain numeric values and have less structured data
    if non_empty_cells.apply(lambda x: isinstance(x, (int, float))).any():
        return False
    return True

def is_divider(row):
    """Check if the row is a divider (only one non-empty cell which does not contain numbers)."""
    non_empty_cells = row.dropna()
    if len(non_empty_cells) == 1 and isinstance(non_empty_cells.iloc[0], str):
        return True
    return False

def extract_SA_data_from_xls(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Load a specific sheet by name
    df = xls.parse('Function',skiprows=1)  # Adjust sheet name as necessary

    # Function to check if a row is completely empty
    def is_empty_row(row):
        return row.isnull().all()

    # Split the DataFrame into chunks separated by empty rows
    table_chunks = []
    current_chunk = []

    first=False
    for index, row in df.iterrows():
        # print(str(row))
        # if index > 10:
        #     break
        # if index < 5:
        #     print('!!!!',row)
        #     print('!!!!',row)
        #     continue
        if is_empty_row(row):
            if current_chunk:
                # table_chunks.append(current_chunk)
                table_chunks.append(pd.DataFrame(current_chunk))
                current_chunk = []
        else:
            # if first is False:
            #     print('first append', row)
            #     first = True
            current_chunk.append(row)

    # Don't forget to add the last chunk if it's not empty
    if current_chunk:
        # table_chunks.append(current_chunk)
        table_chunks.append(pd.DataFrame(current_chunk))

    # Dictionary to store the data
    data_tables = {}

    # Process each chunk to create DataFrame if it has at least 3 rows
    for chunk in table_chunks:
        if len(chunk) >= 3:  # Ignore chunks with fewer than 3 rows
            table_name = chunk.iloc[0].dropna().values[0]  # First row, first non-empty cell as table name
            
            # headers = chunk[1].dropna().tolist()  # Second row as headers
            # data = chunk[2:]  # Remaining rows as data
            headers = chunk.iloc[1].dropna().reset_index(drop=True)  # Second row as headers, reset index
            data = chunk.iloc[2:].reset_index(drop=True)  # Remaining rows as data, reset index

            # drop the columns of data where all values are NaN
            data = data.dropna(axis=1, how='all')
                    
            # Create DataFrame
            # data_tables[table_name] = pd.DataFrame(data, columns=headers)
            # data.columns = headers[:len(data.columns)]
            try:
                data.columns = headers[:len(data.columns)]
            except:
                pass
            data_tables[table_name] = data

    return data_tables