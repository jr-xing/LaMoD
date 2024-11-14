import numpy as np
from skimage.draw import polygon
import cv2
import copy
def generate_binary_mask_from_contour(contour, image_shape, implementation='skimg'):
    """
    Generates a binary mask from a 2D contour.
    
    Parameters:
    contour (numpy.ndarray): 2D numpy array with shape (n_points, 2) representing the contour coordinates.
    image_shape (tuple): 2-tuple representing the shape of the image.
    
    Returns:
    numpy.ndarray: Binary mask with shape (image_shape).
    """
    if implementation == 'skimg':
        mask = np.zeros(image_shape, dtype=np.uint8)
        rr, cc = polygon(contour[:, 1], contour[:, 0], mask.shape)
        mask[rr, cc] = 255
    elif implementation in ['opencv', 'cv2']:
        mask = np.zeros(image_shape, dtype=np.uint8)
        c = copy.deepcopy(contour)
        c[:, [0, 1]] = c[:, [1, 0]]
        cv2.drawContours(mask, [c.astype(int)], 0, 255, -1)
    return mask

def generate_binary_masks_from_contour_seq(contours, image_shape, implementation='skimg', centering=True, contour_ori_center=None):
    """
    Generates a binary mask from a sequence of 2D contours.
    
    Parameters:
    contours (ndarray): 1d array of 2D numpy arrays with shape (n_frames) representing the contour coordinates of each frame.
        Each element is a numpy array of shape (n_points, 2) representing a contour.
    image_shape (tuple): 2-tuple representing the shape of the image.
    
    Returns:
    numpy.ndarray: Binary mask with shape (image_shape).
    """
    # mask = np.zeros(image_shape, dtype=np.uint8)
    n_contours = len(contours)
    

    # if centering and center is not None:
    #     image_center = np.array(image_shape) / 2
    if centering:
        image_center = np.array(image_shape) / 2
        contour_ori_center = contours[0].mean(axis=0) if contour_ori_center is None else contour_ori_center
    else:
        image_center = np.array([0, 0])
        contour_ori_center = np.array([0, 0])
    
    mask = np.zeros((image_shape[0], image_shape[1], n_contours), dtype=np.uint8)
    for frame_idx, contour in enumerate(contours):
        mask[:, :, frame_idx] = generate_binary_mask_from_contour(contour-contour_ori_center+image_center, image_shape, implementation)
    return mask

# def generate_binary_masks_from_contour_seq(contours, image_shape, implementation='skimg', centering=True, ori_center=None):
#     """
#     Generates a binary mask from a sequence of 2D contours.
    
#     Parameters:
#     contours (ndarray): 1d array of 2D numpy arrays with shape (n_frames) representing the contour coordinates of each frame.
#         Each element is a numpy array of shape (n_points, 2) representing a contour.
#     image_shape (tuple): 2-tuple representing the shape of the image.
    
#     Returns:
#     numpy.ndarray: Binary mask with shape (image_shape).
#     """
#     # mask = np.zeros(image_shape, dtype=np.uint8)
#     n_contours = len(contours)
    

#     # if centering and center is not None:
#     #     image_center = np.array(image_shape) / 2
#     if centering:
#         #if center is None:
#         #    image_center = np.array(image_shape) / 2
#         #else:
#         #    image_center = center
#         image_center = np.array(image_shape) / 2
#         contour_ori_center = ori_center if ori_center is not None else contours[0].mean(axis=0)
#         # print(image_shape, image_center, contour_ori_center)
#     else:
#         image_center = np.array([0, 0])
#         contour_ori_center = np.array([0, 0])
#         # print(image_shape, image_center, contour_ori_center)
    
#     mask = np.zeros((image_shape[0], image_shape[1], n_contours), dtype=np.uint8)
#     for frame_idx, contour in enumerate(contours):
#         mask[:, :, frame_idx] = generate_binary_mask_from_contour(contour-contour_ori_center+image_center, image_shape, implementation)
#     return mask

def generate_binary_mask_from_two_contours_seq(contours, image_shape, implementation='skimg', centering=True):
    """
    Generates a binary mask from a sequence of two 2D contours.
    
    Parameters:
    contours (ndarray): A numpy array of shape (n_frames, n_contours) representing multiple sequences of contours.
        Each element is a numpy array of shape (n_points, 2) representing a contour.
        n_contours should be 2, otherwise the later contours will be ignored.
    image_shape (tuple): 2-tuple representing the shape of the image.
    
    Returns:
    numpy.ndarray: Binary mask with shape (image_shape).
    """
    # mask = np.zeros(image_shape, dtype=np.uint8)
    mask0 = generate_binary_masks_from_contour_seq(contours[:,0], image_shape, implementation, centering=centering)
    mask1 = generate_binary_masks_from_contour_seq(contours[:,1], image_shape, implementation, centering=centering, ori_center=contours[:,0])
    mask = np.logical_xor(mask0, mask1)
    return mask

def rescale_contour(contour, original_spacing=(1,1), target_spacing=(2,2), center=None, rescale_center=False):
    """
    Rescales a 2D contour according to the pixel spacing.
    
    Parameters:
    contour (numpy.ndarray): 2D numpy array with shape (n_points, 2) representing the contour coordinates.
    original_spacing (tuple): 2-tuple representing the x and y physical pixel spacing of the original image. Default is (1, 1).
    target_spacing (tuple): 2-tuple representing the x and y physical pixel spacing of the target image. Default is (2, 2).
    center (tuple): 1x2 numpy array representing the center of the contour. If None, the center is calculated as the mean of the contour points.
    
    Returns:
    numpy.ndarray: Rescaled contour.
    """
    # Calculate the scaling factors for x and y dimensions
    scaling_factor = (original_spacing[0] / target_spacing[0], original_spacing[1] / target_spacing[1])

    # Calculate the center of the contour
    center = contour.mean(axis=0) if center is None else center
    
    # Rescale the contour
    if rescale_center:
        rescaled_contour = contour * scaling_factor
    else:
        rescaled_contour = (contour - center) * scaling_factor + center
    # print(rescaled_contour.mean(axis=0))
    return rescaled_contour

def rescale_contours_seq(contours, original_spacing=(1,1), target_spacing=(2,2), center=None, rescale_center=False):
    """
    Rescales a sequence of 2D contours according to the pixel spacing.
    All contours will be centered to the center of the first contour.
    
    Parameters:
    contours (ndarray): List of 2D numpy arrays with shape (n_points, 2) representing the contour coordinates.
    original_spacing (tuple): 2-tuple representing the x and y physical pixel spacing of the original image.
    target_spacing (tuple): 2-tuple representing the x and y physical pixel spacing of the target image.
    center (tuple): 1x2 numpy array representing the center of the contour. If None, the center is calculated as the mean of the contour points.
    
    Returns:
    list: List of rescaled contours.
    """
    rescaled_contours = []
    init_contour = contours[0]
    init_contour_rescaled = rescale_contour(
        init_contour, 
        original_spacing=original_spacing, 
        target_spacing=target_spacing,
        rescale_center=rescale_center)
    rescaled_contours.append(init_contour_rescaled)

    init_contour_rescaled_center = init_contour_rescaled.mean(axis=0)
    if center is None:
        center = init_contour_rescaled_center
    for contour in contours[1:]:
        rescaled_contour = rescale_contour(
            contour, 
            original_spacing=original_spacing, 
            target_spacing=target_spacing, 
            center=center,
            rescale_center=rescale_center)
        rescaled_contours.append(rescaled_contour)
    return rescaled_contours

def rescale_multiple_contours_seq(contours, original_spacing=(1,1), target_spacing=(2,2), center=None, rescale_center=False):
    """
    Rescales multiple sequences of contours.

    Different from rescale_contours_seq, here the variable contours is a (n_frames, n_contours) numpy array, 
        where each element is a (n_points, 2) numpy array representing a contour
    When center is None, it should use the center of the first contour of the first frame

    Args:
        contours (ndarray): A numpy array of shape (n_frames, n_contours) representing multiple sequences of contours.
                            Each element is a numpy array of shape (n_points, 2) representing a contour.
        original_spacing (tuple, optional): The original spacing between points in the contours. Defaults to (1, 1).
        target_spacing (tuple, optional): The target spacing between points in the rescaled contours. Defaults to (2, 2).
        center (ndarray, optional): The center point to use for rescaling. If None, the center of the first contour of the first frame is used. Defaults to None.

    Returns:
        ndarray: A numpy array of the same shape as the input `contours`, containing the rescaled contours.
    """
    n_frames, n_contours = contours.shape
    rescaled_contours = np.zeros_like(contours)
    init_contour = contours[0, 0]
    init_contour_rescaled = rescale_contour(
        init_contour, 
        original_spacing=original_spacing, 
        target_spacing=target_spacing)
    rescaled_contours[0, 0] = init_contour_rescaled

    init_contour_rescaled_center = init_contour_rescaled.mean(axis=0)
    if center is None:
        center = init_contour_rescaled_center
    
    rescaled_contours[0,1] = rescale_contour(
        contours[0, 1],
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        center=center,
        rescale_center=rescale_center)
    
    for frame_idx in range(1, n_frames):
        for contour_idx in range(n_contours):
            contour = contours[frame_idx, contour_idx]
            rescaled_contour = rescale_contour(
                contour, 
                original_spacing=original_spacing, 
                target_spacing=target_spacing, 
                center=center,
                rescale_center=rescale_center)
            rescaled_contours[frame_idx, contour_idx] = rescaled_contour

    return rescaled_contours