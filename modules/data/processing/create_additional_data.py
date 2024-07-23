import numpy as np
from modules.data.processing.displacement_utils import extract_radial_tangent_components
def create_radial_gradient(shape):
    """Generate a radial gradient for a given shape."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0] / 2, shape[1] / 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r / r.max()
    return r

def apply_gradient_to_mask(mask, gradient=None, scale_factor=1.0):
    """
    Apply a radial gradient to a binary mask.

    :param mask: A 2D or 3D numpy array representing the binary mask.
    :return: Masked gradient image.
    """
    if mask.ndim == 2:
        if gradient is None:
            gradient = create_radial_gradient(mask.shape)
        return mask * gradient
    elif mask.ndim == 3:
        if gradient is None:
            gradient = create_radial_gradient(mask.shape[1:])
        return mask * gradient[..., np.newaxis]
    else:
        raise ValueError("Mask must be either 2D or 3D.")

def create_mask_gradient_pattern(data: list, config):
    """
    Create a gradient mask for each slice in the data.

    :param data: A list of data dictionaries.
    :param config: A dictionary containing the configuration parameters.
    :return: A list of data dictionaries with the gradient mask added.

    Example of config: 
        {
            "type": "gradient_mask",
            "mask_key": "myo_masks",
            "gradient_mask_key": "grad_myo_masks"
        }
    """
    mask_key = config.get('mask_key', 'myo_masks')
    gradient_mask_key = config.get('gradient_mask_key', 'grad_myo_masks')    
    scale_factor = config.get('scale_factor', 1.0)
    # radial_gradient_img = create_3d_radial_gradient(data[0][mask_key].shape, scale_factor)
    radial_gradient = create_radial_gradient(data[0][mask_key].shape)
    for datum in data:
        datum[gradient_mask_key] = apply_gradient_to_mask(datum[mask_key], radial_gradient, scale_factor)
    
    return data

def compute_image_gradient(img: np.ndarray, config):
    """
    Compute the gradient of an image.

    :param img: A 2D or 3D numpy array representing the image.
    :param config: A dictionary containing the configuration parameters.
    :return: A 2D or 3D numpy array representing the gradient of the image.
    """
    if img.ndim == 2:
        img_grad = np.gradient(img)
    elif img.ndim == 3:
        img_grad = np.gradient(img, axis=(0, 1))
    else:
        raise ValueError("Image must be either 2D or 3D.")
    # compute magnitude of gradient
    img_grad_mag = np.abs(img_grad[0]) + np.abs(img_grad[1])
    # normalize
    img_grad_mag = img_grad_mag / np.max(img_grad_mag)
    return img_grad_mag

def create_image_gradient(data: list, config):
    """Create a gradient image for each slice in the data

    :param data: A list of data dictionaries.
    :param config: A dictionary containing the configuration parameters.
    :return: A list of data dictionaries with the gradient image added.

    Example of config: 
        {
            "type": "image_gradient",
            "image_key": "images",
            "gradient_image_key": "grad_images"
        }
    """
    image_key = config.get('image_key', 'images')
    gradient_image_key = config.get('gradient_image_key', 'grad_images')
    for datum in data:
        datum[gradient_image_key] = compute_image_gradient(datum[image_key].astype(float), config)
    
    return data


def create_addition_data(data:list, config):
    """
    Create additional data for each slice in the data.

    :param data: A list of data dictionaries.
    :param config: A dictionary containing the configuration parameters.
    :return: A list of data dictionaries with the additional data added.

    Example of config: 
        {
            "gradient_mask":{
                "type": "gradient_mask",
                "mask_key": "myo_masks",
                "gradient_mask_key": "grad_myo_masks"
        }
    """
    for key, value in config.items():
        additional_data_type = value.get('type', None)
        print(f'Creating additional data of type: {additional_data_type}')
        if additional_data_type == 'mask_gradient_pattern':
            data = create_mask_gradient_pattern(data, value)
        elif additional_data_type == 'image_gradient':
            data = create_image_gradient(data, value)
        elif additional_data_type == 'radial_tagent_components':
            disp_key = value.get('disp_key', 'DENSE_disp')
            disp_component_key_prefix = value.get('disp_component_key_prefix', disp_key)
            for datum_idx, datum in enumerate(data):
                disp = datum[disp_key] # should have shape (2, H, W, T)
                _, H, W, T = disp.shape
                disp_rad = np.zeros((2, H, W, T))
                disp_tan = np.zeros((2, H, W, T))
                for t in range(T):
                    radial_component, tangent_component = extract_radial_tangent_components(disp[...,t])
                    disp_rad[...,t] = radial_component
                    disp_tan[...,t] = tangent_component
                datum[disp_component_key_prefix + '_rad'] = disp_rad
                datum[disp_component_key_prefix + '_tan'] = disp_tan
        else:
            raise ValueError(f'Unsupported additional data type: {key}')
    
    return data