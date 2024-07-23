import numpy as np
def extract_radial_tangent_components(displacement_field, center=None):
    """
    Extracts radial and tangent components of a displacement field.

    Args:
    - displacement_field: A numpy array of shape (2, H, W) representing the displacement vectors.
    - center: A tuple (x, y) specifying the center point. Defaults to the center of the displacement field.

    Returns:
    - radial_component: The radial component of the displacement field.
    - tangent_component: The tangent component of the displacement field.
    """
    _, H, W = displacement_field.shape
    if center is None:
        center = (W // 2, H // 2)
    
    # Adjust coordinate grid generation to ensure shape compatibility
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Calculate the direction vectors from the center to each point
    direction_vectors = np.stack([(X - center[0]), (Y - center[1])], axis=0)
    
    # Normalize direction vectors to get unit vectors
    norms = np.sqrt(direction_vectors[0]**2 + direction_vectors[1]**2)
    norms[norms == 0] = 1  # To avoid division by zero
    unit_direction_vectors = direction_vectors / norms
    
    # Calculate radial components
    displacement_magnitudes = np.sum(displacement_field * unit_direction_vectors, axis=0)
    radial_component = unit_direction_vectors * displacement_magnitudes
    
    # Calculate tangent components
    tangent_component = displacement_field - radial_component
    
    return radial_component, tangent_component