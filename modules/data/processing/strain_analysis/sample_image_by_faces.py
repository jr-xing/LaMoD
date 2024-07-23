import numpy as np
from skimage.draw import polygon

def sample_image_by_faces(image, vertices, faces, mask=None):
    """
    Samples an image by given faces and vertices, computing the regional average intensity value for each face,
    considering only the pixels within the specified mask.
    
    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - vertices: (Nv, 2) numpy array with 2D coordinates for Nv vertices.
    - faces: (Nf, 4) numpy array with each row containing indices of vertices forming a face.
    - mask: Optional boolean numpy array of the same shape as image to specify the region to consider.
    
    Returns:
    - sampling_results: numpy array of shape (Nf, ) with average intensity values for each face.
    """
    Nf = faces.shape[0]  # Number of faces
    sampling_results = np.zeros(Nf)
    
    for i in range(Nf):
        # Get vertex coordinates for the current face
        face_vertices = vertices[faces[i]]
        
        # Generate polygon mask for the face
        rr, cc = polygon(face_vertices[:, 1], face_vertices[:, 0], image.shape)
        
        # Apply the input mask if it's provided
        if mask is not None:
            rr, cc = rr[mask[rr, cc]], cc[mask[rr, cc]]
        
        # Ensure there are pixels to calculate an average for
        if len(rr) > 0 and len(cc) > 0:
            # Compute average intensity within the polygon and the input mask
            sampling_results[i] = np.mean(image[rr, cc])
        else:
            # If no pixels are found within the mask for this face, set average to NaN or some indicator value
            print(f'No pixels found within the mask for face {i}.')
            sampling_results[i] = np.nan
    
    return sampling_results

# Example usage:
# Assuming you have an image `img` and the vertices and faces arrays `verts` and `faces`
# Additionally, if you have a mask `mask_img` of the same shape as `img`
# img = np.random.rand(100, 100)  # Example 2D image
# verts = np.array([[10, 10], [80, 10], [80, 80], [10, 80], [45, 45]])  # Example vertices
# faces = np.array([[0, 1, 4, 3], [1, 2, 4, 0], [2, 3, 4, 1]])  # Example faces
# mask_img = np.random.rand(100, 100) > 0.5  # Example mask
# sampling_results = sample_image_by_faces(img, verts, faces, mask_img)
# print(sampling_results)


import numpy as np
from scipy.interpolate import interp2d

def sample_image_at_face_centers(image, vertices, faces, mask=None):
    """
    Samples an image by computing the centers of given faces, using image interpolation
    to sample the image value at these centers.
    
    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - vertices: (Nv, 2) numpy array with 2D coordinates for Nv vertices.
    - faces: (Nf, 4) numpy array with each row containing indices of vertices forming a face.
    - mask: Optional boolean numpy array of the same shape as image to specify the region to consider.
    
    Returns:
    - sampling_results: numpy array of shape (Nf, ) with sampled values at each face center.
    """
    Nf = faces.shape[0]  # Number of faces
    sampling_results = np.zeros(Nf)
    
    # Create an interpolator for the image
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    interpolator = interp2d(x, y, image, kind='linear')
    
    for i in range(Nf):
        # Compute the centroid of the face
        face_vertices = vertices[faces[i]]
        centroid = np.mean(face_vertices, axis=0)
        
        # Check if the centroid is within the mask (if mask is provided)
        if mask is not None:
            if not mask[int(centroid[1]), int(centroid[0])]:
                sampling_results[i] = np.nan
                continue
        
        # Sample the image at the centroid using interpolation
        sampling_results[i] = interpolator(centroid[0], centroid[1])
    
    return sampling_results

# Example usage:
# Assuming you have an image `img` and the vertices and faces arrays `verts` and `faces`
# Additionally, if you have a mask `mask_img` of the same shape as `img`
# img = np.random.rand(100, 100)  # Example 2D image
# verts = np.array([[10, 10], [80, 10], [80, 80], [10, 80], [45, 45]])  # Example vertices
# faces = np.array([[0, 1, 4, 3], [1, 2, 4, 0], [2, 3, 4, 1]])  # Example faces
# mask_img = np.random.rand(100, 100) > 0.5  # Example mask
# sampling_results = sample_image_at_face_centers(img, verts, faces, mask_img)
# print(sampling_results)
