# from modules.trainer.reg_trainer import RegTrainer
# from modules.trainer.joint_registration_regression_trainer import JointRegistrationRegressionTrainer
# from modules.trainer.LMA_trainer import LMATrainer
# from modules.trainer.strainmat_pred_trainer import StrainmatPredTrainer
# from modules.trainer.strainmat_LMA_trainer import StrainmatLMATrainer
# from modules.trainer.joint_registration_strainmat_LMA import JointRegisterStrainmatLMATrainer
from modules.trainer.base_trainer import BaseTrainer
from modules.trainer.regression_trainer import RegressionTrainer
from modules.trainer.registration_trainer import RegTrainer
# from modules.trainer.ae_trainer import AETrainer
# from modules.trainer.DENSE_disp_pred_trainer import DENSEDispPredTrainer
# from modules.trainer.DENSE_disp_pred_naive_trainer import DENSEDispPredNaiveTrainer
# from modules.trainer.DENSE_disp_cond_gene_trainer import DENSEDispCondGeneTrainer
# from modules.trainer.DENSE_disp_cond_pred_trainer import DENSEDispCondPredTrainer
# from modules.trainer.DENSE_disp_regression_trainer import DENSEDispRegressionTrainer
# from modules.trainer.DENSE_latent_denoiser_trainer import DENSELatentDenoiserTrainer
# from modules.trainer.DENSE_bio_info_motion_tracking_trainer import DENSEBioInfoMotionTrackingTrainer
# from modules.trainer.DENSE_bio_info_motion_tracking_VAE_trainer import DENSEBioInfoMotionTrackingVAETrainer

# from modules.trainer.DENSE_disp_pred_alter_trainer import DENSEDispPredAlterTrainer

def build_trainer(trainer_config, device=None, full_config=None):
    trainer_scheme = trainer_config['scheme']
    if trainer_scheme.lower() in ['regression']:
        return RegressionTrainer(trainer_config, device, full_config)
    if trainer_scheme.lower() in ['reg', 'registration']:
        return RegTrainer(trainer_config, device, full_config)
    # elif trainer_scheme.lower() in ['ae',' autoencoder']:
    #     return AETrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred':
    #     return DENSEDispPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred_naive':
    #     return DENSEDispPredNaiveTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_cond_pred':
    #     return DENSEDispCondPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_cond_gene':
    #     return DENSEDispCondGeneTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_regression':
    #     return DENSEDispRegressionTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_latent_denoiser':
    #     return DENSELatentDenoiserTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_bio_info_motion_tracking':
    #     return DENSEBioInfoMotionTrackingTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_bio_info_motion_tracking_VAE':
    #     return DENSEBioInfoMotionTrackingVAETrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'DENSE_disp_pred_alter':
    #     return DENSEDispPredAlterTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'LMA':
    #     return LMATrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'joint_registration_regression':
    #     return JointRegistrationRegressionTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'strainmat_pred':
    #     return StrainmatPredTrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'strainmat_LMA':
    #     return StrainmatLMATrainer(trainer_config, device, full_config)
    # elif trainer_scheme == 'joint_registration_strainmat_LMA':
    #     return JointRegisterStrainmatLMATrainer(trainer_config, device, full_config)
    else:
        raise NotImplementedError(f"trainer scheme {trainer_scheme} not implemented")

import torch
def patchify(full_volume, patch_length, patchify_axis=2, overlap_length=None):
    """
    Break an input PyTorch tensor into several overlapping patches along a specified axis.

    Args:
        full_volume (torch.Tensor): Input tensor to be divided into patches.
            Can have any number of dimensions.
        patch_length (int): Length of each patch along the specified axis.
            Must be less than or equal to the size of the input tensor along patchify_axis.
        patchify_axis (int, optional): Axis along which to create patches.
            Defaults to 2.
        overlap_length (int, optional): Length of overlap between consecutive patches.
            If None, defaults to patch_length // 2.
            Must be less than patch_length.

    Returns:
        tuple: Contains:
            - list[torch.Tensor]: List of tensor patches
            - list[tuple[int, int]]: List of (start_idx, end_idx) for each patch

    Example:
        >>> x = torch.randn(1, 3, 31, 64, 64)  # [N,C,D,H,W]
        >>> patches, indices = patchify(x, patch_length=16, patchify_axis=2)
        >>> # Returns patches and their indices like:
        >>> # First patch:  0->15
        >>> # Second patch: 8->23
        >>> # Last patch:   15->30
    """
    # Input validation
    if patchify_axis < 0 or patchify_axis >= full_volume.ndim:
        raise ValueError(f"patchify_axis must be between 0 and {full_volume.ndim-1}")
    
    axis_length = full_volume.shape[patchify_axis]
    if patch_length > axis_length:
        raise ValueError(f"patch_length ({patch_length}) cannot be greater than the tensor size along axis {patchify_axis} ({axis_length})")
    
    # Set default overlap length if not provided
    if overlap_length is None:
        overlap_length = patch_length // 2
    
    if overlap_length >= patch_length:
        raise ValueError(f"overlap_length ({overlap_length}) must be less than patch_length ({patch_length})")
    
    # Calculate starting indices for each patch
    stride = patch_length - overlap_length
    start_indices = list(range(0, axis_length - patch_length + 1, stride))
    
    # Handle the case where the last patch doesn't align perfectly
    if start_indices[-1] + patch_length < axis_length:
        start_indices.append(axis_length - patch_length)
    
    # Create slices for each patch
    patches = []
    patch_indices = []
    for start_idx in start_indices:
        end_idx = start_idx + patch_length
        
        # Create slice objects for all dimensions
        slices = [slice(None)] * full_volume.ndim
        slices[patchify_axis] = slice(start_idx, end_idx)
        
        # Extract patch
        patch = full_volume[tuple(slices)]
        patches.append(patch)
        patch_indices.append((start_idx, end_idx))
    
    return patches, patch_indices

def merge_patches(patches, patch_indices, patchify_axis=2, output_size=None):
    """
    Merge a list of overlapping patches back into a single tensor.
    Overlapping regions are averaged.

    Args:
        patches (list[torch.Tensor]): List of tensor patches to merge.
        patch_indices (list[tuple[int, int]]): List of (start_idx, end_idx) for each patch.
        patchify_axis (int, optional): Axis along which patches were created.
            Defaults to 2.
        output_size (int, optional): Size of output tensor along patchify_axis.
            If None, calculated from patch indices.

    Returns:
        torch.Tensor: Merged tensor with same shape as original input.

    Example:
        >>> patches, indices = patchify(x, patch_length=16, patchify_axis=2)
        >>> merged = merge_patches(patches, indices, patchify_axis=2)
        >>> assert merged.shape == x.shape
    """
    if not patches or not patch_indices:
        raise ValueError("Empty patches or indices provided")
    
    # Get the shape of the output tensor
    output_shape = list(patches[0].shape)
    if output_size is None:
        output_size = max(end_idx for _, end_idx in patch_indices)
    output_shape[patchify_axis] = output_size
    
    # Initialize output tensor and weight tensor (for averaging)
    output = torch.zeros(output_shape, dtype=patches[0].dtype, device=patches[0].device)
    weights = torch.zeros(output_shape, dtype=patches[0].dtype, device=patches[0].device)
    
    # Add each patch to the output tensor
    for patch, (start_idx, end_idx) in zip(patches, patch_indices):
        # Create slice objects for all dimensions
        slices = [slice(None)] * output.ndim
        slices[patchify_axis] = slice(start_idx, end_idx)
        
        # Add patch values and weights
        output[tuple(slices)] += patch
        weights[tuple(slices)] += 1
    
    # Average overlapping regions
    # Add small epsilon to avoid division by zero
    weights = weights.clamp(min=1e-6)
    output = output / weights
    
    return output

# # Example usage
# def test_patchify_merge():
#     # Create sample data
#     x = torch.randn(1, 3, 31, 64, 64)
    
#     # Patchify
#     patches, indices = patchify(x, patch_length=16, patchify_axis=2)
    
#     # Merge back
#     merged = merge_patches(patches, indices, patchify_axis=2)
    
#     # Verify shape
#     assert merged.shape == x.shape
#     print("Shapes match:", x.shape)
    
#     # Check if values are close (allowing for small numerical differences)
#     print("Max difference:", torch.max(torch.abs(merged - x)))