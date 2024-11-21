import torch
import torch.nn as nn

from models import build_model
from modules.trainer.normalize import *
from modules.trainer import patchify, merge_patches
class LaMoD(nn.Module):
    def __init__(self, network_config, device='cpu', skip_load_pretrained=False):
        super().__init__()
        self.models = nn.ModuleDict()
        for model_name, model_config in network_config.items():
            self.models[model_name] = build_model(model_config, skip_load_pretrained=skip_load_pretrained, device=device)
            # self.networks[model_name] = self.networks[model_name]#.to(device)
        self.device = device
    # def to_device(self, device):
    #     for network_name, network in self.networks.items():
    #         network.to(device)
    
    def load_checkpoint(self, registration_checkpoint_fname=None, diffusion_checkpoint_fname=None, motion_regression_checkpoint_fname=None):
        if registration_checkpoint_fname is not None:
            self.networks['registration'].load_state_dict(
                registration_checkpoint_fname, strict=False, map_location=self.device)
        if registration_checkpoint_fname is not None:
            self.networks['diffusion'].load_state_dict(
                registration_checkpoint_fname, strict=False, map_location=self.device)
        if registration_checkpoint_fname is not None:
            self.networks['motion_regression'].load_state_dict(
                registration_checkpoint_fname, strict=False, map_location=self.device)

    # def inference(self, video, disp_mask=None, ori_n_frames=None, train_config={}, DENSE_disp=None, skip_diffusion = False, repeats = 1):
    #     if repeats == 1:
    #         return self.inference_once(video, disp_mask, ori_n_frames, train_config, DENSE_disp, skip_diffusion)
    #     else:            
    #         repeat_pred_dicts = []
    #         repeat_target_dicts = []
    #         for repeat_idx in range(repeats):
    #             curr_pred_dict, curr_target_dict = self.inference_once(video, disp_mask, ori_n_frames, train_config, DENSE_disp, skip_diffusion)
    #             repeat_pred_dicts.append(repeat_pred_dicts)
    #             repeat_target_dicts.append(curr_target_dict)

    #         # note: only the different LaMoD need to be stored



    def inference(self, video, disp_mask=None, ori_n_frames=None, train_config={}, DENSE_disp=None, skip_diffusion = False):
        # Prepare data
        # the input video variable should be a pytorch tensor with shape [1, T, H, W] or [N, 1, T, H, W]
        if video.ndim == 4:
            video = video[None]
        N, C, Nfr, H, W = video.shape
        if ori_n_frames is None:
            ori_n_frames = [Nfr]
        src = video[:, :, 0:1].repeat(1, 1, Nfr-1, 1, 1)     # [N, 1, Nfr-1, H, W]
        tar = video[:, :, 1:]                                # [N, 1, Nfr-1, H, W]

        # Prepare models
        reg_model = self.models['registration']
        diffusion_model = self.models['latent']
        motion_regression_model = self.models[train_config.get('reg_model_name', 'motion_regression')]

        # --------------------------------------------------#
        # ------------------ Registration ------------------#
        # --------------------------------------------------#
        src = src.to(self.device, dtype=torch.float32)
        tar = tar.to(self.device, dtype=torch.float32)

        reg_forward_data = train_config.get('reg_forward_data', 'latent')
        with torch.no_grad():
            reg_pred_dict = reg_model.encode(src, tar)

            if reg_forward_data in ['displacement_field', 'disp']:
                reg_pred = reg_pred_dict['displacement'] # should have shape [N, 2, T, H, W] = [N, 2, T, 128, 128] 
                reg_pred_disp = reg_pred
            elif reg_forward_data == 'latent':
                reg_pred = reg_pred_dict['latent']
                reg_pred_disp = reg_pred_dict['displacement']
                        
        # --------------------------------------------------#
        # -------------------- Diffusion -------------------#
        # --------------------------------------------------#        
        if not skip_diffusion:
            diffusion_normalize_method = train_config.get('diffusion_normalize_method', 'standardize')
            diffusion_patchify = train_config.get('diffusion_patchify', True)
            diffusion_patchify_length = train_config.get('diffusion_patchify_length', 16)
            diffusion_patchify_overlap_length = train_config.get('diffusion_patchify_overlap_length', None)
            if diffusion_patchify:
                reg_pred_patches, reg_pred_patch_indices = patchify(reg_pred, diffusion_patchify_length, 2, diffusion_patchify_overlap_length)
            else:
                reg_pred_patches = [reg_pred]
            z_denoised_patches = []
            for curr_reg_pred_patch in reg_pred_patches:
                if diffusion_normalize_method == '1_std':
                    z_norm, z_mean, z_std, z_min, z_max = latent_1_std(curr_reg_pred_patch)
                elif diffusion_normalize_method == 'std_scaler':
                    z_norm, z_std = latent_std_scaler(curr_reg_pred_patch)
                elif diffusion_normalize_method == 'none':
                    z_norm = curr_reg_pred_patch
                elif diffusion_normalize_method == 'log_scaler':
                    z_norm, z_std = latent_log_scaler(curr_reg_pred_patch)
                elif diffusion_normalize_method == 'plus_mins_1':
                    z_norm, z_min, z_max = latent_plus_mins_1_scaler(curr_reg_pred_patch)
                elif diffusion_normalize_method == 'standardize':
                    z_norm, z_mean, z_std = latent_standardize(curr_reg_pred_patch)
                else:
                    raise ValueError(f'Invalid diffusion_normalize_method: {diffusion_normalize_method}')

                noise_loss, z_start, noise, zt, pred_noise = diffusion_model(
                    z_norm, noise_smoothing=True, compute_loss=False)

                z_denoised_norm, predicted_noise_list = diffusion_model.p_sample_loop(
                    shape=zt.shape,
                    cond=None,
                    cond_scale=1.,
                    img=zt,
                    noise=pred_noise,
                    noise_smoothing=True)
                if diffusion_normalize_method == 'plus_mins_1':
                    z_denoised_patch = latent_plus_mins_1_reverse(z_denoised_norm, z_min, z_max)
                elif diffusion_normalize_method == '1_std':
                    z_denoised_patch = latent_1_std_reverse(z_denoised_norm, z_mean, z_std, z_min, z_max)
                elif diffusion_normalize_method == 'std_scaler':
                    z_denoised_patch = latent_std_reverse(z_denoised_norm, z_std)
                elif diffusion_normalize_method == 'log_scaler':
                    z_denoised_patch = latent_log_reverse(z_denoised_norm, z_std)
                elif diffusion_normalize_method == 'standardize':    
                    z_denoised_patch = latent_unstandardize(z_denoised_norm, z_mean, z_std)
                else:
                    raise ValueError(f'Invalid diffusion_normalize_method: {diffusion_normalize_method}')
                z_denoised_patches.append(z_denoised_patch)
            if diffusion_patchify:
                z_denoised = merge_patches(z_denoised_patches, reg_pred_patch_indices)
            else:
                z_denoised = z_denoised_patch
        else:
            z_denoised = reg_pred
            pred_noise = None,
            noise = None
        # --------------------------------------------------#
        # ----------------- Reconstruction -----------------#
        # --------------------------------------------------#
        regression_output = motion_regression_model(z_denoised)

        regression_output_type = train_config.get('regression_output_type', 'disp')
        if regression_output_type in ['displacement_field', 'disp']:
            LaMoD_disp_pred = regression_output
        elif regression_output_type == 'momemtum':
            raise NotImplemented(f'pred_type {regression_output_type} not implemented')
        elif regression_output_type == 'velocity':
            velo_regression = regression_output
            batch_size, _, T, H, W = velo_regression.shape
            v_2D = torch.concat([velo_regression[:,:,t] for t in range(T)], dim=0)
            v_2D_YX = v_2D.permute(0,1,3,2)
            m_2D_YX = reg_model.metric.flat(v_2D_YX)
            u_2D_YX_seq, v_2D_YX_seq, m_2D_YX_seq, ui_YX_2D_seq = \
                reg_model.MEpdiff.my_expmap( m_2D_YX, num_steps=reg_model.TSteps)
            ui_2D = ui_YX_2D_seq[-1].permute(0,1,3,2)
            ui = torch.stack([ui_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            LaMoD_disp_pred = ui
        else:
            raise ValueError(f'pred_type {regression_output_type} not implemented')
        
        if train_config.get('disp_masking', False) and disp_mask is not None:
            LaMoD_disp_pred = LaMoD_disp_pred * disp_mask
        
        # if train_config.get('mask_padded_frames', True):
            # vol_ori_n_frames = batch['ori_n_frames']
        for i, ori_n_frame in enumerate(ori_n_frames):
            LaMoD_disp_pred[i, :, ori_n_frame:] = 0
            # DENSE_disp_GT[i, :, ori_n_frame:] = 0

        pred_dict = {
            'reg_disp': reg_pred_disp,
            'noise': pred_noise,
            'disp_mask': disp_mask,
            'LaMoD_disp': LaMoD_disp_pred,
        }
        target_dict = {
            'src': src,
            'tar': tar,
            'noise': noise,
            'DENSE_disp': DENSE_disp,
        }
        for key, value in pred_dict.items():
            if isinstance(value, torch.Tensor):
                pred_dict[key] = value.detach().cpu()
        for key, value in target_dict.items():
            if isinstance(value, torch.Tensor):
                target_dict[key] = value.detach().cpu()
        return pred_dict, target_dict    