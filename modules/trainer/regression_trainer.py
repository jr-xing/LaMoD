import wandb
import torch
import numpy as np
import datetime, json, copy
import random
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from modules.loss import LossCalculator
from modules.trainer.base_trainer import BaseTrainer
import random
from scipy import ndimage
from models.diffusion.ddpm import q_sample, p_sample, p_sample_loop
from modules.data.processing.rotation import rot_img_seq, rotate_displacement_field_seq
# lm.FluidMetric(self.fluid_params)
import lagomorph as lm
from train_denoisier_video_diffusion import latent_1_std, latent_std_scaler, latent_log_scaler, latent_plus_mins_1_scaler, latent_standardize, latent_1_std_reverse, latent_std_reverse, latent_log_reverse, latent_plus_mins_1_reverse, latent_unstandardize

class DummyLrScheduler:
    """
    Dummy learning rate scheduler that does nothing
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

def latent_std_scaler(z):
    return (z/z.std()), z.std()

def latent_std_reverse(z, sd):
    return z*sd

def latent_log_scaler(z):
    z_ = torch.exp(z)
    return latent_std_scaler(z_)

def latent_log_reverse(z, sd):
    z_ = latent_std_reverse(z,sd)
    return torch.log(z_) 

class DisplacementFieldResizer:
    """Resize the displacement field to the target size using pytorch functionals
    e.g. in 2D case, (N, 2, H, W) -> (N, 2, H_new, W_new)
         in 2D+T case, (N, 2, T, H, W) -> (N, 2, T, H_new, W_new)
         in 3D case, (N, 2, D, H, W) -> (N, 2, D_new, H_new, W_new)    
    """
    def __init__(self, target_size, interpolation_mode='nearest', twoD_plus_T=True):
        self.target_size = target_size
        self.twoD_plus_T = twoD_plus_T
        self.interpolation_mode = interpolation_mode

    def __call__(self, displacement_field):
        if displacement_field.dim() == 4:
            return self.resize_2d(displacement_field)
        elif displacement_field.dim() == 5 and not self.twoD_plus_T:
            return self.resize_3d(displacement_field)
        elif displacement_field.dim() == 5 and self.twoD_plus_T:
            return self.resize_2d_plus_T(displacement_field)
        else:
            raise ValueError(f'displacement_field dim {displacement_field.dim()} not supported')
        
    def resize_2d(self, displacement_field):
        return torch.nn.functional.interpolate(displacement_field, size=self.target_size, mode='bilinear', align_corners=False)
    
    def resize_2d_plus_T(self, displacement_field):
        N, C, T, H, W = displacement_field.shape
        H_new, W_new = self.target_size
        # resize each time frame separately
        resized_displacement_field = torch.zeros(N, C, T, H_new, W_new, device=displacement_field.device)
        for t in range(T):
            resized_displacement_field[:,:,t] = torch.nn.functional.interpolate(
                displacement_field[:,:,t], 
                size=(H_new, W_new), 
                mode=self.interpolation_mode)
        return resized_displacement_field
    
    
    def resize_3d(self, displacement_field):
        return torch.nn.functional.interpolate(displacement_field, size=self.target_size, mode='trilinear', align_corners=False)

# import lagomorph as lm
# from models.registration.EpdiffLib import Epdiff
from modules.data.processing.rotation_module import ImageSeqRotator, DisplacementFieldSeqRotator
class RegressionTrainer(BaseTrainer):
    def __init__(self, trainer_config, device=None, full_config=None):
        super().__init__(trainer_config, device=device, full_config=full_config)
        self.displacement_field_resizer = DisplacementFieldResizer(
            target_size=(48, 48), 
            interpolation_mode='nearest',
            twoD_plus_T=True)
        self.displacement_field_resizer_128 = DisplacementFieldResizer(
            target_size=(128, 128), 
            interpolation_mode='bilinear',
            twoD_plus_T=True)
        self.init_rotation_modules(trainer_config, device)
                
    def init_rotation_modules(self, trainer_config, device):
        # Pre-compute the rotation utils 
        self.rotation_angles = trainer_config.get('rotation_angles', [0, np.pi/2, np.pi, -np.pi/2])
        self.img_seq_rotators = []
        self.disp_seq_rotators = []
        for theta in self.rotation_angles:
            img_seq_rotator = ImageSeqRotator(theta, device, torch.float32, torch.Size((2, 1, 49, 128, 128)))
            disp_seq_rotator = DisplacementFieldSeqRotator(theta, device, torch.float32, torch.Size((2, 2, 49, 48, 48)))
            self.img_seq_rotators.append(img_seq_rotator)
            self.disp_seq_rotators.append(disp_seq_rotator)

    def batch_forward(self, batch, models, loss_name_prefix, mode='train', train_config={}, full_config={}, curr_epoch=-1):
        batch_forward_method = train_config.get('batch_forward_method', 'myo_mask')
        if batch_forward_method in ['img', 'myo_mask']:
            # Use only image or binary mask as input
            # This is for pre-training the registration model or training the baseline methods
            return self.batch_forward_img(batch, models, loss_name_prefix, mode, train_config, full_config, curr_epoch)
        elif batch_forward_method == 'reg_disp_pred':
            # First extract latent motion features and then reconstruct the final motion
            # No diffusion model included. This is just for pre-training the motion encoder.
            return self.batch_forward_reg_disp_pred(batch, models, loss_name_prefix, mode, train_config, full_config, curr_epoch)    
        elif batch_forward_method == 'reg_disp_pred_diffusion':
            # First extract latent motion features, then refine the features throught the diffusion process, and finally reconstruct the final motion.
            # This is for jointly training our proposed LaMoD Method
            return self.batch_forward_reg_disp_pred_diffusion(batch, models, loss_name_prefix, mode, train_config, full_config, curr_epoch)    
        else:
            raise ValueError(f'batch_forward_method {batch_forward_method} not implemented')
    
    def batch_forward_reg_disp_pred(self, batch, models, loss_name_prefix, mode='train', train_config={}, full_config={}, curr_epoch=-1):
        motion_regression_model = models[train_config.get('reg_model_name', 'motion_regression')]
        reg_model = models['registration']

        src, tar = batch['src'], batch['tar']
        src = src.to(self.device, dtype=torch.float32)
        tar = tar.to(self.device, dtype=torch.float32)
        
        if mode in ['training', 'train']:
            DENSE_disp_GT = batch['DENSE_disp']
        elif mode == 'test':
            DENSE_disp_GT = batch['DENSE_disp'] if 'DENSE_disp' in batch.keys() else torch.zeros_like(tar)
        DENSE_disp_GT = DENSE_disp_GT.to(self.device, dtype=torch.float32)

        if train_config.get('disp_masking', False):
            DENSE_mask = torch.abs(DENSE_disp_GT)> 1e-10

        if train_config.get('enable_random_rotate', False) and mode in ['training', 'train']:
            random_rotate_prob_thres = train_config.get('random_rotate_prob_thres', 0.3)
            random_rotate_prob = np.random.uniform(0, 1)
            if random_rotate_prob < random_rotate_prob_thres:
                pass
            else:
                rotator_idx = random.choice(range(len(self.rotation_angles)))                
                src = self.img_seq_rotators[rotator_idx](src)
                tar = self.img_seq_rotators[rotator_idx](tar)
                DENSE_disp_GT = self.disp_seq_rotators[rotator_idx](DENSE_disp_GT)                
                

        reg_forward_data = train_config.get('reg_forward_data', 'displacement_field')
        with torch.no_grad():
            reg_pred_dict = reg_model(src, tar)

            if reg_forward_data in ['displacement_field', 'disp']:
                reg_pred = reg_pred_dict['displacement'] # should have shape [N, 2, T, H, W] = [N, 2, T, 128, 128] 
                reg_pred_disp = reg_pred
            elif reg_forward_data == 'latent':
                reg_pred = reg_pred_dict['latent']
                reg_pred_disp = reg_pred_dict['displacement']
        
        if train_config.get('resize_before_regression', False):
            reg_pred = self.displacement_field_resizer(reg_pred)
        
        regression_output = motion_regression_model(reg_pred)

        regression_output_type = train_config.get('regression_output_type', 'disp')
        if regression_output_type in ['displacement_field', 'disp']:
            DENSE_disp_pred = regression_output
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
            # ui = torch.stack(torch.split(ui_2D, batch_size, dim=0), dim=2)
            DENSE_disp_pred = ui
        else:
            raise ValueError(f'pred_type {regression_output_type} not implemented')
        
        if train_config.get('resize_before_masking', False):
            DENSE_disp_pred = self.displacement_field_resizer_128(DENSE_disp_pred)
            DENSE_disp_GT = self.displacement_field_resizer_128(DENSE_disp_GT)
        if train_config.get('disp_masking', False):
            DENSE_disp_pred = DENSE_disp_pred * DENSE_mask

        if train_config.get('mask_padded_frames', False):
            
            vol_ori_n_frames = batch['ori_n_frames']
            # print(f'Masking Padded! {vol_ori_n_frames}')
            for i, ori_n_frames in enumerate(vol_ori_n_frames):
                DENSE_disp_pred[i, :, ori_n_frames:] = 0
                DENSE_disp_GT[i, :, ori_n_frames:] = 0
        
        pred_dict = {
            'reg_disp': reg_pred_disp,
            # 'DENSE_mask': DENSE_mask,
            # 'DENSE_disp': DENSE_disp_pred,
            # 'tar': Sdef,
            'DENSE_disp': DENSE_disp_pred,
        }
        target_dict = {
            'src': src,
            'tar': tar,
            # 'DENSE_disp': DENSE_disp_GT,
            'DENSE_disp': DENSE_disp_GT,
        }
        # return pred_dict, target_dict
        
        total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)
        with torch.no_grad():
            total_error, error_values_dict = self.evaluator(pred_dict, target_dict)

        # update the epoch loss dict
        batch_loss_dict = {}
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            batch_loss_dict[record_name] = loss_value
        
        # update the epoch error 
        batch_error_dict = {}
        for error_name, error_value in error_values_dict.items():
            record_name = f'{loss_name_prefix}/{error_name}'
            batch_error_dict[record_name] = error_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, batch_loss_dict, batch_error_dict, pred_dict, target_dict
    

    def batch_forward_reg_disp_pred_diffusion(self, batch, models, loss_name_prefix, mode='train', train_config={}, full_config={}, curr_epoch=-1):        
        reg_model = models['registration']
        diffusion_model = models['latent']
        motion_regression_model = models[train_config.get('reg_model_name', 'motion_regression')]

        # --------------------------------------------------#
        # ------------------ Registration ------------------#
        # --------------------------------------------------#
        src, tar = batch['src'], batch['tar']
        src = src.to(self.device, dtype=torch.float32)
        tar = tar.to(self.device, dtype=torch.float32)
        
        if mode in ['training', 'train']:
            DENSE_disp_GT = batch['DENSE_disp']
        elif mode == 'test':
            DENSE_disp_GT = batch['DENSE_disp'] if 'DENSE_disp' in batch.keys() else torch.zeros_like(tar)
        DENSE_disp_GT = DENSE_disp_GT.to(self.device, dtype=torch.float32)

        if train_config.get('disp_masking', False):
            DENSE_mask = torch.abs(DENSE_disp_GT)> 1e-10
            # DENSE_disp_pred = DENSE_disp_pred * DENSE_mask

        if train_config.get('enable_random_rotate', False) and mode in ['training', 'train']:
            random_rotate_prob_thres = train_config.get('random_rotate_prob_thres', 0.3)
            random_rotate_prob = np.random.uniform(0, 1)
            # print(f'random rotate prob: {random_rotate_prob}')
            if random_rotate_prob < random_rotate_prob_thres:
                pass
            else:
                # theta_list = [0., np.pi/2, np.pi, -np.pi/2]
                # theta_idx = random.choice(len(self.rotation_angles))
                # src = self.img_seq_rotators[theta_idx](src)
                # tar = self.img_seq_rotators[theta_idx](tar)
                # DENSE_disp_GT = self.disp_seq_rotators[theta_idx](DENSE_disp_GT)
                # rotator_idx = 1
                rotator_idx = random.choice(range(len(self.rotation_angles)))                
                src = self.img_seq_rotators[rotator_idx](src)
                tar = self.img_seq_rotators[rotator_idx](tar)
                DENSE_disp_GT = self.disp_seq_rotators[rotator_idx](DENSE_disp_GT)
                

        reg_forward_data = train_config.get('reg_forward_data', 'displacement_field')
        with torch.no_grad():
            reg_pred_dict = reg_model(src, tar)

            if reg_forward_data in ['displacement_field', 'disp']:
                reg_pred = reg_pred_dict['displacement'] # should have shape [N, 2, T, H, W] = [N, 2, T, 128, 128] 
                reg_pred_disp = reg_pred
            elif reg_forward_data == 'latent':
                reg_pred = reg_pred_dict['latent']
                reg_pred_disp = reg_pred_dict['displacement']
            # reg_disp_pred = reg_disp_pred.roll(shifts=1, dims=1)
        
        
        
        # --------------------------------------------------#
        # -------------------- Diffusion -------------------#
        # --------------------------------------------------#
        diffusion_normalize_method = train_config.get('diffusion_normalize_method', 'standardize')
        if diffusion_normalize_method == '1_std':
            z_norm, z_mean, z_std, z_min, z_max = latent_1_std(reg_pred)
        elif diffusion_normalize_method == 'std_scaler':
            z_norm, z_std = latent_std_scaler(reg_pred)
        elif diffusion_normalize_method == 'none':
            z_norm = reg_pred
        elif diffusion_normalize_method == 'log_scaler':
            z_norm, z_std = latent_log_scaler(reg_pred)
        elif diffusion_normalize_method == 'plus_mins_1':
            z_norm, z_min, z_max = latent_plus_mins_1_scaler(reg_pred)
        elif diffusion_normalize_method == 'standardize':
            z_norm, z_mean, z_std = latent_standardize(reg_pred)
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
            z_denoised = latent_plus_mins_1_reverse(z_denoised_norm, z_min, z_max)
        elif diffusion_normalize_method == '1_std':
            z_denoised = latent_1_std_reverse(z_denoised_norm, z_mean, z_std, z_min, z_max)
        elif diffusion_normalize_method == 'std_scaler':
            z_denoised = latent_std_reverse(z_denoised_norm, z_std)
        elif diffusion_normalize_method == 'log_scaler':
            z_denoised = latent_log_reverse(z_denoised_norm, z_std)
        elif diffusion_normalize_method == 'standardize':    
            z_denoised = latent_unstandardize(z_denoised_norm, z_mean, z_std)
        else:
            raise ValueError(f'Invalid diffusion_normalize_method: {diffusion_normalize_method}')

        # --------------------------------------------------#
        # ----------------- Reconstruction -----------------#
        # --------------------------------------------------#
        if train_config.get('resize_before_regression', False):
            z_denoised = self.displacement_field_resizer(z_denoised)
        regression_output = motion_regression_model(z_denoised)
        # pred_noise = pred_noise# * src_tar_union_mask

        regression_output_type = train_config.get('regression_output_type', 'disp')
        if regression_output_type in ['displacement_field', 'disp']:
            DENSE_disp_pred = regression_output
        elif regression_output_type == 'momemtum':
            raise NotImplemented(f'pred_type {regression_output_type} not implemented')
        elif regression_output_type == 'velocity':
            velo_regression = regression_output
            # if train_config.get('disp_masking', False):
            #     velo_mask = torch.abs(DENSE_disp_GT)> 1e-10
            #     # velo_mask.requires_grad = True
                # velo_regression = velo_regression * DENSE_mask
            # velo_regression = reg_pred_dict['velocity']
            batch_size, _, T, H, W = velo_regression.shape
            v_2D = torch.concat([velo_regression[:,:,t] for t in range(T)], dim=0)
            v_2D_YX = v_2D.permute(0,1,3,2)
            m_2D_YX = reg_model.metric.flat(v_2D_YX)
            u_2D_YX_seq, v_2D_YX_seq, m_2D_YX_seq, ui_YX_2D_seq = \
                reg_model.MEpdiff.my_expmap( m_2D_YX, num_steps=reg_model.TSteps)
            ui_2D = ui_YX_2D_seq[-1].permute(0,1,3,2)
            ui = torch.stack([ui_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            # ui = torch.stack(torch.split(ui_2D, batch_size, dim=0), dim=2)
            DENSE_disp_pred = ui
        else:
            raise ValueError(f'pred_type {regression_output_type} not implemented')
        if train_config.get('resize_before_masking', False):
            DENSE_disp_pred = self.displacement_field_resizer_128(DENSE_disp_pred)
            DENSE_disp_GT = self.displacement_field_resizer_128(DENSE_disp_GT)
        if train_config.get('disp_masking', False):
            DENSE_disp_pred = DENSE_disp_pred * DENSE_mask
        
        if train_config.get('mask_padded_frames', False):
            # print('Masking Padded!')
            vol_ori_n_frames = batch['ori_n_frames']
            for i, ori_n_frames in enumerate(vol_ori_n_frames):
                DENSE_disp_pred[i, :, ori_n_frames:] = 0
                DENSE_disp_GT[i, :, ori_n_frames:] = 0

        pred_dict = {
            'reg_disp': reg_pred_disp,
            'noise': pred_noise,
            # 'DENSE_mask': DENSE_mask,
            # 'DENSE_disp': DENSE_disp_pred,
            # 'tar': Sdef,
            'DENSE_disp': DENSE_disp_pred,
        }
        target_dict = {
            'src': src,
            'tar': tar,
            'noise': noise,
            # 'DENSE_disp': DENSE_disp_GT,
            'DENSE_disp': DENSE_disp_GT,
        }
        # return pred_dict, target_dict
        
        # total_loss = 0
        total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)
        with torch.no_grad():
            total_error, error_values_dict = self.evaluator(pred_dict, target_dict)

        # update the epoch loss dict
        batch_loss_dict = {}
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            batch_loss_dict[record_name] = loss_value
        
        # update the epoch error 
        batch_error_dict = {}
        for error_name, error_value in error_values_dict.items():
            record_name = f'{loss_name_prefix}/{error_name}'
            batch_error_dict[record_name] = error_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, batch_loss_dict, batch_error_dict, pred_dict, target_dict    
    
    def batch_forward_img(self, batch, models, loss_name_prefix, mode='train', train_config={}, full_config={}, curr_epoch=-1, force_augment=False):
        motion_regression_model = models['motion_regression']

        src, tar = batch['src'], batch['tar']
        src = src.to(dtype=torch.float32)
        tar = tar.to(dtype=torch.float32)
        # pair = torch.cat([src, tar], dim=1).to(dtype=torch.float32)

        if mode in ['training', 'train']:
            DENSE_disp_GT = batch['DENSE_disp']
        elif mode == 'test':
            DENSE_disp_GT = batch['DENSE_disp'] if 'DENSE_disp' in batch.keys() else torch.zeros_like(DENSE_disp_pred)
        DENSE_disp_GT = DENSE_disp_GT.to(self.device, dtype=torch.float32)

        # force_augment = True
        if (train_config.get('enable_random_rotate', False) and mode in ['training', 'train']) or force_augment:
            random_rotate_prob_thres = train_config.get('random_rotate_prob_thres', 0.3)
            random_rotate_prob = np.random.uniform(0, 1)
            # print(f'random rotate prob: {random_rotate_prob}')
            if (random_rotate_prob < random_rotate_prob_thres) and not force_augment:
                pass
            else:
                
                # print(f'{theta=}')
                # print(f'randomly rotate {theta} degree')
                # pair = rot_img_seq(pair, theta=theta, dtype=src.dtype, device=src.device)
                rotator_idx = random.choice(range(len(self.rotation_angles)))
                # rotator_idx = 1
                src = self.img_seq_rotators[rotator_idx](src)
                tar = self.img_seq_rotators[rotator_idx](tar)
                DENSE_disp_GT = self.disp_seq_rotators[rotator_idx](DENSE_disp_GT)
                
                # theta_list = [0., np.pi/2, np.pi, -np.pi/2]
                # theta_list = [np.pi/2]
                # theta = random.choice(theta_list)
                # DENSE_disp_pred = rotate_displacement_field_seq(DENSE_disp_pred, theta=theta, dtype=DENSE_disp_pred.dtype, device=DENSE_disp_pred.device)
                # src = rot_img_seq(src, theta=theta, dtype=src.dtype, device=src.device)
                # tar = rot_img_seq(tar, theta=theta, dtype=src.dtype, device=src.device)
                # DENSE_disp_GT = rotate_displacement_field_seq(DENSE_disp_GT, theta=theta, dtype=DENSE_disp_GT.dtype, device=DENSE_disp_GT.device)
        
        pair = torch.cat([src, tar], dim=1).to(dtype=torch.float32)
        pair_resized = self.displacement_field_resizer(pair).to(self.device)

        DENSE_disp_pred = motion_regression_model(pair_resized)
        

        # if mode in ['training', 'train']:
        #     DENSE_disp_GT = batch['DENSE_disp']
        # elif mode == 'test':
        #     DENSE_disp_GT = batch['DENSE_disp'] if 'DENSE_disp' in batch.keys() else torch.zeros_like(DENSE_disp_pred)
        DENSE_disp_GT = DENSE_disp_GT.to(self.device, dtype=torch.float32)

        if train_config.get('resize_before_masking', False):
            DENSE_disp_pred = self.displacement_field_resizer_128(DENSE_disp_pred)
            DENSE_disp_GT = self.displacement_field_resizer_128(DENSE_disp_GT)

        if train_config.get('disp_masking', False):
            DENSE_mask = torch.abs(DENSE_disp_GT)> 1e-10
            DENSE_disp_pred = DENSE_disp_pred * DENSE_mask
            # pred_noise = pred_noise# * src_tar_union_mask


        pred_dict = {
            # 'reg_disp': reg_disp_pred,
            # 'DENSE_mask': DENSE_mask,
            'src': src,
            'tar': tar,
            'DENSE_disp': DENSE_disp_pred,
        }
        target_dict = {
            'pair': pair,
            'DENSE_disp': DENSE_disp_GT,
        }
        # return pred_dict, target_dict
        if mode in ['training', 'train']:
            total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)
        elif mode == 'test':
            with torch.no_grad():
                total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)
        
        with torch.no_grad():
            total_error, error_values_dict = self.evaluator(pred_dict, target_dict)

        # update the epoch loss dict
        batch_loss_dict = {}
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            batch_loss_dict[record_name] = loss_value
        
        # update the epoch error 
        batch_error_dict = {}
        for error_name, error_value in error_values_dict.items():
            record_name = f'{loss_name_prefix}/{error_name}'
            batch_error_dict[record_name] = error_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, batch_loss_dict, batch_error_dict, pred_dict, target_dict

    
    def batch_eval(self, batch_pred, batch):
        return {}

    def plot_results(self, data: list, save_fig_dir: str or Path or None=None, save_plot=False, plot_frame_idx=10, n_plot = 5, n_plot_patients=-1, DENSE_quiver_scale=75, reg_quiver_scale=200):
        data_to_plot = data[:n_plot] if n_plot > 0 else data
        for datum in data_to_plot:
            patient_id = datum['subject_id']
            slice_full_name = f"{datum['subject_id']}-{Path(datum['DENSE_slice_mat_filename']).stem}"
            fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=False)
            # vis_idx = 30
            vis_frame_idx = 5
            # scale = 45
            axs[0,0].imshow(datum['src'][0,vis_frame_idx].astype(float), cmap='gray')
            axs[0,0].set_title('Source image')
            axs[0,0].invert_yaxis()
            axs[0,1].imshow(datum['tar'][0,vis_frame_idx].astype(float), cmap='gray')
            axs[0,1].set_title('Target image')
            axs[0,1].invert_yaxis()            # reverse the y axis
            axs[0,2].imshow(datum['tar'][0,vis_frame_idx].astype(float) - datum['src'][0,vis_frame_idx].astype(float), cmap='gray')
            axs[0,2].set_title('Target - Source image')
            axs[0,2].invert_yaxis()            # reverse the y axis

            axs[1,0].quiver(datum['DENSE_disp'][0,vis_frame_idx], datum['DENSE_disp'][1,vis_frame_idx], scale=DENSE_quiver_scale)
            axs[1,0].set_title('GT DENSE displacement')
            axs[1,1].quiver(datum['DENSE_disp_pred'][0,vis_frame_idx], datum['DENSE_disp_pred'][1,vis_frame_idx], scale=DENSE_quiver_scale)
            axs[1,1].set_title('Predicted DENSE displacement')
            # hide axis[1,2]
            axs[1,2].axis('off')

            # axs[1,2].quiver(datum['registration_disp_pred'][1,vis_frame_idx], datum['registration_disp_pred'][0,vis_frame_idx], scale=reg_quiver_scale)
            # axs[1,2].set_title('Predicted registration displacement')
            # make all aspect ratio equal
            for ax in axs.flatten():
                ax.set_aspect('equal')
            
            fig.suptitle(f'{slice_full_name}')
            if save_plot:
                plt.savefig(Path(save_fig_dir) / f'{slice_full_name}.png')

    
    def plot_results_splitted_frames(self, data: list, save_fig_dir: str or Path or None=None, save_plot=False, plot_frame_idx=10, n_plot = -1, n_plot_patients=-1, quiver_scale=75):
        # collect the data to plot
        data_to_plot = []
        unique_patient_ids = list(set([data['subject_id'] for data in data]))
        unique_patient_ids = unique_patient_ids[:n_plot_patients] if n_plot_patients > 0 else unique_patient_ids
        for patient_id in unique_patient_ids:
            patient_data = [data for data in data if data['subject_id'] == patient_id]
            patient_unique_slice_indices = list(set([int(data['src_slice_idx']) for data in patient_data]))
            for slice_idx in patient_unique_slice_indices:
                slice_data = [data for data in patient_data if data['src_slice_idx'] == slice_idx]
                # slice_data = sorted(slice_data, key=lambda x: x['timepoint'])
                slice_datum_plot = slice_data[plot_frame_idx]
                DENSE_error = np.linalg.norm(slice_datum_plot['DENSE_disp'] - slice_datum_plot['DENSE_disp_pred']).mean()
                data_to_plot.append({
                    'patient_id': patient_id,
                    'slice_idx': slice_idx,
                    'src_slice_idx': slice_datum_plot['src_slice_idx'],
                    'DENSE_disp': slice_datum_plot['DENSE_disp'],
                    'DENSE_disp': slice_datum_plot['DENSE_disp'],
                    'DENSE_disp_pred': slice_datum_plot['DENSE_disp_pred'],
                    'registration_disp_pred': slice_datum_plot['registration_disp_pred'],
                    'DENSE_error': DENSE_error,
                })
        data_to_plot = data_to_plot[:n_plot] if n_plot > 0 else data_to_plot
        
        # quiver_scale = 75
        # plot the data
        for patient_id in unique_patient_ids:
            patient_data_to_plot = [datum for datum in data_to_plot if datum['patient_id'] == patient_id]
            fig, axs = plt.subplots(
                3, len(patient_data_to_plot), 
                figsize=(3*len(patient_data_to_plot), 10), 
                sharex=False, sharey=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.05)
            disp_Y, disp_X = np.mgrid[0:patient_data_to_plot[0]['DENSE_disp'].shape[1], 0:patient_data_to_plot[0]['DENSE_disp'].shape[2]]
            reg_Y, reg_X = np.mgrid[0:patient_data_to_plot[0]['registration_disp_pred'].shape[1], 0:patient_data_to_plot[0]['registration_disp_pred'].shape[2]]
            for i, datum in enumerate(patient_data_to_plot):
                axs[0, i].quiver(disp_X, disp_Y, datum['DENSE_disp'][0], datum['DENSE_disp'][1], scale=quiver_scale)
                axs[0, i].set_title(f'GT SL{i} FR{plot_frame_idx}')
                axs[1, i].quiver(disp_X, disp_Y, datum['DENSE_disp_pred'][0], datum['DENSE_disp_pred'][1], scale=quiver_scale)
                axs[1, i].set_title('Pred E={:.2f}'.format(datum['DENSE_error']))
                axs[2, i].quiver(reg_X, reg_Y, datum['registration_disp_pred'][1], datum['registration_disp_pred'][0], scale=quiver_scale//2)
                axs[2, i].set_title('Reg')
            fig.suptitle(f'{patient_id}')
            if save_plot:
                plt.savefig(Path(save_fig_dir) / f'{patient_id}.png')

            # patient_unique_slice_indices = list(set([datum['src_slice_idx'] for datum in patient_data_to_plot]))
            # for slice_idx in patient_unique_slice_indices:
            #     slice_data_to_plot = [datum for datum in patient_data_to_plot if datum['src_slice_idx'] == slice_idx]
            #     slice_data_to_plot = sorted(slice_data_to_plot, key=lambda x: x['timepoint'])
            #     fig = plt.figure(figsize=(10, 10))
            #     for i, datum in enumerate(slice_data_to_plot):
            #         ax = fig.add_subplot(2, 5, i+1)
            #         ax.imshow(datum['DENSE_disp_pred'][0,0,...].cpu().numpy(), cmap='gray')
            #         ax.set_title(f'timepoint: {datum["timepoint"]}')
            #     fig.suptitle(f'patient_id: {patient_id}, slice_idx: {slice_idx}')
            #     plt.savefig(save_fig_dir / f'patient_id_{patient_id}_slice_idx_{slice_idx}.png')
            #     plt.close(fig)
    def test_inference(self, data, models, test_config={}, full_config={}, wandb_experiment=None):
        inference_result_dicts = []
        for datum in data:
            inference_result_dict = self.inference_datum(
                datum=datum, 
                model=models, 
                device=self.device, 
                frame_patchify_n_frames=8,
                test_config=test_config, 
                full_config=full_config)
            inference_result_dicts.append(inference_result_dict)
            datum.update(inference_result_dict)
            if wandb_experiment is not None:
                wandb.log({
                    'final-test-full/DENSE_reconstruction_error': inference_result_dict['DENSE_disp_error'],
                })
        return inference_result_dicts

    def inference_datum(self, datum, model, device, frame_patchify_n_frames=8, test_config={}, full_config={}):
        src_frame = datum['myo_masks'][...,0]
        # src_volume: repeat the src_frame frame_patchify_n_frames times to form a volume and convert it to tensor
        src_vol = torch.from_numpy(np.repeat(src_frame[None, ...], frame_patchify_n_frames, axis=0)).to(device).float()[None, None]
        src_vol_mask = np.abs(src_vol.cpu().numpy()) > 1e-3
        
        tar_vol_full = datum['myo_masks'][...,1:]
        n_patchs = np.ceil(tar_vol_full.shape[-1] / frame_patchify_n_frames).astype(int)

        # if self doesn't have evalator, create one
        if not hasattr(self, 'evaluator'):
            self.evaluator = LossCalculator(full_config.get('evaluation', {}))
        
        # Prediction (Reconstruction)
        recons_list = []
        reg_disp_list = []
        reg_deformed_list = []
        for patch_idx in range(n_patchs):
            # Patchify the target volume
            start_idx_ori = patch_idx * frame_patchify_n_frames
            end_idx_ori = (patch_idx + 1) * frame_patchify_n_frames

            # if the last patch is not full, using some earlier frames to fill it
            # note that this means the last patch will have some overlap with the previous patch
            # and we need to handle this when merging the patches
            if end_idx_ori > tar_vol_full.shape[-1]:
                patch_actual_n_frames = tar_vol_full.shape[-1] - start_idx_ori
                end_idx = tar_vol_full.shape[-1]
                start_idx = end_idx - frame_patchify_n_frames
            else:
                patch_actual_n_frames = frame_patchify_n_frames
                start_idx = start_idx_ori
                end_idx = end_idx_ori
            tar_vol = np.moveaxis(tar_vol_full[..., start_idx:end_idx], -1, 0)
            tar_vol = torch.from_numpy(tar_vol).to(device).float()[None, None]
            
            # Prediction
            with torch.no_grad():
                # motion_regression_model = models['motion_regression']
                # print('latent_scaling_factor', latent_scaling_factor)

                # src, tar = batch['src'], batch['tar']
                pair = torch.cat([src_vol, tar_vol], dim=1).to(device, dtype=torch.float32)
                recons = model(pair).cpu()
                if test_config.get('disp_masking', False):
                    recons = recons * src_vol_mask
                            
                if end_idx_ori > tar_vol_full.shape[-1]:
                    recons = recons[:,:, -patch_actual_n_frames:]
                
                recons_list.append(recons)

            
        # merging the patches
        recons_full = torch.cat(recons_list, dim=2)[0]
        recons_dict = {
            'DENSE_disp': recons_full.moveaxis(1,-1).numpy()
            }
        target_dict = {
            'DENSE_disp': datum['DENSE_Lag_displacement_field'][...,1:]
        }
        total_error, error_values_dict = self.evaluator(recons_dict, target_dict)

        return_dict = {
            'DENSE_displacement_field_pred': recons_full.moveaxis(1,-1).numpy(),
            # 'DENSE_disp_GT': datum['DENSE_Lag_displacement_field'][...,1:].numpy(),
            'DENSE_disp_error': total_error,
            'error_values_dict': error_values_dict
        }
            
        return return_dict
