import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registration.regnet import RegNet
from models.diffusion.video_diffusion_pytorch import Unet3D as VideoUNet3D, GaussianDiffusion as VideoDiffusion
from models.diffusion.ldm_encoder_decoder import Decoder as DiffusionDecoder
from models.reconstruction.UNetR.UNetR import UNETR as UNetR
from models.reconstruction.ViViT.vivit_regression import ViViTReg 
from models.reconstruction.TransUNet3D.nn_transunet.networks.nnunet_model import Generic_UNet as TransUNet3D
from models.reconstruction.StrainNet.model import Unet as StrainNet


def load_pretrained_model(model, pretrained_model_path, strict=True):
    print(f'Loading pretrained model from {pretrained_model_path}')
    pretrained_model_state_dict = torch.load(pretrained_model_path)
    model.load_state_dict(
        pretrained_model_state_dict, strict=strict)
    return model

def build_model(model_config, all_config=None, skip_load_pretrained=False):
    if model_config['type'] == 'RegNet':
        # from models.registration.regnet import RegNet
        model = RegNet(model_config)
        if model_config.get('load_pretrained', False) and not skip_load_pretrained:
            model = load_pretrained_model(model, model_config['pretrained_model_path'])
        return model
    elif model_config['type'] == 'VideoDiffusion':
        unet_config = model_config['UNet']
        diffusion_config = model_config['Diffusion']
        unet = VideoUNet3D(**unet_config)
        if unet_config.get('load_pretrained', False):
            unet = load_pretrained_model(unet, unet_config['pretrained_model_path'])
        diffusion = VideoDiffusion(unet, **diffusion_config)
        if diffusion_config.get('load_pretrained', False) and not skip_load_pretrained:
            diffusion = load_pretrained_model(diffusion, diffusion_config['pretrained_model_path'], strict=False)

        if model_config['Diffusion'].get('beta_schedule_overwrite', False):
            
            # diffusion.beta_schedule = model_config['Diffusion']['beta_schedule']
            from models.diffusion.video_diffusion_pytorch import linear_beta_schedule, cosine_beta_schedule            
            # betas = linear_beta_schedule(model.betas.shape[0], scale=model_config['Diffusion']['beta_schedule_scale'])
            timesteps = diffusion.betas.shape[0]
            beta_schedule_method = model_config['Diffusion']['beta_schedule_method']
            beta_schedule_scale = model_config['Diffusion']['beta_schedule_scale']
            print(f'Overwriting beta schedule with {beta_schedule_method} schedule with scale {beta_schedule_scale}')
            if beta_schedule_method == 'linear':
                betas = linear_beta_schedule(timesteps, scale=beta_schedule_scale)
            elif beta_schedule_method == 'cosine':
                betas = cosine_beta_schedule(timesteps)
            else:
                raise ValueError(f"Unrecognized beta schedule method: {beta_schedule_method}")
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

            diffusion.betas = betas#.to(device)
            diffusion.alphas_cumprod = alphas_cumprod#.to(device)
            diffusion.alphas_cumprod_prev = alphas_cumprod_prev#.to(device)
            diffusion.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)#.to(device)
            diffusion.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)#.to(device)
            diffusion.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)#.to(device)
            diffusion.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)#.to(device)
            diffusion.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)#.to(device)
        return diffusion
    elif model_config['type'] == 'DiffusionDecoder':
        model = DiffusionDecoder(**model_config)
        if model_config.get('load_pretrained', False) and not skip_load_pretrained:
            model = load_pretrained_model(model, model_config['pretrained_model_path'])
        return model
    elif model_config['type'] == 'UNetR':
        model = UNetR(**model_config)
        if model_config.get('load_pretrained', False):
            model = load_pretrained_model(model, model_config['pretrained_model_path'])
        return model
    elif model_config['type'] == 'ViViTReg':
        model = ViViTReg(**model_config)
        return model
    elif model_config['type'].lower() == 'strainnet':
        model = StrainNet(**model_config)
        return model
    elif model_config['type'].lower() == 'transunet3d':
        class InitWeights_He(object):
            def __init__(self, neg_slope=1e-2):
                self.neg_slope = neg_slope

            def __call__(self, module):
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                    module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
                    if module.bias is not None:
                        module.bias = nn.init.constant_(module.bias, 0)
        default_dict = {
            "input_channels": 1,
            "base_num_features": 32,
            "num_classes": 2,
            "num_pool": 4,
            "num_conv_per_stage": 2,
            "feat_map_mul_on_downscale": 2,
            "conv_op": nn.Conv3d,
            "norm_op": nn.BatchNorm3d,
            "norm_op_kwargs": {'eps': 1e-05, 'affine': True},
            "dropout_op": nn.Dropout2d,
            "dropout_op_kwargs": {'p': 0, 'inplace': True},
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {'negative_slope': 0.01, 'inplace': True},
            "deep_supervision": False,
            "dropout_in_localization": False,
            "final_nonlin": lambda x: x,
            "weightInitializer": InitWeights_He(1e-2),
            "pool_op_kernel_sizes": [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
            "conv_kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "upscale_logits": False,
            "convolutional_pooling": True,
            "convolutional_upsampling": True,
        }
    
        default_dict.update(model_config)
        model = TransUNet3D(**default_dict)
        if model_config.get('load_pretrained', False):
            model = load_pretrained_model(model, model_config['pretrained_model_path'])
        return model
    else:
        raise ValueError(f'Unknown model type: {model_config["type"]}')