from torch.autograd import Variable
import enum
import imageio as iio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
import torch, random
import math
import numpy as np
import torch as th
from scipy import ndimage
from torchvision import transforms
from torch.nn import HuberLoss
# from tqdm.auto import tqdm

# def re_range(img):
#     mmin=img.min()
#     mmax=img.max()
#     img = (img-img.min())/(img.max()-img.min())
#     img = img*2 -1
#     return img, mmin, mmax

# def un_range(img, mmin, mmax):
#     img = ((img+1)/2)*(mmax-mmin)
#     img+=mmin
#     return img

# def forward_process(img,T,noise):
#     xt = torch.clone(img)
#     alpha1 = torch.sqrt(alpha_bars[T-1])
#     alpha2 = torch.sqrt(1-alpha_bars[T-1])
#     dif_img = (alpha1*img) + (alpha2*noise)
#     # for t in range(T,-1,-1):
#     #     xt = torch.sqrt(alphas[t])*xt + ((1-alphas[t])/torch.sqrt(1-alpha_bars[t]))*noise
#     return dif_img

# def reverse_process(imgT,T,noise_pred):
#     xt = torch.clone(imgT)
#     # xt= torch.tensor(re_range(ndimage.gaussian_filter(torch.randn(img.size()), 3)))
#     # xt= torch.tensor(noise_pred)
#     for t in range(T-1,-1,-1):
#         z = torch.from_numpy(ndimage.gaussian_filter(torch.randn(imgT.size()),3)).cuda()
#         sigma= ((1-alpha_bars[t-1])/(1-alpha_bars[t]))*betas[t]
#         sigma = torch.sqrt(sigma)
#         sigma=0
#         pred_mean = ((1-alphas[t])/torch.sqrt(1-alpha_bars[t])) * noise_pred
#         xt = (1/torch.sqrt(alphas[t])) * (xt - pred_mean) + sigma*z
    
#     return xt

# def get_betas(num_diffusion_timesteps):
#     scale = 1000 / num_diffusion_timesteps
#     # scale=1
#     beta_start = scale * 1e-6
#     beta_end = scale * 2e-4
#     betas= np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
#     # betas=[]
#     # for t in range(num_diffusion_timesteps):
#     #     betas.append((1/(num_diffusion_timesteps-t+2))*1e-3)
#     return torch.from_numpy(np.array(betas))

# def get_alphas(betas):
#     return 1-betas

# def get_alpha_bars(alphas):
#     return torch.cumprod(alphas,0)

# total_time=50
# betas = get_betas(total_time).cuda()
# alphas = get_alphas(betas).cuda()
# alpha_bars = get_alpha_bars(alphas).cuda()

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

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, scale=1):
    # scale = 1000 / 500
    beta_start = 0.000001*scale# 0.0000_6
    beta_end = 0.0002*scale# 0.012
    # beta_start = 0.0001*scale
    # beta_end = 0.02*scale
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
# timesteps = 50
# def get_timesteps(timesteps):
#     return timesteps
# timesteps = 50
# betas = linear_beta_schedule(timesteps=timesteps)

# # define alphas 
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# # calculations for diffusion q(x_t | x_{t-1}) and others
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# # calculations for posterior q(x_{t-1} | x_t, x_0)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def q_sample(x_start, t, noise=None, timesteps=50, beta_scale=1.0):
    
    # timesteps = 50

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps, scale=beta_scale)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t.cuda() * x_start + sqrt_one_minus_alphas_cumprod_t.cuda() * noise


@torch.no_grad()
def p_sample(noise, x, t, t_index, 
             betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t.cuda() * (
        x - betas_t.cuda() * noise.cuda() / sqrt_one_minus_alphas_cumprod_t.cuda()
    )

    if t_index == 0:
        return model_mean.cuda()
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t).cuda() * noise.cuda()

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(img, noise, timesteps=50, beta_scale=1.0):
    # timesteps = 50
    betas = linear_beta_schedule(timesteps=timesteps, scale=beta_scale)
    # betas = cosine_beta_schedule(timesteps=timesteps)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    # device = 'cuda'
    device = 'cuda:0'
    # timesteps=50
    b = img.size()[0]
    imgs = []

    # for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    #     img = p_sample(noise, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        # imgs.append(img)
    for i in reversed(range(0, timesteps)):
        img = p_sample(
            noise, img, torch.full((b,), i, device=device, dtype=torch.long), i,
            betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
    return img


