# https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2))
                            #    padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        # return self.block(x)
        for layer in self.block:
            # print(x.shape)
            x = layer(x)
        # print(x.shape)
        return x
    
# Map from (N, T+1, D1) to (N, T, D2)
class LatentReshape(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.linear = nn.Linear(d1, d2)

    def forward(self, x):
        # x is of shape (N, T+1, D)
        # We want to transform it to be of shape (N, T, D1)
        # We can do this by flattening, applying a linear transformation, and then reshaping
        N, T_plus_1, D = x.shape
        x = x.view(N, -1)  # Flatten to (N, (T+1)*D1)
        x = self.linear(x)  # Apply linear transformation
        x = x.view(N, T_plus_1-1, -1)  # Reshape to (N, T, D2)
        return x
# def determine_latent_H(ori_dim, image_size):
#     """
#     Determine the latent size based on the original dimension and image size.

#     The latent size is determined such that:
#     1. latent_size**2 is greater than or equal to ori_dim.
#     2. image_size is divisible by latent_size.

#     Parameters:
#     ori_dim (int): The original dimension.
#     image_size (int): The size of the image.

#     Returns:
#     int: The determined latent size.
#     """
#     latent_size = int(ori_dim**0.5)
#     if latent_size**2 < ori_dim:
#         latent_size += 1

#     while image_size % latent_size != 0:
#         latent_size += 1

#     return latent_size

def determine_latent_H(ori_dim, image_size):
    """
    Determine the latent size based on the original dimension and image size.

    The latent size is determined such that:
    1. latent_size**2 is greater than or equal to ori_dim.
    2. There exists a non-negative integer k such that latent_size * (2**k) equals image_size.

    Parameters:
    ori_dim (int): The original dimension.
    image_size (int): The size of the image.

    Returns:
    int: The determined latent size.
    """
    latent_size = int(ori_dim**0.5)
    if latent_size**2 < ori_dim:
        latent_size += 1

    while image_size % latent_size != 0 or (image_size / latent_size) & ((image_size / latent_size) - 1) != 0:
        latent_size += 1

    return latent_size

# latent decoder: decode latent feature with shape (N, T+1, D) 
# into (N, C, T, H, W) 
# where C is the number of channels, 
# H and W are the height and width of the image
# it should contains linear layers, reshaping layer, convTranspose3d layers and activation layers
class LatentDecoder(nn.Module):
    def __init__(self, 
                 dim = 192,        # D: dimension of the latent feature
                 num_frames = 8,   # T: number of frames
                 out_channels = 2, # C: number of channels in the output image
                 image_size = 128, # H, W: size of the image
                 n_total_conv_blocks = 3
    ):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.out_channels = out_channels
        
        # init_reform: map the input from (N, T+1, D) to (N, T, H'**2), 
        # where H' is the smallest integer such that H'**2 >= D and mode(image_size, H') == 0
        # self.latent_reshape_size = determine_latent_H(dim, image_size)
        if image_size == 48:
            self.latent_reshape_size = 24
        elif image_size == 128:
            self.latent_reshape_size = 32
        self.init_latent_reshape = LatentReshape(
            d1=(num_frames+1)*dim, 
            d2=num_frames*(self.latent_reshape_size**2))
        # latent_H = int(np.sqrt(image_size))

        # deconv3d blocks
        # note that those deconv3d blocks do not affect the time dimension (i.e. T)
        # note that after the initial reshape, there will be another reshape layer 
        # and the shape of the deconv inputs is (N, 1, T, sqrt(dim), sqrt(dim))
        # thus the # of deconv3d blocks is determined by the image_size and the latent_H
        # kernel_size = (1,3,3)
        kernel_size = 3
        n_upsampling_needed = int(np.log2(image_size // self.latent_reshape_size))
        if n_upsampling_needed > n_total_conv_blocks:
            raise ValueError(f"n_upsampling_needed: {n_upsampling_needed} > n_total_conv_blocks: {n_total_conv_blocks}")

        self.init_conv3D_block = Conv3DBlock(1, out_channels, kernel_size=kernel_size)
        self.conv_blocks = nn.ModuleList()
        for _ in range(n_upsampling_needed):
            self.conv_blocks.append(Deconv3DBlock(out_channels, out_channels, kernel_size=kernel_size))
        for _ in range(n_total_conv_blocks - n_upsampling_needed):
            self.conv_blocks.append(Conv3DBlock(out_channels, out_channels, kernel_size=kernel_size))

        # n_deconv3d_blocks = int(np.log2(image_size // latent_H))
        # self.init_deconv3d_block = Deconv3DBlock(1, out_channels, kernel_size=kernel_size)
        # # self.deconv3d_blocks = nn.ModuleList([Deconv3DBlock(1, out_channels, kernel_size=3)])
        # self.deconv3d_blocks = nn.ModuleList()
        # for _ in range(n_deconv3d_blocks-1):
        #     self.deconv3d_blocks.append(Deconv3DBlock(out_channels, out_channels, kernel_size=kernel_size))

    def forward(self, x):
        # x is of shape (N, T+1, D)
        # print(x.shape)
        x = self.init_latent_reshape(x) # x is of shape (N, T, D)
        # print(x.shape)
        
        x = x.view(
            x.shape[0], 1, 
            x.shape[1], 
            int(self.latent_reshape_size), 
            int(self.latent_reshape_size))     # x is of shape (N, 1, T, sqrt(D), sqrt(D))
        
        x = self.init_conv3D_block(x) # x is of shape (N, 2, T, sqrt(D), sqrt(D))
        # print(x.shape)
        
        for block in self.conv_blocks:
            x = block(x)
        
        return x
    
        




class ViViTReg(nn.Module):
    def __init__(self, 
                 image_size, patch_size, num_classes, num_frames, 
                 dim = 192, depth = 4, heads = 3, pool = 'cls', 
                 in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, n_decoder_total_conv_blocks=3,**kwargs):
        super().__init__()
        
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # warning: unrecognized arguments in kwargs
        print(f'kwargs: {kwargs}')

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        
        self.latent_decoder = \
            LatentDecoder(dim=dim, 
                          num_frames=num_frames, 
                          out_channels=in_channels, 
                          image_size=image_size,
                          n_total_conv_blocks=n_decoder_total_conv_blocks)

    def forward(self, x):
        # transform the input x from shape (N, C, T, H, W) into (N, T, C, H, W) to meet the requirement of the model
        x = x.permute(0, 2, 1, 3, 4)

        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = self.latent_decoder(x)
        
        return x
        

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # return self.mlp_head(x)
    