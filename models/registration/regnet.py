import os
# from EpdiffLib import Epdiff
from models.registration.EpdiffLib import Epdiff
import lagomorph as lm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.registration.svf import Svf, SpatialTransformer#, Grad

from timeit import default_timer

################################################################
# fourier layer
################################################################

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims: int, in_channels: int, out_channels: int, kernel: int=3, stride: int=1, padding: int=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=(128, 128),
                 infeats=2,
                 nb_features=[[16, 32, 32], [32, 32, 32, 16, 16]],
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 out_channels=None,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        if out_channels is None:
            out_channels = ndims
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # if nb_features is a string (e.g. "[[16, 32, 32], [32, 32, 32, 16, 16]]"), convert to list
        if isinstance(nb_features, str) and nb_features.startswith('['):
            nb_features = eval(nb_features)

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            # feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = int(np.round(nb_features * feat_mult ** np.arange(nb_levels)))#.astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        for i in range(len(max_pool)):
            if isinstance(max_pool[i], list):
                max_pool[i] = tuple(max_pool[i])

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling: list[MaxPooling] = [MaxPooling(s) for s in max_pool]
        self.upsampling: list[nn.Upsample] = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        self.skip_connect = kwargs.get('skip_connect', True)
        print(f"Skip connect: {self.skip_connect}")
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(int(ndims), int(prev_nf), int(nf)))
                prev_nf = nf
            self.decoder.append(convs)
            if (not half_res or level < (self.nb_levels - 2)) and self.skip_connect:
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(int(ndims), int(prev_nf), int(nf)))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.final_nf, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        # encoder forward pass
        x_history = [x]   #torch.Size([3, 2, 100, 100])

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)  #100x100 -> 50x50 -> 25x25  ->12x12  ->6x6
                                        #128x128   64x64   32x32   16x16
# [20, 16, 64, 64, 64]   [20, 32, 32, 32, 32]  [20, 32, 16, 16, 16]  [20, 32, 8, 8, 8]
        encoder_output = x
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                if self.skip_connect:
                    x = torch.cat([x, x_history.pop()], dim=1)
                # else:
                #     print("No skip connect")
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        x = self.flow(x)  #[20, 16, 128, 128] -> [20, 2, 128, 128]
        # return x.permute(0,2,3,1)   #-> [20, 128, 128, 2]
        # return x.permute(0,2,3,4,1)   #-> [20, 128, 128, 128,3]
        
        return x, encoder_output
        
        # if(len(x.shape)==4):
        #     return x.permute(0,2,3,1), encoder_output   #-> [20, 128, 128, 2]
        # return x.permute(0,2,3,4,1), encoder_output   #-> [20, 128, 128, 128,3]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        # encoder forward pass
        x_history = [x]

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        return x, x_history
    
    def decode(self, x: torch.Tensor, x_history: list) -> torch.Tensor:
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                if self.skip_connect:
                    x = torch.cat([x, x_history.pop()], dim=1)
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        x = self.flow(x)
        # if(len(x.shape)==4):
        #     return x.permute(0,2,3,1)
        # return x.permute(0,2,3,4,1)
        return x

class RegNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # imagesize = args.imagesize
        # alpha = args.alpha
        # gamma = args.gamma
        self.imagesize = self.model_config.get('imagesize', [128,128])
        self.alpha = self.model_config.get('alpha', 2.0)
        self.gamma = self.model_config.get('gamma', 1.0)
        self.sigma = self.model_config.get('sigma', 0.03)
        self.infeats = self.model_config.get('infeats', 2)
        
        self.disp_generator = self.model_config.get('disp_generator', 'Epdiff')
        self.twoD_plus_T = self.model_config.get('twoD_plus_T', False)        
        if self.disp_generator.lower() == 'epdiff':
            self.MEpdiff = Epdiff(alpha=self.alpha, gamma=self.gamma)
            self.fluid_params = [self.alpha, 0, self.gamma]
            self.metric = lm.FluidMetric(self.fluid_params)
            self.unet_output = self.model_config.get('unet_output', 'momentum')
        elif self.disp_generator.lower() == 'svf':

            self.svf = Svf(
                steps = self.model_config.get('steps', 7),
                inshape = self.imagesize[1:] if len(self.imagesize)==3 and self.twoD_plus_T else self.imagesize,
            )
            self.spatial_transformer = SpatialTransformer(
                size = self.imagesize[1:] if len(self.imagesize)==3 and self.twoD_plus_T else self.imagesize,
                mode = self.model_config.get('spatial_transformer_mode', 'bilinear')
            )
            self.unet_output = self.model_config.get('unet_output', 'velocity')

        self.TSteps = self.model_config.get('TSteps', 10)
        self.twoDT_to_2D_method = self.model_config.get('twoDT_to_2D_method', 'slicing')
        
        if len(self.imagesize) == 3 and self.twoD_plus_T:
            out_channels = 2
        else:
            out_channels = None
        self.register_unet = Unet(
            inshape=self.imagesize,
            infeats=self.infeats,
            nb_features=model_config.get('nb_features', [[16, 32, 32], [32, 32, 32, 16, 16]]),
            nb_levels=model_config.get('nb_levels', None),
            max_pool=model_config.get('max_pool', 2),
            feat_mult=model_config.get('feat_mult', 1),
            nb_conv_per_level=model_config.get('nb_conv_per_level', 1),
            half_res=model_config.get('half_res', False),
            skip_connect=model_config.get('skip_connect', True),
            out_channels=out_channels
            )
        # self.criterion = nn.MSELoss()

        # self.enable_disp_conv_layers = self.model_config.get('enable_disp_conv_layers', False)
        # print(f'enable_disp_conv_layers: {self.enable_disp_conv_layers}')
        # self.disp_conv_layers = nn.ModuleList()
        # if self.enable_disp_conv_layers:            
        #     for i in range(2):
        #         self.disp_conv_layers.append(nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1))
        # else:
        #     # append identity layers
        #     for i in range(2):
        #         self.disp_conv_layers.append(nn.Identity())

    
    def forward_2D_input_2D_network(self, src: torch.Tensor, tar: torch.Tensor, query_step=-1) -> torch.Tensor:
        """Both src and tar are 2D images, and the network is also 2D.
        """
        input = torch.cat((src,tar), dim=1)  #[20, 2, 32, 32]
        output, encoder_output = self.register_unet(input)        #[20, 32, 32, 2]
        if self.disp_generator.lower() == 'epdiff':
            m = output               # momemtum
            v = self.metric.sharp(m) # velocity
            u_seq, v_seq, m_seq, ui_seq = \
                self.MEpdiff.my_expmap( m, num_steps=self.TSteps)
            u = u_seq[-1]
            ui = ui_seq[-1]
            Sdef = lm.interp(src, u)
            return {
                'velocity': v,
                'momentum': m,
                # 'displacement_T_S': u,
                # 'displacement_S_T': ui,
                'displacement': ui,
                'deformed_source': Sdef
            }
        elif self.disp_generator.lower() == 'svf':
            v = output
            u = self.svf(v)[0]
            Sdef = self.spatial_transformer(src, u)[0]
            return {
                'velocity': v,
                'momentum': None,
                'displacement': u,
                'deformed_source': Sdef
            }
    
    def forward_2DT_input_2D_network(self, src: torch.Tensor, tar: torch.Tensor, query_step=-1) -> torch.Tensor:
        """Both src and tar are 2D+T image sequences, but the network is 2D.
            In this case, we need to reshape the input and output.
        """
        batch_size, _, T, H, W = src.shape
        src = src.view(batch_size*T, 1, H, W)
        tar = tar.view(batch_size*T, 1, H, W)
        input = torch.cat((src,tar), dim=1)  #[20, 2, 32, 32]
        output, encoder_output = self.register_unet(input)        #[20, 32, 32, 2]
        if self.disp_generator.lower() == 'epdiff':
            m = output               # momemtum
            v = self.metric.sharp(m) # velocity
            u_seq, v_seq, m_seq, ui_seq = \
                self.MEpdiff.my_expmap( m, num_steps=self.TSteps)
            u = u_seq[-1]
            ui = ui_seq[-1]
            Sdef = lm.interp(src, u)

            m = m.view(batch_size, 2, T, H, W)
            v = v.view(batch_size, 2, T, H, W)
            u = u.view(batch_size, 2, T, H, W)
            ui = ui.view(batch_size, 2, T, H, W)
            Sdef = Sdef.view(batch_size, 1, T, H, W)

            return {
                'velocity': v,
                'momentum': m,
                # 'displacement_T_S': u,
                # 'displacement_S_T': ui,
                'displacement': ui,
                'deformed_source': Sdef
            }
        elif self.disp_generator.lower() == 'svf':
            v = output
            v = v.view(batch_size*T, 2, H, W)
            u = self.svf(v)[0]
            # ui = self.svf(-v)[0]
            Sdef = self.spatial_transformer(
                src.view(batch_size*T, 1, H, W),
                u)[0]
            return {
                'velocity': v.view(batch_size, 2, T, H, W),
                'momentum': None,
                'displacement': u.view(batch_size, 2, T, H, W),
                'deformed_source': Sdef.view(batch_size, 1, T, H, W)
            }
    
    def forward_2DT_input_3D_network(self, src: torch.Tensor, tar: torch.Tensor, query_step=-1) -> torch.Tensor:
        """Both src and tar are 2D+T image sequences, and the network is 3D.
            In this case, we need to reshape the input and output only in the wrapping step.
        """
        
        # src: (N, C, T, H, W) -> (N, C, T, W, H)
        # src = src.permute(0, 1, 2, 4, 3)
        # tar = tar.permute(0, 1, 2, 4, 3)
        input = torch.cat((src,tar), dim=1)  #[20, 2, 32, 32]
        output, encoder_output = self.register_unet(input) 
        batch_size, _, T, H, W = input.shape
        if self.disp_generator.lower() == 'epdiff':
            if self.unet_output == 'momentum':
                m = output
                # 2D+T to 2D
                m_2D = torch.concat([m[:,:,t] for t in range(T)], dim=0)
                src_2D = torch.concat([src[:,:,t] for t in range(T)], dim=0)
                
            elif self.unet_output == 'velocity':
                v = output
                # 2D+T to 2D
                v_2D = torch.concat([v[:,:,t] for t in range(T)], dim=0)
                src_2D = torch.concat([src[:,:,t] for t in range(T)], dim=0)
                
            # Compute mementum / velocity
            if self.unet_output == 'momentum':
                m_2D_YX = m_2D.permute(0,1,3,2)
                v_2D_YX = self.metric.sharp(m_2D_YX)
                v_2D = v_2D_YX.permute(0,1,3,2)
            elif self.unet_output == 'velocity':
                v_2D_YX = v_2D.permute(0,1,3,2)
                m_2D_YX = self.metric.flat(v_2D_YX)
                m_2D = m_2D_YX.permute(0,1,3,2)
                
            src_2D_YX = src_2D.permute(0,1,3,2)
            u_2D_YX_seq, v_2D_YX_seq, m_2D_YX_seq, ui_YX_2D_seq = \
                self.MEpdiff.my_expmap( m_2D_YX, num_steps=self.TSteps)
            u_2D_YX = u_2D_YX_seq[-1]
            Sdef_2D_YX = lm.interp(src_2D_YX, u_2D_YX)

            Sdef_2D = Sdef_2D_YX.permute(0,1,3,2)            
            u_2D = u_2D_YX_seq[-1].permute(0,1,3,2)
            ui_2D = ui_YX_2D_seq[-1].permute(0,1,3,2)
        
            # convert back to 2D+T
            m = torch.stack(torch.split(m_2D, batch_size, dim=0), dim=2)
            v = torch.stack(torch.split(v_2D, batch_size, dim=0), dim=2)
            u = torch.stack(torch.split(u_2D, batch_size, dim=0), dim=2)
            ui = torch.stack(torch.split(ui_2D, batch_size, dim=0), dim=2)
            Sdef = torch.stack(torch.split(Sdef_2D, batch_size, dim=0), dim=2)
            
            # m = torch.stack([m_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            # v = torch.stack([v_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            # u = torch.stack([u_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            # ui = torch.stack([ui_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            # Sdef = torch.stack([Sdef_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)

            # permute x and y back
            # Sdef: (N, C, T, W, H) -> (N, C, T, H, W)
            # Sdef = Sdef.permute(0, 1, 2, 4, 3)

            return_dict = {
                'velocity': v,
                'momentum': m,
                'displacement_T_S': u,
                # 'displacement_S_T': ui,
                'displacement': ui,
                'deformed_source': Sdef,
                'latent': encoder_output
            }
            if query_step != -1:
                return_dict['query_step'] = query_step
                query_displacement_2D = ui_YX_2D_seq[query_step].permute(0,1,3,2)
                query_displacement = torch.stack(torch.split(query_displacement_2D, batch_size, dim=0), dim=2)                
                query_deformed_source_2D = lm.interp(src_2D_YX, u_2D_YX)#u_2D_YX_seq[-1])
                query_deformed_source = torch.stack(torch.split(query_deformed_source_2D, batch_size, dim=0), dim=2)
                return_dict['query_displacement'] = query_displacement
                return_dict['query_deformed_source'] = query_deformed_source                
            return return_dict
        
        elif self.disp_generator.lower() == 'svf':
            v = output # with shape (N, 2, T, H, W)
            batch_size, _, T, H, W = v.shape
            # v_2D = v.view(batch_size*T, 2, H, W)
            # src_2D = src.view(batch_size*T, 1, H, W)
            v_2D = torch.concat([v[:,:,t] for t in range(T)], dim=0)
            src_2D = torch.concat([src[:,:,t] for t in range(T)], dim=0)
            
            u_2D = self.svf(v_2D)[0]
            Sdef_2D = self.spatial_transformer(
                src_2D,
                u_2D)[0]
            ui_2D = self.svf(-v_2D)[0]
            # v = v.view(batch_size, 2, T, H, W),
            # u =  u.view(batch_size, 2, T, H, W),
            # Sdef = Sdef.view(batch_size, 1, T, H, W)
            # convert back to 3D
            # m = torch.stack([m[t] for t in range(T)], dim=2)
            v = torch.stack([v_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            u = torch.stack([u_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            ui = torch.stack([ui_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            Sdef = torch.stack([Sdef_2D[t*batch_size:(t+1)*batch_size] for t in range(T)], dim=2)
            
            return {
                'velocity': v,
                'momentum': None,
                'displacement': ui,
                'deformed_source': Sdef,
                'latent': encoder_output
            }
    
    
    def forward_3D_input_3D_network(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'Not implemented yet for 3D input and 3D network')

    def forward(self, src: torch.Tensor, tar: torch.Tensor, query_step=-1) -> torch.Tensor:
        if len(self.imagesize) == 2 and not self.twoD_plus_T:
            # both src and tar are 2D images, and the network is also 2D
            return self.forward_2D_input_2D_network(src, tar, query_step=query_step)
        elif len(self.imagesize) == 2 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, but the network is 2D
            return self.forward_2DT_input_2D_network(src, tar, query_step=query_step)
        elif len(self.imagesize) == 3 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, and the network is 3D
            return self.forward_2DT_input_3D_network(src, tar, query_step=query_step)
        elif len(self.imagesize) == 3 and not self.twoD_plus_T:
            # both src and tar are 3D volumes, and the network is 3D
            return self.forward_3D_input_3D_network(src, tar, query_step=query_step)
        else:
            raise ValueError(f'Unrecognized input shape: {src.shape}, {tar.shape}')

    def inference(self, src: torch.Tensor, tar: torch.Tensor, frame_interval=8) -> torch.Tensor:
        """Make inference on the input tensors.
        The input tensors can be either 2D images or 2D+T image sequences.
        """
        if len(self.imagesize) == 2 and not self.twoD_plus_T:
            # both src and tar are 2D images, and the network is also 2D
            return self.forward_2D_input_2D_network(src, tar)
        elif len(self.imagesize) == 2 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, but the network is 2D
            return self.forward_2DT_input_2D_network(src, tar)
        elif len(self.imagesize) == 3 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, and the network is 3D
            for patch_idx in range(0, src.shape[2], frame_interval):
                src_patch = src[:,:,patch_idx:patch_idx+frame_interval,:,:]
                tar_patch = tar[:,:,patch_idx:patch_idx+frame_interval,:,:]
                pred_dict = self.forward_2DT_input_3D_network(src_patch, tar_patch)
                if patch_idx == 0:
                    return_dict = pred_dict
                else:
                    for key in pred_dict.keys():
                        return_dict[key] = torch.cat([return_dict[key], pred_dict[key]], dim=2)
            return return_dict
        elif len(self.imagesize) == 3 and not self.twoD_plus_T:
            # both src and tar are 3D volumes, and the network is 3D
            raise NotImplementedError(f'Not implemented yet for 3D input and 3D network')
            # return self.forward_3D_input_3D_network(src, tar)


    
    
    def __repr__(self):
        return f'RegNet({self.model_config})'

    def encode(self, src: torch.Tensor, tar: torch.Tensor,
               src_img: torch.Tensor or None=None, tar_img: torch.Tensor or None=None) -> torch.Tensor:
        if len(self.imagesize) == 2 and not self.twoD_plus_T:
            # both src and tar are 2D images, and the network is also 2D
            input = torch.cat((src,tar), dim=1)  #[20, 2, 32, 32]
            encoder_output, encoder_latent_features = self.register_unet.encode(input)
            return {
                'latent': encoder_output,
                'latent_history': encoder_latent_features
            }
        elif len(self.imagesize) == 2 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, but the network is 2D
            batch_size, _, T, H, W = src.shape
            src = src.view(batch_size*T, 1, H, W)
            tar = tar.view(batch_size*T, 1, H, W)
            input = torch.cat((src,tar), dim=1)
            encoder_output, encoder_latent_features = self.register_unet.encode(input)
            encoder_output = encoder_output.view(batch_size, T, H, W)
            return  {
                'latent': encoder_output,
                'latent_history': encoder_latent_features
            }
        elif len(self.imagesize) == 3 and self.twoD_plus_T:
            # both src and tar are 2D+T image sequences, and the network is 3D
            input = torch.cat((src,tar), dim=1)  #[20, 2, T, 32, 32]
            encoder_output, encoder_latent_features = self.register_unet.encode(input)
            return {
                'latent': encoder_output,
                'latent_history': encoder_latent_features
            }
        elif len(self.imagesize) == 3 and not self.twoD_plus_T:
            # both src and tar are 3D volumes, and the network is 3D
            # return self.forward_3D_input_3D_network(src, tar)
            raise NotImplementedError(f'Not implemented yet for 3D input and 3D network')
        else:
            raise ValueError(f'Unrecognized input shape: {src.shape}, {tar.shape}')

    
