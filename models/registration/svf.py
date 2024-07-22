import torch
import torch.nn as nn
import torch.nn.functional as F
class Svf(nn.Module):
    def __init__(self, inshape=(128,128), steps=7):
        super().__init__()
        self.nsteps = steps # # of integral steps
        assert self.nsteps >= 0, 'nsteps should be >= 0, found: %d' % self.nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
    

    # def integrate(self, pos_flow):
    # def Svf_shooting(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
    def forward(self, pos_flow):  
        # pos_flow: velocity with shape [b, 2, 128, 128, 128]
        dims = len(pos_flow.shape)-2
        if dims == 2:
            b,c,w,h = pos_flow.shape
            if c != 2 and c != 3:
                pos_flow = pos_flow.permute(0,3,1,2)
        elif dims == 3:
            b,c,w,h,d = pos_flow.shape
            if c != 3:
                pos_flow = pos_flow.permute(0,4,1,2,3)

        vec = pos_flow
        dispList = []
        
        vec = vec * self.scale
        # dispList.append(vec)

        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            # dispList.append(vec)
        
        return vec, dispList   #len

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow   #self.grid:  identity
        new_locs_unnormalize = self.grid + flow
        shape = flow.shape[2:]
        #  new_locs  :  torch.Size([1, 3, 64, 64, 64])
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 1) #[1, 64, 64, 64,3]
            new_locs_unnormalize = new_locs_unnormalize[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 4, 1)
            new_locs_unnormalize = new_locs_unnormalize[..., [2, 1, 0]]

        warped = F.grid_sample(src, new_locs, mode=self.mode)
        # print(new_locs.shape)   #[b, 64, 64, 64, 3]
        # print(warped.shape)     #[6, 3, 64, 64, 64]
        # return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return (warped, new_locs_unnormalize)
    
# class Grad:
#     """
#     N-D gradient loss.
#     """

#     def __init__(self, penalty='l1', loss_mult=None):
#         self.penalty = penalty
#         self.loss_mult = loss_mult

#     def loss(self, _, y_pred):
#         if(len(y_pred.shape) == 5):
#             dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
#             dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
#             dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

#             if self.penalty == 'l2':
#                 dy = dy * dy
#                 dx = dx * dx
#                 dz = dz * dz

#             d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
#             grad = d / 3.0

#             if self.loss_mult is not None:
#                 grad *= self.loss_mult
#             return grad
#         elif(len(y_pred.shape) == 4):
#             dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
#             dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
#             if self.penalty == 'l2':
#                 dy = dy * dy
#                 dx = dx * dx
#             d = torch.mean(dx) + torch.mean(dy)
#             grad = d / 2.0

#             if self.loss_mult is not None:
#                 grad *= self.loss_mult
#             return grad
        
#     def __call__(self, _, y_pred):
#         return self.loss(_, y_pred)