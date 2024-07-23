import torch
import torch.nn.functional as F
def get_rot_mat(theta, device):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]], device=device)

class ImageRotator:
    """
    Rotates a 2D pytorch tensor with shape (B, C, H, W) by a specified angle.
    """
    def __init__(self, theta, device, dtype, data_shape):
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.data_shape = data_shape
        self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(data_shape[0],1,1)
        self.grid = F.affine_grid(self.rot_mat, data_shape, align_corners=False).type(dtype).to(device)

    def __call__(self, x):
        return F.grid_sample(x, self.grid, align_corners=False)#.to(self.device)

    def __repr__(self):
        return f"ImageRotator(angle={self.theta})"
    

class ImageSeqRotator:
    """
    Rotates a 3D pytorch tensor with shape (B, 1, T, H, W) by a specified angle.
    """
    def __init__(self, theta, device, dtype, data_shape):
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.data_shape = data_shape
        self.data_shape_2D = (data_shape[0]*data_shape[2], data_shape[1], data_shape[3], data_shape[4])
        self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(self.data_shape_2D[0],1,1)
        self.grid = F.affine_grid(self.rot_mat, self.data_shape_2D, align_corners=False).type(dtype).to(device)

    def __call__(self, x):
        if x.shape != self.data_shape:
            self.data_shape = x.shape
            self.data_shape_2D = (self.data_shape[0]*self.data_shape[2], self.data_shape[1], self.data_shape[3], self.data_shape[4])
            self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(self.data_shape_2D[0],1,1)
            self.grid = F.affine_grid(self.rot_mat, self.data_shape_2D, align_corners=False).type(x.dtype).to(x.device)
        B, C, T, H, W = x.shape
        x_2D = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rotated_labels_pre_2d = F.grid_sample(x_2D, self.grid, align_corners=False)
        rotated_labels = rotated_labels_pre_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return rotated_labels

    def __repr__(self):
        return f"ImageSeqRotator(angle={self.theta})"
    

class DisplacementFieldRotator:
    """
    Rotates a 2D pytorch tensor with shape (B, 2, H, W) by a specified angle.
    """
    def __init__(self, theta, device, dtype, data_shape):
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.data_shape = data_shape
        self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(data_shape[0],1,1)
        self.grid = F.affine_grid(self.rot_mat, data_shape, align_corners=False).type(dtype).to(device)

    def __call__(self, x):
        rotated_labels_pre = F.grid_sample(x, self.grid, align_corners=False)
        rotated_labels = torch.zeros(x.size())
        u_labels = rotated_labels_pre[:, 0]
        v_labels = rotated_labels_pre[:, 1]
        rotated_labels[:, 0, :, :] = u_labels * torch.cos(torch.tensor(self.theta)) + v_labels * torch.sin(torch.tensor(self.theta))
        rotated_labels[:, 1, :, :] = -u_labels * torch.sin(torch.tensor(self.theta)) + v_labels * torch.cos(torch.tensor(self.theta))
        return rotated_labels

    def __repr__(self):
        return f"DisplacementFieldRotator(angle={self.theta})"
    

class DisplacementFieldSeqRotator:
    """
    Rotates a 3D pytorch tensor with shape (B, 2, T, H, W) by a specified angle.
    """
    def __init__(self, theta, device, dtype, data_shape):
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.data_shape = data_shape
        self.data_shape_2D = (data_shape[0]*data_shape[2], data_shape[1], data_shape[3], data_shape[4])
        self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(self.data_shape_2D[0],1,1)
        self.grid = F.affine_grid(self.rot_mat, self.data_shape_2D, align_corners=False).type(dtype).to(device)

        self.cos_theta = torch.cos(torch.tensor(self.theta))
        self.sin_theta = torch.sin(torch.tensor(self.theta))

    def __call__(self, x):
        if x.shape != self.data_shape:
            self.data_shape = x.shape
            self.data_shape_2D = (self.data_shape[0]*self.data_shape[2], self.data_shape[1], self.data_shape[3], self.data_shape[4])
            self.rot_mat = get_rot_mat(self.theta, self.device)[None, ...].type(self.dtype).repeat(self.data_shape_2D[0],1,1)
            self.grid = F.affine_grid(self.rot_mat, self.data_shape_2D, align_corners=False).type(x.dtype).to(x.device)
        B, C, T, H, W = x.shape
        x_2D = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rotated_labels_pre_2d = F.grid_sample(x_2D, self.grid, align_corners=False)

        # Rotate the displacement vectors according to the chosen angle
        u_disp = rotated_labels_pre_2d[:, 0]
        v_disp = rotated_labels_pre_2d[:, 1]

        rotated_disp_2d = torch.zeros_like(rotated_labels_pre_2d)
        # rotated_labels = torch.zeros(x.size())
        rotated_disp_2d[:, 0, :, :] = u_disp * self.cos_theta + v_disp * self.sin_theta
        rotated_disp_2d[:, 1, :, :] = -u_disp * self.sin_theta + v_disp * self.cos_theta
        rotated_disp = rotated_disp_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return rotated_disp


    def __repr__(self):
        return f"DisplacementFieldSeqRotator(angle={self.theta})"