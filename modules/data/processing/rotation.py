import torch
import torch.nn.functional as F
def get_rot_mat(theta, device):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]], device=device)


def rot_img(x, theta, dtype, device):
    rot_mat = get_rot_mat(theta, device)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    # print(rot_mat)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False).type(dtype).to(device)
    x = F.grid_sample(x, grid, align_corners=False).to(device)
    return x

def rot_img_seq(x, theta, dtype, device):
    B, C, T, H, W = x.shape
    x_2D = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # rotated_labels_pre_2d = rot_img(x_2D, -theta, dtype, device)
    rotated_labels_pre_2d = rot_img(x_2D, theta, dtype, device)
    rotated_labels = rotated_labels_pre_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return rotated_labels

def rotate_displacement_field(x, theta, dtype, device):
    # rotated_labels_pre = torch.zeros(x.size())
    # rotated_labels = torch.zeros(x.size())
    rotated_labels_pre = rot_img(x, theta, dtype, device)
    rotated_labels = torch.zeros(x.size())
    u_labels = rotated_labels_pre[:, 0]
    v_labels = rotated_labels_pre[:, 1]
    rotated_labels[:, 0, :, :] = u_labels * torch.cos(torch.tensor(theta)) + v_labels * torch.sin(torch.tensor(theta))
    rotated_labels[:, 1, :, :] = -u_labels * torch.sin(torch.tensor(theta)) + v_labels * torch.cos(torch.tensor(theta))
    return rotated_labels

# def rotate_displacement_field(x, theta, dtype, device):
#     # rotated_labels_pre = torch.zeros(x.size())
#     # rotated_labels = torch.zeros(x.size())
#     rotated_labels = rot_img(x, theta, dtype, device)
#     # rotated_labels = torch.zeros(x.size())
#     u_labels = rotated_labels[:, 0]
#     v_labels = rotated_labels[:, 1]
#     rotated_labels[:, 0, :, :] = u_labels * torch.cos(torch.tensor(theta)) + v_labels * torch.sin(torch.tensor(theta))
#     rotated_labels[:, 1, :, :] = -u_labels * torch.sin(torch.tensor(theta)) + v_labels * torch.cos(torch.tensor(theta))
#     return rotated_labels

# def rot_vol(x, theta, dtype, device):
#     # dtype = disp.dtype
#     # return None

#     # Create rotation matrix
#     rot_mat = get_rot_mat_2dt(theta, device)[None, ...].type(dtype)
#     # print(rot_mat.shape)
    
#     # Repeat rotation matrix for batch processing
#     rot_mat_batch = rot_mat.repeat(x.size(0), 1, 1)
#     # rot_mat_batch = rot_mat[None, ...].repeat(x.size(0), 1, 1)
#     # print(rot_mat_batch.shape)
#     # print('AAA')
#     print(rot_mat_batch)

#     # Create affine grid for the entire batch
#     grid = F.affine_grid(rot_mat_batch, x.size(), align_corners=False).type(dtype)

#     # Rotate the displacement field
#     rotated_x = F.grid_sample(x, grid, align_corners=False)

#     return rotated_x

# def rotate_displacement_field_seq(x, theta, dtype, device):
#     rotated_disp = rot_vol(x, theta, dtype, device)

#     # Rotate the displacement vectors according to the chosen angle
#     cos_theta = torch.cos(torch.tensor(theta))
#     sin_theta = torch.sin(torch.tensor(theta))
#     u_disp = rotated_disp[:, 0, :, :, :]
#     v_disp = rotated_disp[:, 1, :, :, :]
#     rotated_disp[:, 0, :, :, :] = u_disp * cos_theta + v_disp * sin_theta
#     rotated_disp[:, 1, :, :, :] = -u_disp * sin_theta + v_disp * cos_theta
#     return rotated_disp

# def rot_seq(x, theta, dtype, device):
#     # x should have shape [N, C, H, W, T]
#     B, C, H, W, T = x.shape
#     # dtype = x.dtype

#     # Create 2D rotation matrix
#     rot_mat = get_rot_mat(theta, device).type(dtype)
    
#     # Prepare the displacement field by merging the batch and frames dimensions
#     disp_reshape = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
    
#     # Create affine grid for each 2D displacement field
#     grid = F.affine_grid(rot_mat.repeat(B * T, 1, 1), disp_reshape.size(), align_corners=False)

#     # Rotate the displacement field
#     rotated_disp = F.grid_sample(disp_reshape, grid, align_corners=False)
    
#     return rotated_disp
    
    # Reshape to original 5D shape
    # rotated_disp_5d = rotated_disp.view(B, T, C, H, W).permute(0, 2, 3, 4, 1)

    # return rotated_disp_5d

# def rotate_displacement_field_seq(x, theta, dtype, device):
#     B, C, H, W, T = x.shape
#     x_2D = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
#     rotated_labels_pre_2d = rot_img(x_2D, theta, dtype, device)

#     # Rotate the displacement vectors according to the chosen angle
#     cos_theta = torch.cos(torch.tensor(theta))
#     sin_theta = torch.sin(torch.tensor(theta))
#     u_disp = rotated_labels_pre_2d[:, 0, :, :]
#     v_disp = rotated_labels_pre_2d[:, 1, :, :]

#     rotated_disp_2d = torch.zeros_like(rotated_labels_pre_2d)
#     rotated_disp_2d[:, 0, :, :] = u_disp * cos_theta + v_disp * sin_theta
#     rotated_disp_2d[:, 1, :, :] = -u_disp * sin_theta + v_disp * cos_theta
#     rotated_disp = rotated_disp_2d.view(B, T, C, H, W).permute(0, 2, 3, 4, 1)
#     return rotated_disp


def rotate_displacement_field_seq(x, theta, dtype, device):
    B, C, T, H, W = x.shape
    # print(B, C, T, H, W)
    x_2D = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # print(x_2D.shape)
    rotated_labels_pre_2d = rot_img(x_2D, theta, dtype, device)

    # Rotate the displacement vectors according to the chosen angle
    cos_theta = torch.cos(torch.tensor(theta))
    sin_theta = torch.sin(torch.tensor(theta))
    u_disp = rotated_labels_pre_2d[:, 0, :, :]
    v_disp = rotated_labels_pre_2d[:, 1, :, :]

    rotated_disp_2d = torch.zeros_like(rotated_labels_pre_2d)
    rotated_disp_2d[:, 0, :, :] = u_disp * cos_theta + v_disp * sin_theta
    rotated_disp_2d[:, 1, :, :] = -u_disp * sin_theta + v_disp * cos_theta
    rotated_disp = rotated_disp_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return rotated_disp


def rotate_displacement_field_seq_inplace(x, theta, dtype, device):
    # Rotate only the vector direction instead of both direction and location
    B, C, T, H, W = x.shape
    # print(B, C, T, H, W)
    x_2D = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # print(x_2D.shape)
    # rotated_labels_pre_2d = rot_img(x_2D, theta, dtype, device)

    # Rotate the displacement vectors according to the chosen angle
    cos_theta = torch.cos(torch.tensor(theta))
    sin_theta = torch.sin(torch.tensor(theta))
    u_disp = x_2D[:, 0, :, :]
    v_disp = x_2D[:, 1, :, :]

    rotated_disp_2d = torch.zeros_like(x_2D)
    rotated_disp_2d[:, 0, :, :] = u_disp * cos_theta + v_disp * sin_theta
    rotated_disp_2d[:, 1, :, :] = -u_disp * sin_theta + v_disp * cos_theta
    rotated_disp = rotated_disp_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return rotated_disp


