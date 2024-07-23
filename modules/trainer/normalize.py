def latent_1_std(z):
    # first divided by max to make it in [...,1], then standardize it by mean and std
    # note this need to return the mean and std for reverse
    z_max = z.max()
    z_min = z.min()
    z = (z - z_min) / (z_max - z_min)
    z_mean = z.mean(dim=(1,2,3,4), keepdim=True)
    z_std = z.std(dim=(1,2,3,4), keepdim=True)
    z = (z - z_mean) / z_std
    return z, z_mean, z_std, z_min, z_max

def latent_1_std_reverse(z, z_mean, z_std, z_min, z_max):
    z = z * z_std + z_mean
    z = z * (z_max - z_min) + z_min
    return z

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

def latent_plus_mins_1_scaler(z):
    z_max = z.max()
    z_min = z.min()
    z = (z - z_min) / (z_max - z_min)
    z = z * 2 - 1
    return z, z_min, z_max

def latent_plus_mins_1_reverse(z, z_min, z_max):
    z = (z + 1) / 2
    z = z * (z_max - z_min) + z_min
    return z

def latent_standardize(z):
    # make mean to 0 and std to 1
    z_mean = z.mean()
    z_std = z.std()
    z = (z - z_mean) / z_std
    return z, z_mean, z_std

def latent_unstandardize(z, z_mean, z_std):
    return z * z_std + z_mean