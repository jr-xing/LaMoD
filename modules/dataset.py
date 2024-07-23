from typing import Any
from torch.utils.data import Dataset as TorchDataset
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import pathlib
class VideoVolDataset(TorchDataset):
    """
    Forward the whole video sequence without any patchifying. 
    """
    def __init__(self, data, config=None, augmentation=None, dataset_name=None):
        super().__init__()
        self.data = data
        self.config = config            # Config about this dataset
        
        self.forward_src_tar_pair = config.get('forward_src_tar_pair', False)
        self.img_key = config.get('img_key', 'myo_masks')
        self.DENSE_disp_key = config.get('DENSE_disp_key', 'DENSE_disp')
                
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        if self.forward_src_tar_pair:
            return self.get_src_tar_pair(index)
        else:
            return self.get_single_volume(index)
    
    def get_src_tar_pair(self, index):
        vol = self.data[index][self.img_key]         # [1, Nfr, H, W]
        disp = self.data[index][self.DENSE_disp_key] # [2, Nfr, H, W]
        ori_n_frames = self.data[index]['ori_n_frames']
        Nfr = vol.shape[1]

        # vol = np.moveaxis(vol, -1, 0)[None, ...]     # [1, Nfr, H, W]
        src = np.repeat(vol[:, 0:1], Nfr-1, axis=1)  # [1, Nfr-1, H, W]
        tar = vol[:, 1:]                             # [1, Nfr-1, H, W]
        # disp = np.moveaxis(disp, -1, 1)              # [2, Nfr, H, W]
        # print(f'{vol[:, 0:1].shape=}')
        datum = {
            'src': src,
            'tar': tar,
            'DENSE_disp': disp[:, :-1],
            'ori_n_frames': ori_n_frames-1,
        }
        # print(f'{disp.shape=}, {img.shape=}')

        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in self.data[index].items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            elif not isinstance(v, (torch.Tensor, np.ndarray)):
                datum[k] = v
            elif isinstance(v, float):
                datum[k] = torch.Tensor([v]).to(torch.float32)
            elif isinstance(v, int):
                datum[k] = torch.Tensor([v]).to(torch.long)
        
        # print("datum['src'].shape: ", datum['src'].shape)

        return datum
    
    def get_single_volume(self, index):
        img = self.data[index][self.img_key]         # [1, Nfr, H, W, ]
        disp = self.data[index][self.DENSE_disp_key] # [2, Nfr, H, W]

        # img = np.moveaxis(img, -1, 0)[None, ...]     # [1, Nfr, H, W]
        # disp = np.moveaxis(disp, -1, 1)              # [2, Nfr, H, W]

        datum = {
            'vol': img,
            'disp': disp,
            'ori_n_frames': self.data[index]['ori_n_frames'],
        }
        # print(f'{disp.shape=}, {img.shape=}')

        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in self.data[index].items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            elif not isinstance(v, (torch.Tensor, np.ndarray)):
                datum[k] = v
            elif isinstance(v, float):
                datum[k] = torch.Tensor([v]).to(torch.float32)
            elif isinstance(v, int):
                datum[k] = torch.Tensor([v]).to(torch.long)
        return datum
