from typing import Any
from torch.utils.data import Dataset as TorchDataset
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import pathlib
from modules.data.datareader.DENSE_cine_IO_utils import align_n_frames_to

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

class VideoVolDatasetWithPreprocessing(TorchDataset):
    """
    Forward the whole video sequence without any patchifying. 
    """
    def __init__(self, data, augmentation=None, config=None, full_config=None, dataset_name=None):
        super().__init__()
        self.data = data
        # self.transform = ToTensorV2()
        self.config = config
        self.full_config = full_config
        self.n_subjects = len(set([datum['subject_id'] for datum in self.data]))
        self.n_slices = len(set([datum['slice_full_id'] for datum in self.data]))
        # self.slice_full_ids = self.get_slice_full_ids()
        self.img_key = config.get('myo_mask_key', 'myo_masks')
        
        self.return_DENSE_disp = config.get('return_DENSE_disp', True)
        self.DENSE_disp_key = config.get('DENSE_disp_key', 'DENSE_displacement_field')

        # Random frame gap to enhance the data diversity
        self.random_frame_gap = int(config.get('random_frame_gap', 0))

        self.frame_aligning_method = config.get('frame_aligning_method', 'constant') # could be 'constant' or 'patchify'
        self.frame_aligning_n_frames = config.get('frame_aligning_n_frames', 10)

        self.forward_src_tar_pair = config.get('forward_src_tar_pair', False)
        
        # self.raw_n = len(self.data)
        
        # self.n_frames_to_use = config.get('n_frames_to_use', 10)
        self.original_n_frames = [d[self.img_key].shape[-1] for d in self.data]
        if self.frame_aligning_method == 'constant':
            # self.original_n_frames = [d[self.img_key].shape[-1] for d in self.data]
            self.align_volume_frames(target_key=self.img_key)    
            self.align_volume_frames(target_key=self.DENSE_disp_key)
        elif self.frame_aligning_method == 'none':
            pass
        else:
            raise ValueError(f'frame_aligning_method={self.frame_aligning_method} not recognized')
        
        
        self.disp_type = config.get('disp_type', 'Lagrangian')
        print(f'Using {self.disp_type} displacement field')
        self.disp_mask_key = config.get('disp_mask_key', 'myo_masks')
    
        
    def __len__(self):
        # return self.N
        return len(self.data)    
    
    def align_volume_frames(self, target_key=None):
        if target_key is None:
            target_key = self.img_key
        n_target_frames = self.frame_aligning_n_frames
        frame_idx = -1
        padding_method = 'constant'
        # print(self.data[0].keys())
        print(f'Aligning displacement field frames to {n_target_frames} frames')
        
        print(f'# of frames before alignment: {list(set([d[target_key].shape[-1] for d in self.data]))}')
        for datum_idx, datum in enumerate(self.data):
            self.data[datum_idx][target_key] = align_n_frames_to(datum[target_key], n_target_frames, frame_idx, padding_method)
        print(f'# of frames after alignment: {list(set([d[target_key].shape[-1] for d in self.data]))}')

    def filter_volume_with_blank_images(self):
        """
        Remove slices from the dataset that contain blank images.
        
        This method iterates over each slice in the dataset and checks if any of the frames in the slice are blank.
        If a blank image is found in a slice, the entire slice is removed from the dataset.
        """
        print("Filtering blank images...")
        print(f"Number of slices before filtering: {len(self.data)}")
        slice_to_pop_indices = []
        # if any of the frames in the slice is blank, remove the whole slice
        for slice_idx, slice_datum in enumerate(self.data):
            for frame_idx in range(slice_datum[self.img_key].shape[-1]):
                img = slice_datum[self.img_key][..., frame_idx]
                if img.sum() == 0:
                    slice_to_pop_indices.append(slice_idx)
                    break
        for slice_idx in sorted(slice_to_pop_indices, reverse=True):
            self.data.pop(slice_idx)
        print("Done filtering blank images.")
        print(f"Number of slices after filtering: {len(self.data)}")        
    
    def __getitem__(self, index):
        if self.forward_src_tar_pair:
            return self.get_src_tar_pair(index)
        else:
            return self.get_single_volume(index)
    
    def get_src_tar_pair(self, index):
        vol = self.data[index][self.img_key]         # [H, W, Nfr]
        disp = self.data[index][self.DENSE_disp_key] # [2, H, W, Nfr]
        Nfr = vol.shape[-1]

        vol = np.moveaxis(vol, -1, 0)[None, ...]     # [1, Nfr, H, W]
        src = np.repeat(vol[:, 0:1], Nfr-1, axis=1)  # [1, Nfr-1, H, W]
        tar = vol[:, 1:]                             # [1, Nfr-1, H, W]
        disp = np.moveaxis(disp, -1, 1)              # [2, Nfr, H, W]

        datum = {
            'src': src,
            'tar': tar,
            'DENSE_disp': disp[:, :-1],
            'ori_n_frames': self.original_n_frames[index]-1,
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
    
    def get_single_volume(self, index):
        img = self.data[index][self.img_key]         # [H, W, Nfr]
        disp = self.data[index][self.DENSE_disp_key] # [2, H, W, Nfr]

        img = np.moveaxis(img, -1, 0)[None, ...]     # [1, Nfr, H, W]
        disp = np.moveaxis(disp, -1, 1)              # [2, Nfr, H, W]

        datum = {
            'vol': img,
            'disp': disp,
            'ori_n_frames': self.original_n_frames[index],
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
