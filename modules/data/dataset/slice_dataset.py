from typing import Any
from torch.utils.data import Dataset as TorchDataset
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import pathlib
from modules.data.datareader.DENSE_cine_IO_utils import align_n_frames_to

class SliceDataset(TorchDataset):
    def __init__(self, data, augmentation=None, config=None, full_config=None, dataset_name=None):
        super().__init__()
        self.data = data
        # self.transform = ToTensorV2()
        self.config = config
        self.full_config = full_config
        self.n_subjects = len(set([datum['subject_id'] for datum in self.data]))
        self.n_slices = len(set([datum['slice_full_id'] for datum in self.data]))
        self.slice_full_ids = self.get_slice_full_ids()
        self.myo_mask_key = config.get('myo_mask_key', 'myo_masks')
        self.n_myo_frames_to_use = config.get('n_myo_frames_to_use', 10)
        self.align_myo_mask_volume_frames()
        
    
    def __len__(self):
        # return self.N
        return len(self.data)
    
    def align_myo_mask_volume_frames(self):
        n_target_frames = self.n_myo_frames_to_use
        frame_idx = -1
        padding_method = 'constant'
        # print(self.data[0].keys())
        print(f'Aligning displacement field frames to {n_target_frames} frames')
        
        print(f'# of frames before alignment: {list(set([d[self.myo_mask_key].shape[-1] for d in self.data]))}')
        for datum_idx, datum in enumerate(self.data):
            self.data[datum_idx][self.myo_mask_key] = align_n_frames_to(datum[self.myo_mask_key], n_target_frames, frame_idx, padding_method)
        print(f'# of frames after alignment: {list(set([d[self.myo_mask_key].shape[-1] for d in self.data]))}')
    
    def __getitem__(self, index):
        # datum = self.data[index].feed_to_network()
        raw_datum = self.data[index]
        # datum = {
        #     'source_img': raw_datum['source_image'],
        #     'target_img': raw_datum['target_image'],            
        # }
        # datum['source_img'] = torch.from_numpy(datum['source_img'][None, :,:]).to(torch.float32)
        # datum['target_img'] = torch.from_numpy(datum['target_img'][None, :,:]).to(torch.float32)

        # if self.config.get('feed_masks', False):
        #     datum['source_mask'] = torch.from_numpy(raw_datum['source_mask'][None, :,:]).to(torch.float32)
        #     datum['target_mask'] = torch.from_numpy(raw_datum['target_mask'][None, :,:]).to(torch.float32)

        # datum['displacement_field_X'] = torch.from_numpy(raw_datum['DENSE_displacement_field_X'][None, :,:]).to(torch.float32)
        # datum['displacement_field_Y'] = torch.from_numpy(raw_datum['DENSE_displacement_field_Y'][None, :,:]).to(torch.float32)

        # datum['TOS'] = torch.from_numpy(raw_datum['TOS']).to(torch.float32)
        # datum['slice_LMA_label'] = torch.Tensor(raw_datum['slice_LMA_label']).to(torch.long)
        # datum['sector_LMA_labels'] = torch.Tensor(raw_datum['sector_LMA_labels']).to(torch.long)
        # datum['strain_mat'] = torch.from_numpy(raw_datum['strain_matrix'][None, :, :]).to(torch.float32)
        datum = {
            'volume': raw_datum['myo_masks']
        }

        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in raw_datum.items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            elif not isinstance(v, (torch.Tensor, np.ndarray)):
                datum[k] = v
            elif isinstance(v, float):
                datum[k] = torch.Tensor([v]).to(torch.float32)
            elif isinstance(v, int):
                datum[k] = torch.Tensor([v]).to(torch.long)
        # datum_augmented = self.transform(**datum)
        # datum_augmented['idx'] = index
        return datum
    

    def get_subject_ids(self):
        return list(set([datum['subject_id'] for datum in self.data]))
    
    def get_slice_full_ids(self):
        return list(set([datum['slice_full_id'] for datum in self.data]))
    
    def get_n_slices(self):
        return len(self.slice_full_ids)
    
    def get_slice(self, slice_idx):
        # data_of_slices = [datum for datum in self.data if datum['slice_full_id'] == self.slice_full_ids[slice_idx]]
        data_indices_of_slices = [idx for idx, datum in enumerate(self.data) if datum['slice_full_id'] == self.slice_full_ids[slice_idx]]
        data_of_slices = [self.__getitem__(idx) for idx in data_indices_of_slices]
        return data_of_slices
    