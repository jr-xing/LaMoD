from typing import Any
from torch.utils.data import Dataset as TorchDataset
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import pathlib
from modules.data.datareader.DENSE_cine_IO_utils import align_n_frames_to

class RegPairDataset(TorchDataset):
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
        self.return_DENSE_disp = config.get('return_DENSE_disp', True)
        self.DENSE_disp_key = config.get('DENSE_disp_key', 'DENSE_displacement_field')
        self.return_src_tar_img = config.get('return_src_tar_img', False)
        self.src_tar_img_key = config.get('src_tar_img_key', 'src_tar_img')
        self.disp_type = config.get('disp_type', 'Lagrangian')
        self.disp_mask_key = config.get('disp_mask_key', 'myo_masks')
        print(f'Using {self.disp_type} displacement field')

        # Random frame gap to enhance the data diversity
        self.random_frame_gap = int(config.get('random_frame_gap', 0))

        self.image_size = config.get('image_size', [128, 128])
        if type(self.image_size) == int:
            self.image_size = [self.image_size, self.image_size]
        elif type(self.image_size) == str:
            self.image_size = eval(self.image_size)

        self.additional_data_infos = config.get('additional_data_infos', {})
        
        # Build a global index for each frame of the slices
        # which maps a global index to the (slice_idx and frame_idx)
        # self.global_frame_idx = []
        # for slice_idx, slice_datum in self.data:
        #     for frame_idx in range(slice_datum[self.myo_mask_key].shape[-1]):
        #         self.global_frame_idx.append((slice_idx, frame_idx))

        # Build pairwise indices
        # Lagrangian: from first frame to later frames (slice_idx, 0, slice_idx, frame_idx)
        self.pairwise_indices_Lagrangian = []
        for slice_idx, slice_datum in enumerate(self.data):
            for frame_idx in range(1, slice_datum[self.myo_mask_key].shape[-1]):
                self.pairwise_indices_Lagrangian.append(
                    (
                        slice_idx, 0,            # source frame slice idx and frame idx
                        slice_idx, frame_idx,    # target frame slice idx and frame idx
                        slice_datum[self.myo_mask_key].shape[-1], slice_datum[self.myo_mask_key].shape[-1]) # source and target frame count
                    )
        # Eulerian: from previous frame to next frame (slice_idx, frame_idx-1, slice_idx, frame_idx)
        self.pairwise_indices_Eulerian = []
        for slice_idx, slice_datum in enumerate(self.data):
            for frame_idx in range(1, slice_datum[self.myo_mask_key].shape[-1]):
                self.pairwise_indices_Eulerian.append(
                        (
                            slice_idx, frame_idx-1, # source frame slice idx and frame idx
                            slice_idx, frame_idx,   # target frame slice idx and frame idx
                            slice_datum[self.myo_mask_key].shape[-1], slice_datum[self.myo_mask_key].shape[-1] # source and target frame count
                        )
                    )
        assert len(self.pairwise_indices_Lagrangian) == len(self.pairwise_indices_Eulerian), f'len(self.pairwise_indices_Lagrangian)={len(self.pairwise_indices_Lagrangian)} != len(self.pairwise_indices_Eulerian)={len(self.pairwise_indices_Eulerian)}'
        self.N = len(self.pairwise_indices_Lagrangian)    
    
    def __len__(self):
        # return self.N
        return self.N
    
    # def align_myo_mask_volume_frames(self):
    #     n_target_frames = self.n_myo_frames_to_use
    #     frame_idx = -1
    #     padding_method = 'constant'
    #     # print(self.data[0].keys())
    #     print(f'Aligning displacement field frames to {n_target_frames} frames')
        
    #     print(f'# of frames before alignment: {list(set([d[self.myo_mask_key].shape[-1] for d in self.data]))}')
    #     for datum_idx, datum in enumerate(self.data):
    #         self.data[datum_idx][self.myo_mask_key] = align_n_frames_to(datum[self.myo_mask_key], n_target_frames, frame_idx, padding_method)
    #     print(f'# of frames after alignment: {list(set([d[self.myo_mask_key].shape[-1] for d in self.data]))}')
    @property
    def pairwise_indices(self):
        if self.disp_type == 'Lagrangian':
            return self.pairwise_indices_Lagrangian
        elif self.disp_type == 'Eulerian':
            return self.pairwise_indices_Eulerian
        else:
            raise NotImplementedError(f'disp_type={self.disp_type} not implemented')
    
    def __getitem__(self, index):
        # datum = self.data[index].feed_to_network()
        src_slice_idx, src_frame_idx, tar_slice_idx, tar_frame_idx, src_slice_n_frames, tar_slice_n_frames = self.pairwise_indices[index]

        # modify the target frame indices to add random_frame_gap
        # generate the gap in [0, random_frame_gap]
        if self.disp_type == 'Eulerian':
            random_frame_gap = np.random.randint(0, self.random_frame_gap+1)
        else:
            random_frame_gap = 0
        tar_frame_idx = min(tar_frame_idx + random_frame_gap, tar_slice_n_frames-1)

        src_slice_id = self.data[src_slice_idx]['slice_full_id']
        tar_slice_id = self.data[tar_slice_idx]['slice_full_id']
        raw_src_datum = self.data[src_slice_idx]
        src = self.data[src_slice_idx][self.myo_mask_key][...,src_frame_idx][None,...]
        tar = self.data[tar_slice_idx][self.myo_mask_key][...,tar_frame_idx][None,...]

        src_disp_mask = self.data[src_slice_idx][self.disp_mask_key][...,src_frame_idx][None,...]
        tar_disp_mask = self.data[tar_slice_idx][self.disp_mask_key][...,tar_frame_idx][None,...]
        src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
        datum = {
            'src': src,
            'tar': tar,
            "src_tar_disp_union_mask": src_tar_disp_union_mask,
            'src_slice_idx': src_slice_idx,
            'src_frame_idx': src_frame_idx,
            'src_slice_id': src_slice_id,
            'tar_slice_idx': tar_slice_idx,
            'tar_frame_idx': tar_frame_idx,
            'tar_slice_id': tar_slice_id,
        }
        if self.return_DENSE_disp:
            datum['DENSE_disp'] = self.data[src_slice_idx][self.DENSE_disp_key][...,tar_frame_idx] if self.DENSE_disp_key in self.data[src_slice_idx].keys() else np.zeros([2, *self.image_size])
        else:
            datum['DENSE_disp'] = np.zeros([2, *self.image_size])
        
        if self.return_src_tar_img:
            datum['src_img'] = self.data[src_slice_idx][self.src_tar_img_key][...,src_frame_idx]
            datum['tar_img'] = self.data[tar_slice_idx][self.src_tar_img_key][...,tar_frame_idx]
        else:
            datum['src_img'] = np.zeros(self.image_size)
            datum['tar_img'] = np.zeros(self.image_size)

        for additional_data_name, additional_data_info in self.additional_data_infos.items():
            additional_data_key = additional_data_info['key']
            additional_data_output_key = additional_data_info.get('output_key', additional_data_key)
            datum[additional_data_output_key] = self.data[src_slice_idx][additional_data_key][...,src_frame_idx][None,...]
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
        # datum = {
        #     'volume': raw_datum['myo_masks']
        # }

        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in raw_src_datum.items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            # elif not isinstance(v, (torch.Tensor, np.ndarray, list)):
            #     datum[k] = v
            elif isinstance(v, str):
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
    