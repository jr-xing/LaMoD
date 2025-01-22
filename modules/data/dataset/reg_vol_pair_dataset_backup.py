from typing import Any
from torch.utils.data import Dataset as TorchDataset
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
import pathlib
from modules.data.datareader.DENSE_cine_IO_utils import align_n_frames_to
class RegVolPairDataset(TorchDataset):
    def __init__(self, data, augmentation=None, config=None, full_config=None, dataset_name=None):
        super().__init__()
        self.data = data
        # self.transform = ToTensorV2()
        self.config = config
        self.full_config = full_config
        self.n_subjects = len(set([datum['subject_id'] for datum in self.data]))
        self.n_slices = len(set([datum['slice_full_id'] for datum in self.data]))
        # self.slice_full_ids = self.get_slice_full_ids()
        self.img_key = config.get('img_key', 'myo_masks')
        
        self.return_DENSE_disp = config.get('return_DENSE_disp', True)
        self.DENSE_disp_key = config.get('DENSE_disp_key', 'DENSE_displacement_field')

        # Random frame gap to enhance the data diversity
        self.random_frame_gap = int(config.get('random_frame_gap', 0))

        self.frame_aligning_method = config.get('frame_aligning_method', 'constant') # could be 'constant' or 'patchify'
        self.frame_aligning_n_frames = config.get('frame_aligning_n_frames', 10)
        self.frame_patchify_n_frames = config.get('frame_patchify_n_frames', -1)
        self.patch_generation_strategy = config.get('patch_generation_strategy', 'random') # could be 'random' or 'sliding_window' or 'minimal_overlap
        self.patch_random_sample_times = config.get('patch_random_sample_times', 1)
        self.raw_n = len(self.data)
        
        # self.n_frames_to_use = config.get('n_frames_to_use', 10)
        if self.frame_aligning_method == 'constant':
            self.align_volume_frames(self.img_key)    
            if self.return_DENSE_disp:
                self.align_volume_frames(target_key=self.DENSE_disp_key)
        elif self.frame_aligning_method == 'patchify':
            # Theortically we should make sure every volume has at least frame_patchify_n_frames frames
            # But for now we just assume that all volumes have enough frames
            pass
            
            # here we create the list of source and target volume starting frame indices
            self.patch_pair_indices_Lagrangian = []
            self.patch_pair_incides_Eulerian = []
            for datum_idx, datum in enumerate(self.data):
                n_frames = datum[self.img_key].shape[-1]
                if self.patch_generation_strategy == 'minimal_overlap':
                    # if patch_generation_strategy is 'minimal_overlap', 
                    # for Lagrangian pairs, the starting frame index of the source volume is always 0, 
                    #   and the starting frame index of the target volume is 1, 1+ patch_size, 1+2*patch_size, ...
                    #   if the last patch is not full, we set the starting frame index of the target volume to n_frames - patch_size
                    # get the starting frame index of the target volume
                    target_starting_frame_indices = list(range(1, n_frames - self.frame_patchify_n_frames + 1, self.frame_patchify_n_frames))
                    if target_starting_frame_indices[-1] < n_frames - self.frame_patchify_n_frames:
                        target_starting_frame_indices.append(n_frames - self.frame_patchify_n_frames)
                    for target_starting_frame_idx in target_starting_frame_indices:
                        self.patch_pair_indices_Lagrangian.append((
                            datum_idx,                  # souce volume index
                            0,                          # souce volume starting frame index
                            datum_idx,                  # target volume index
                            target_starting_frame_idx,  # target volume starting frame index
                            ))
                    # for Eulerian pairs, it is equivalent to sliding window so we don't need to consider it here
                    pass
                elif self.patch_generation_strategy == 'random':
                    pass
                elif self.patch_generation_strategy == 'sliding_window':
                    raise NotImplementedError('patch_generation_strategy=sliding_window not implemented')
        else:
            raise ValueError(f'frame_aligning_method={self.frame_aligning_method} not recognized')
        
        
        self.disp_type = config.get('disp_type', 'Lagrangian')
        print(f'Using {self.disp_type} displacement field')
        self.disp_mask_key = config.get('disp_mask_key', 'myo_masks')
        

        self.return_src_tar_img = config.get('return_src_tar_img', False)
        self.src_tar_img_key = config.get('src_tar_img_key', 'src_tar_img')
        
    def __len__(self):
        # return self.N
        if self.frame_aligning_method == 'patchify' and self.patch_generation_strategy == 'minimal_overlap':
            if self.disp_type == 'Eulerian':
                raise NotImplementedError('frame_aligning_method=patchify and patch_generation_strategy=minimal_overlap not implemented for disp_type=Eulerian')
            elif self.disp_type == 'Lagrangian':
                return len(self.patch_pair_indices_Lagrangian)
        elif self.frame_aligning_method == 'patchify' and self.patch_generation_strategy == 'random':
            return len(self.data) * self.patch_random_sample_times
        else:
            return len(self.data)
    
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
    
    def __getitem_bk__(self, index):
        # datum = self.data[index].feed_to_network()
        raw_datum = self.data[index]
        datum = {}

        raw_src = raw_datum[self.img_key] # should have shape (H, W, T)
        H, W, T = raw_src.shape
        if self.disp_type == 'Eulerian':
            src = raw_src[..., :T-1]
            tar = raw_src[..., 1:]
            src_disp_mask = raw_datum[self.disp_mask_key][..., :T-1]
            tar_disp_mask = raw_datum[self.disp_mask_key][..., 1:]
            src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
        elif self.disp_type == 'Lagrangian':
            # src: repeat the very first frame for T-1 times
            src = np.repeat(raw_src[..., 0:1], T-1, axis=-1)
            tar = raw_src[..., 1:]
            src_disp_mask = np.repeat(raw_datum[self.disp_mask_key][..., 0:1], T-1, axis=-1)
            tar_disp_mask = raw_datum[self.disp_mask_key][..., 1:]
            src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
            # raise NotImplementedError(f'disp_type={self.disp_type} not implemented')
        else:
            raise ValueError(f'disp_type={self.disp_type} not recognized')
        
        if self.return_DENSE_disp:
            datum['DENSE_disp'] = raw_datum[self.DENSE_disp_key][..., 1:]
        # else:
            # datum['DENSE_disp'] = np.zeros([2, *self.image_size])
            # pass
        
        if self.frame_aligning_method == 'constant':
            # reshape from (H, W, T) to (1, T, H, W)
            src = np.moveaxis(src, -1, 0)[None, ...]
            tar = np.moveaxis(tar, -1, 0)[None, ...]
            src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]
            if self.return_DENSE_disp:
                datum['DENSE_disp'] = np.moveaxis(datum['DENSE_disp'], -1, 0)[None, ...]
        elif self.frame_aligning_method == 'patchify':
            # randomly generate the starting frame index
            starting_frame_idx = np.random.randint(0, T - self.frame_patchify_n_frames)
            # print(f'Using frame {starting_frame_idx} to {starting_frame_idx+self.frame_patchify_n_frames-1} for patchifying')
            src = src[..., starting_frame_idx:starting_frame_idx+self.frame_patchify_n_frames]
            tar = tar[..., starting_frame_idx:starting_frame_idx+self.frame_patchify_n_frames]
            src_tar_disp_union_mask = src_tar_disp_union_mask[..., starting_frame_idx:starting_frame_idx+self.frame_patchify_n_frames]
            if self.return_DENSE_disp:
                datum['DENSE_disp'] = datum['DENSE_disp'][..., starting_frame_idx:starting_frame_idx+self.frame_patchify_n_frames]

            # reshape from (H, W, T) to (1, T, H, W)
            src = np.moveaxis(src, -1, 0)[None, ...]
            tar = np.moveaxis(tar, -1, 0)[None, ...]
            src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]
            if self.return_DENSE_disp:
                datum['DENSE_disp'] = np.moveaxis(datum['DENSE_disp'], -1, 1)#[None, ...]
            datum['temporal_patch_starting_frame_idx'] = starting_frame_idx
            datum['temporal_patch_n_frames'] = self.frame_patchify_n_frames            
        else:
            raise ValueError(f'frame_aligning_method={self.frame_aligning_method} not recognized')

        datum['src'] = src
        datum['tar'] = tar
        datum['src_tar_disp_union_mask'] = src_tar_disp_union_mask
        
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
    
    def __getitem__(self, index) -> Any:
        if self.frame_aligning_method == 'patchify' and self.patch_generation_strategy == 'random':
            return self.__getitem__aligned_or_random_patchy(index%self.raw_n)
        elif self.frame_aligning_method == 'patchify' and self.patch_generation_strategy == 'minimal_overlap':
            return self.__getitem__patchify_minimal_overlap(index)
        elif self.frame_aligning_method == 'constant':
            return self.__getitem__aligned_or_random_patchy(index)

    def __getitem__patchify_minimal_overlap(self, index):

        datum = {}
        if self.disp_type == 'Eulerian':
            raise NotImplementedError('frame_aligning_method=patchify and patch_generation_strategy=minimal_overlap not implemented for disp_type=Eulerian')
            src_datum_idx, src_starting_frame_idx, tar_datum_idx, tar_starting_frame_idx = self.patch_pair_incides_Eulerian[index]
        else:
            src_datum_idx, src_starting_frame_idx, tar_datum_idx, tar_starting_frame_idx = self.patch_pair_indices_Lagrangian[index]
            # print(src_datum_idx, src_starting_frame_idx, tar_datum_idx, tar_starting_frame_idx)

            raw_src = self.data[src_datum_idx][self.img_key] # should have shape (H, W, T)
            raw_tar = self.data[tar_datum_idx][self.img_key] # should have shape (H, W, T)
            H, W, T = raw_src.shape
            
            src = np.repeat(raw_src[..., 0:1], self.frame_patchify_n_frames, axis=-1)
            tar = raw_tar[..., tar_starting_frame_idx:tar_starting_frame_idx+self.frame_patchify_n_frames]

            # src_disp_mask = np.repeat(self.data[src_datum_idx][self.disp_mask_key][..., 0:1], self.frame_patchify_n_frames, axis=-1)
            # src_disp_mask = raw_src > 1e-5
            # tar_disp_mask = self.data[tar_datum_idx][self.disp_mask_key][..., tar_starting_frame_idx:tar_starting_frame_idx+self.frame_patchify_n_frames]
            # src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
            # src_tar_disp_union_mask = src_disp_mask
            # src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]

        

        if self.return_DENSE_disp:
            datum['DENSE_disp'] = self.data[tar_datum_idx][self.DENSE_disp_key][..., tar_starting_frame_idx:tar_starting_frame_idx+self.frame_patchify_n_frames]

        src = np.moveaxis(src, -1, 0)[None, ...]
        tar = np.moveaxis(tar, -1, 0)[None, ...]
        # src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]
        if self.return_DENSE_disp:
            datum['DENSE_disp'] = np.moveaxis(datum['DENSE_disp'], -1, 1)
        
        datum['src'] = src
        datum['tar'] = tar
        # datum['src_tar_disp_union_mask'] = src_tar_disp_union_mask
        # datum['src_tar_disp_union_mask'] = np.abs(datum['DENSE_disp']) > 1e-5
        # print(datum['DENSE_disp'].shape)
        
        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in self.data[src_datum_idx].items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            elif k == 'DENSE_myo_mask_bbox':
                datum[k] = str(v)
            elif not isinstance(v, (torch.Tensor, np.ndarray)):
                datum[k] = v
            elif isinstance(v, float):
                datum[k] = torch.Tensor([v]).to(torch.float32)
            elif isinstance(v, int):
                datum[k] = torch.Tensor([v]).to(torch.long)
            elif k == 'DENSE_myo_contour':
                n_contour_frame, _ = v.shape
                epi_contours = [v[fidx, 0] for fidx in range(n_contour_frame)]
                endo_contours = [v[fidx, 1] for fidx in range(n_contour_frame)]
                datum['DENSE_epi_contours'] = epi_contours
                datum['DENSE_endo_contours'] = endo_contours
            elif k == 'DENSE_mag_images':
                datum[k] = v
            elif k == 'PositionA' or k == 'PositionB':
                datum[k] = v
            

        datum['src_starting_frame_idx'] = src_starting_frame_idx
        datum['src_total_n_frames'] = T
        datum['tar_starting_frame_idx'] = tar_starting_frame_idx
        datum['tar_total_n_frames'] = T
        datum['temporal_patch_starting_frame_idx'] = src_starting_frame_idx
        datum['temporal_patch_n_frames'] = self.frame_patchify_n_frames
        return datum

    def __getitem__aligned_or_random_patchy(self, index, patch_start_idx=None):
        # datum = self.data[index].feed_to_network()
        raw_datum = self.data[index]
        datum = {}

        raw_src = raw_datum[self.img_key] # should have shape (H, W, T)
        H, W, T = raw_src.shape
        if self.disp_type == 'Eulerian':
            src = raw_src[..., :T-1]
            tar = raw_src[..., 1:]
            src_disp_mask = raw_datum[self.disp_mask_key][..., :T-1]
            tar_disp_mask = raw_datum[self.disp_mask_key][..., 1:]
            src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
        elif self.disp_type == 'Lagrangian':
            src = np.repeat(raw_src[..., 0:1], T-1, axis=-1)
            tar = raw_src[..., 1:]
            src_disp_mask = np.repeat(raw_datum[self.disp_mask_key][..., 0:1], T-1, axis=-1)
            tar_disp_mask = raw_datum[self.disp_mask_key][..., 1:]
            src_tar_disp_union_mask = np.logical_or(src_disp_mask, tar_disp_mask)
            # raise NotImplementedError(f'disp_type={self.disp_type} not implemented')
        else:
            raise ValueError(f'disp_type={self.disp_type} not recognized')
        
        if self.return_DENSE_disp:
            datum['DENSE_disp'] = raw_datum[self.DENSE_disp_key][..., 1:]
        # else:
            # datum['DENSE_disp'] = np.zeros([2, *self.image_size])
            # pass
        
        if self.frame_aligning_method == 'constant':
            # reshape from (H, W, T) to (1, T, H, W)
            src = np.moveaxis(src, -1, 0)[None, ...]
            tar = np.moveaxis(tar, -1, 0)[None, ...]
            src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]
            if self.return_DENSE_disp:
                datum['DENSE_disp'] = np.moveaxis(datum['DENSE_disp'], -1, 0)[None, ...]
        elif self.frame_aligning_method == 'patchify':
            # randomly generate the starting frame index
            if patch_start_idx is not None:
                src_starting_frame_idx = patch_start_idx
                # tar_starting_frame_idx = patch_start_idx
            elif self.disp_type == 'Eulerian':
                src_starting_frame_idx = np.random.randint(0, T - self.frame_patchify_n_frames)
            elif self.disp_type == 'Lagrangian':
                src_starting_frame_idx = 0

            # random_frame_gap = min(self.random_frame_gap, T - self.frame_patchify_n_frames)
            # tar_starting_frame_idx = src_starting_frame_idx + np.random.randint(1, random_frame_gap)
            if self.random_frame_gap > 1 or self.disp_type == 'Lagrangian':
                # tar_starting_frame_idx = src_starting_frame_idx + np.random.randint(1, self.random_frame_gap)
                tar_starting_frame_idx = np.random.randint(src_starting_frame_idx, T - self.frame_patchify_n_frames)
            else:
                tar_starting_frame_idx = src_starting_frame_idx
            datum['src_starting_frame_idx'] = src_starting_frame_idx
            datum['tar_starting_frame_idx'] = tar_starting_frame_idx
            # tar_starting_frame_idx = src_starting_frame_idx
            # print(f'Using frame {starting_frame_idx} to {starting_frame_idx+self.frame_patchify_n_frames-1} for patchifying')
            src = src[..., src_starting_frame_idx:src_starting_frame_idx+self.frame_patchify_n_frames]
            tar = tar[..., tar_starting_frame_idx:tar_starting_frame_idx+self.frame_patchify_n_frames]
            src_tar_disp_union_mask = src_tar_disp_union_mask[..., src_starting_frame_idx:src_starting_frame_idx+self.frame_patchify_n_frames]
            if self.return_DENSE_disp:
                if self.random_frame_gap > 0.5 and self.disp_type == 'Eulerian':
                    raise ValueError('random_frame_gap >= 1 not supported for return_DENSE_disp when disp_type is Eulerian')
                else:
                    # datum['DENSE_disp'] = datum['DENSE_disp'][..., src_starting_frame_idx:src_starting_frame_idx+self.frame_patchify_n_frames]
                    datum['DENSE_disp'] = datum['DENSE_disp'][..., tar_starting_frame_idx:tar_starting_frame_idx+self.frame_patchify_n_frames]

            # reshape from (H, W, T) to (1, T, H, W)
            src = np.moveaxis(src, -1, 0)[None, ...]
            tar = np.moveaxis(tar, -1, 0)[None, ...]
            src_tar_disp_union_mask = np.moveaxis(src_tar_disp_union_mask, -1, 0)[None, ...]
            if self.return_DENSE_disp:
                datum['DENSE_disp'] = np.moveaxis(datum['DENSE_disp'], -1, 1)#[None, ...]
            datum['temporal_patch_starting_frame_idx'] = src_starting_frame_idx
            datum['temporal_patch_n_frames'] = self.frame_patchify_n_frames            
        else:
            raise ValueError(f'frame_aligning_method={self.frame_aligning_method} not recognized')

        datum['src'] = src
        datum['tar'] = tar
        datum['src_tar_disp_union_mask'] = src_tar_disp_union_mask
        
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
            elif k == 'DENSE_myo_contour':
                n_contour_frame, _ = v.shape
                epi_contours = [v[fidx, 0] for fidx in range(n_contour_frame)]
                endo_contours = [v[fidx, 1] for fidx in range(n_contour_frame)]
                datum['DENSE_epi_contours'] = epi_contours
                datum['DENSE_endo_contours'] = endo_contours
            elif k == 'DENSE_mag_images':
                datum[k] = v
        # datum_augmented = self.transform(**datum)
        # datum_augmented['idx'] = index
        return datum