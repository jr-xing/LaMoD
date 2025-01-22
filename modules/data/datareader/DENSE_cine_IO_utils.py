import numpy as np
def align_n_frames_to(volume, n_target_frames, frame_idx=-1, padding_method='edge'):
    """
    Modifies an input video volume to have a specified number of frames. If the input volume has more frames
    than `n_target_frames`, it simply keeps the first `n_target_frames` frames. If the volume has less frames, 
    it pads the volume at the end (of the frame dimension) to make up the difference using a specified padding 
    strategy.

    Parameters:
    - volume: A 3D numpy array of shape (H, W, n_frames) by default.
    - n_target_frames: An integer with the required frame number 
    - frame_idx: An integer with the index of frame dimension in `volume` (-1 means the last dimension)
    - padding_method: A string representing the padding method if volume frames < n_target_frames. Options include
      'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap'. 
      Defaults to 'edge' padding.

    Returns:
    - 3D numpy array with `n_target_frames` frames by either cropping or padding the input volume using the 
      specified padding strategy.

    Raises:
    - ValueError: If a non-supported `padding_method` is applied, numpy will raise a ValueError.
    """
    n_frames = volume.shape[frame_idx]
  
    # case when input volume has frames greater than or equal to n_target_frames
    if n_frames >= n_target_frames:
        # if frame_idx is -1, just slice from the end
        if frame_idx == -1:
            return volume[..., :n_target_frames]
        # if frame_idx is not -1, then need to slice along specific axis
        else:
            indices = [slice(None)] * volume.ndim
            indices[frame_idx] = slice(0, n_target_frames)
            return volume[tuple(indices)]
  
    # case when input volume's frames are less than n_target_frames
    else:
        # calculate the padding amount
        padding_amount = n_target_frames - n_frames

        # create the padding scheme which pads only at the end of dimension
        paddings = [(0, 0)] * volume.ndim
        paddings[frame_idx] = (0, padding_amount)

        return np.pad(volume, paddings, mode=padding_method)
    
import numpy as np
import copy
def append_additional_data_from_npy(ori_data, npy_filename=None, config={}, file_source='from_Nellie'):
    if file_source == 'from_Nellie':
        data_type = config.get('data_type', 'phi_displacement')
        if npy_filename is None:
            npy_filename = '/p/miauva/data/Jerry/medical-images/Cardiac-FromKen/task-specific/2023-09-25-DENSE-guided-cine-registration/deform_result_unchecked-20231018.npy'
        new_data = np.load(npy_filename, allow_pickle=True)
        updated_data = []
        for ori_slice in ori_data:
            patient_id = ori_slice['patient_id']
            cine_slice_idx = ori_slice['cine_slice_idx']
            cine_slice_location = ori_slice['cine_slice_location']

            # print(patient_id, cine_slice_idx, cine_slice_location)

            registration_slices_idx_and_data = [(i, s) for i, s in enumerate(new_data) if s['patient_id'] == patient_id and s['cine_slice_idx'] == cine_slice_idx and s['cine_slice_location'] - cine_slice_location < 1e-1]
            registration_slices = [s[1] for s in registration_slices_idx_and_data]
            # print([s[0] for s in registration_slices_idx_and_data])

            if len(registration_slices) == 0:
                print('not found')
                continue
            elif len(registration_slices) > 1:
                print('more than 1 found')
                continue
            else:
                registration_slice = registration_slices[0]

            merged_slice = copy.deepcopy(ori_slice)
            merged_slice['cine_lv_myo_masks_merged_displacement_field_X'] = registration_slice['phi_displacement'][0]
            merged_slice['cine_lv_myo_masks_merged_displacement_field_Y'] = registration_slice['phi_displacement'][1]
            # for key in registration_slice.keys():
            #     if key not in merged_slice.keys():
            #         if key == 'displacement':
            #             cine_lv_myo_masks_merged_displacement_X = registration_slice['displacement'][0]
            #             cine_lv_myo_masks_merged_displacement_Y = registration_slice['displacement'][1]
            #             merged_slice['cine_lv_myo_masks_merged_displacement_X'] = cine_lv_myo_masks_merged_displacement_X
            #             merged_slice['cine_lv_myo_masks_merged_displacement_Y'] = cine_lv_myo_masks_merged_displacement_Y
            #         else:
            #             # merged_slice[try_replace_key(key)] = registration_slice[key]
            #             merged_slice[key] = registration_slice[key]

            updated_data.append(merged_slice)
        return updated_data
    else:
        raise NotImplementedError('Only support data from Nellie for now.')
    

# def filter_slices()
import re
def get_site_name_from_datum(datum):
    sites_info = {
        "UVA": {
            "InstitutionName": ["UVA", "UVa", "University of Virginia"],
            "InstitutionAddress": [],
            "StudyDescription": ["UVa\^"],
        },
        "SaintEtienne": {
            "InstitutionName": ["IRMAS NORD3TR"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        "Kentucky": {
            "InstitutionName": ["University of Kentucky"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        "Glasgow": {
            "InstitutionName": ["Golden Jubilee National Hospital"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        "StFrancis": {
            "InstitutionName": ["ST FRANCIS HOSPITAL", "St. Francis Diagnostic"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        "RoyalBrompton": {
            "InstitutionName": ["Royal Brompton SKYRA"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        "Emory": {
            "InstitutionName": ["Emory University"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        },
        # "Standford": {
        "Stanford": {
            "InstitutionName": ["Palo Alto VAMC"],
            "InstitutionAddress": [],
            "StudyDescription": [],
        }
    }
    founded_site_name = None
    for site_name, site_info in sites_info.items():
        if len(datum['SequenceInfo'][0,0].InstitutionName) == 0:
            if any([re.findall(name, datum['SequenceInfo'][0,0].StudyDescription) for name in site_info['StudyDescription']]):
                founded_site_name = site_name
                found_site_of_datum = True
        elif any([re.findall(name, datum['SequenceInfo'][0,0].InstitutionName) for name in site_info['InstitutionName']]) or any([re.findall(name, datum['SequenceInfo'][0,0].StudyDescription) for name in site_info['StudyDescription']]):
            founded_site_name = site_name
            found_site_of_datum = True
        elif datum['SequenceInfo'][0,0].InstitutionName == 'F':
            founded_site_name = 'UNKNOWN'
            found_site_of_datum = True
    if found_site_of_datum == False:
        founded_site_name = 'UNKNOWN'
        # raise ValueError(f'{datum_idx}')
    return founded_site_name