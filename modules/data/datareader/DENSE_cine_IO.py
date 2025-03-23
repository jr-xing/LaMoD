import numpy as np
from pathlib import Path
import copy
import re
from modules.data.datareader.BaseDatum import BaseDatum
from modules.data.datareader.BaseDataReader import BaseDataReader
from modules.data.augmentation.affine import rotate, translate
from skimage.morphology import dilation
# from .DENSE_cine_IO_utils import get_site_name_from_datum
class DENSECineDatum(BaseDatum):
    def load_datum(self):
        pass
    
    @staticmethod
    def load_data(data_list):
        pass

import re
from pathlib import Path
def glob_star(filename_pattern):
    """
    Returns a list of filenames that match the given pattern.
    Only wildcard character '*' is supported in the pattern.

    Parameters:
    filename_pattern (str): The pattern to match filenames against. The pattern can contain a wildcard character '*'.

    Returns:
    list: A sorted list of filenames that match the pattern.

    Examples:
    >>> glob_star('/p/a*.txt')
    ['/p/a1.txt', '/p/a2.txt']
    """
    # convert the filename_pattern to a regex
    regex = re.compile(filename_pattern.replace('*', '.*'))
    # get the directory of the filename_pattern
    filename_dir = Path(filename_pattern).parent
    # get all the files under the directory
    filenames = [str(filename) for filename in list(filename_dir.glob('*'))]
    # filter the filenames using the regex
    filenames_matched = sorted([filename for filename in filenames if regex.match(filename)])
    return filenames_matched

class DENSECINEDataReader(BaseDataReader):
    # def load_record_from_dir(self, config):
    #     return super().load_record_from_dir(config)

    def load_record_from_npy(self, data_config, data_reader_name=None, raw_data=None):
        # collect filenames
        if 'npy_filename' in data_config['loading'].keys() and 'npy_filenames' not in data_config['loading'].keys():
            npy_filename = data_config['loading']['npy_filename']
            npy_filenames = [npy_filename]
        elif 'npy_filename' not in data_config['loading'].keys() and 'npy_filenames' in data_config['loading'].keys():
            raw_npy_filenames = data_config['loading']['npy_filenames']
            if isinstance(raw_npy_filenames, str):
                raw_npy_filenames = [raw_npy_filenames]

            # Since the filename may contain regular expression, we need to find the corresponding files
            npy_filenames = []
            for raw_npy_filename in raw_npy_filenames:
                if '*' in raw_npy_filename:
                    npy_filenames += glob_star(raw_npy_filename)
                else:
                    npy_filenames.append(raw_npy_filename)
            print(f'Found {len(npy_filenames)} npy files: \n{npy_filenames}')
        else:
            raise NotImplementedError('Either npy_filename or npy_filenames should be provided in data_config[\'loading\']')
        if raw_data is None or len(raw_data) == 0:
            raw_data = []
            loading_method = data_config['loading'].get('method', 'general_slices')
            for npy_filename in npy_filenames:                      
                # raw_pairs = load_pair_list_from_npy_file(npy_filename, data_config)
                # if loading_method == 'cine_registration_pairs':
                #     raw_data = load_cine_pairs_from_npy_file(npy_filename, data_config)
                # elif loading_method == 'DENSE_slices':
                #     raw_data = load_DENSE_slices_from_npy_file(npy_filename, data_config)
                # elif loading_method == 'general_slice':
                #     raw_data = load_slices_from_npy_file(npy_filename, data_config)
                # else:
                #     raise NotImplementedError(f'loading_method {loading_method} not implemented')

                data_reader_name_str = data_reader_name if data_reader_name is not None else ''
                
                raw_data += load_slices_from_npy_file(npy_filename, data_config)
        
        all_data = []
        for raw_datum_idx, raw_datum in enumerate(raw_data):
            datum_dict = raw_datum
            
            if 'patient_id' in raw_datum.keys():
                datum_dict['subject_id'] = raw_datum['patient_id'] + f'-{data_reader_name_str}'
            if 'subject_id' in raw_datum.keys():
                datum_dict['subject_id'] += f'-{data_reader_name_str}'
            # datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_DENSE_time_idx"]}_{raw_datum["target_DENSE_time_idx"]}'
            # if loading_method == 'cine_registration_pairs':
            #     # datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["slice_idx"]}_{raw_datum["source_time_idx"]}_{raw_datum["target_time_idx"]}'
            #     datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_time_idx"]}_{raw_datum["target_time_idx"]}'
            if loading_method in ['DENSE_slices', 'general_slices']:
                datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["slice_idx"]}'
            # {
            #     'source_img': raw_datum['source_cine_mask'],
            #     'target_img': raw_datum['target_cine_mask'],
            #     'subject_id': raw_datum['patient_id'],
            # }
            # datum_dict['site'] = get_site_name_from_datum(raw_datum)
            datum = DENSECineDatum(data_dict=datum_dict)
            all_data.append(datum)

        resize = data_config['loading'].get('resize', False)
        if resize:
            print('resizing DENSE-cine images...', end='')
            from skimage.transform import resize
            for datum in all_data:
                datum['image'] = resize(datum['image'], [128, 128])
            print('DONE!')

        return all_data, raw_data

def augment_datum(datum: dict, config=None):
    if config is None:
        config = {
            'translate': {
                'y': 0,
                'x': 0,
            },
            'rotate': {
                'n_rotate_sectors': 0,
            },
        }
    translate_y = config['translate']['y']
    translate_x = config['translate']['x']
    n_rotate_sectors = config['rotate']['n_rotate_sectors']    
    
    datum_aug = rotate(datum, n_rotate_sectors)
    datum_aug = translate(datum_aug, translate_y, translate_x)
    datum_aug['augmented'] = True
    
    return datum_aug

def augment_all_data(data_list, data_config):
    augment_translate_times_y = data_config['loading'].get('augment_translate_times_y', 0)
    augment_translate_times_x = data_config['loading'].get('augment_translate_times_x', 0)
    augment_rotate_times = data_config['loading'].get('augment_rotate_times', 0)
    if augment_translate_times_y == 0:
        augment_translate_ys = [0]
    elif augment_translate_times_y == 1:
        augment_translate_ys = [5]
    else:
        if augment_translate_times_y % 2 == 0:
            augment_translate_ys_positive = np.linspace(0, 10, augment_translate_times_y//2+2).astype(int)[1:-1]
            augment_translate_ys_negative = -augment_translate_ys_positive
        else:
            augment_translate_ys_positive = np.linspace(0, 10, np.ceil(augment_translate_times_y/2)+2).astype(int)[1:-1]
            augment_translate_ys_negative = -augment_translate_ys_positive[:-1]
        augment_translate_ys = np.concatenate([augment_translate_ys_positive, augment_translate_ys_negative])
    
    if augment_translate_times_x == 0:
        augment_translate_xs = [0]
    elif augment_translate_times_x == 1:
        augment_translate_xs = [5]
    else:
        if augment_translate_times_x % 2 == 0:
            augment_translate_xs_positive = np.linspace(0, 10, augment_translate_times_x//2+2).astype(int)[1:-1]
            augment_translate_xs_negative = -augment_translate_xs_positive
        else:
            augment_translate_xs_positive = np.linspace(0, 10, np.ceil(augment_translate_times_x/2).astype(int)+2).astype(int)[1:-1]
            augment_translate_xs_negative = -augment_translate_xs_positive[:-1]
        augment_translate_xs = np.concatenate([augment_translate_xs_positive, augment_translate_xs_negative])
    augment_rotate_interval = data_config['loading'].get('augment_rotate_interval', 10)
    if augment_rotate_interval == -1:
        augment_rotate_n_sectors = np.linspace(1, 126, augment_rotate_times+2).astype(int)[1:-1]
    else:
        augment_rotate_n_sectors = (np.arange(1,20)*augment_rotate_interval)[:augment_rotate_times]

    augmented_data_list = []
    default_augment_config = {
        'translate': {
            'y': 0,
            'x': 0,
        },
        'rotate': {
            'n_rotate_sectors': 0,
        },
    }
    print(f'Augmenting data: translate_ys={augment_translate_ys}, translate_xs={augment_translate_xs}, rotate_n_sectors={augment_rotate_n_sectors}')
    # data_keys_to_check = ['StrainInfo', 'TOSAnalysis']
    data_keys_to_check = []
    for datum in data_list:
        # check whether the data is valid
        key_missing = False
        for key in data_keys_to_check:
            if key not in datum.keys():
                print(f'Warning: key {key} not found in datum of patient {datum["patient_id"]}')
                key_missing = True
                continue
        if key_missing:
            continue


        for augment_translate_y in augment_translate_ys:
            for augment_translate_x in augment_translate_xs:
                for augment_rotate_n_sector in augment_rotate_n_sectors:
                    augment_config = default_augment_config
                    augment_config['translate']['y'] = augment_translate_y
                    augment_config['translate']['x'] = augment_translate_x
                    augment_config['rotate']['n_rotate_sectors'] = augment_rotate_n_sector
                    augment_config['rotate']['augment_rotate_angle'] = - augment_rotate_n_sector * 360 / 126
                    augmented_datum = copy.deepcopy(datum)
                    augmented_datum.update(augment_datum(augmented_datum, augment_config))
                    augmented_datum['augmented'] = True
                    # augmented_datum['augment_translate_y'] = augment_translate_y
                    # augmented_datum['augment_translate_x'] = augment_translate_x
                    # augmented_datum['augment_rotate_n_sector'] = augment_rotate_n_sector
                    # augmented_datum['augment_rotate_angle'] = - augment_rotate_n_sector * 360 / 126
                    augmented_data_list.append(augmented_datum)
    
    print(f'Augmented data from {len(data_list)} to {len(augmented_data_list)}')
    return augmented_data_list

def get_data_from_slice(data, loading_configs):
    """
    loading_configs should be a list of dictionaries e.g. [{'key': 'LMA_label', 'LMA_threshold'}]
    """
    loaded_data = {}
    for loading_config in loading_configs:
        key = loading_config['key']
        output_key = loading_config.get('output_key', key)
        if key == 'TOS':
            loaded_data[output_key] = data['TOSAnalysis']['TOSfullRes_Jerry']
        elif key == 'LMA_sector_labels':
            LMA_threshold = loading_config.get('LMA_threshold', 25)
            loaded_data[output_key] = (data['TOSAnalysis']['TOSfullRes_Jerry'] > LMA_threshold).astype(int)
        elif key == 'strain_matrix':
            loaded_data[output_key] = data['StrainInfo']['CCmid']
        elif key == 'DENSE_myo_masks':
            generate_from_disp = loading_config.get('generate_from_disp', False)
            generate_from_data_key = loading_config.get('generate_from_data_key', 'DENSE_displacement_field_X')
            if generate_from_disp:
                loaded_data[output_key] = (np.abs(data[generate_from_data_key]) > 1e-5).astype(float)
            else:
                loaded_data[output_key] = data[key]
        elif key in ['cine_myo_masks', 'cine_lv_myo_masks', 'cine_lv_myo_masks_merged']:
            keep_interpolated_frames = loading_config.get('keep_interpolated_frames', False)
            interpolated_frames_indicator_key = loading_config.get('interpolated_frames_indicator_key', 'cine_lv_myo_masks_merged_is_interpolated_labels')
            full_data = data[key]
            if keep_interpolated_frames:
                loaded_data[output_key] = full_data
            else:
                loaded_data[output_key] = full_data[..., np.where(data[interpolated_frames_indicator_key]==0)[0]]
        elif key == 'DENSE_displacement_field':
            generate_from_components = loading_config.get('generate_from_components', True)
            # scale_by_pixelspace = loading_config.get('scale_by_pixelspace', False)
            # disp_scale = data['DENSEInfo']['PixelSpacing'][0] if scale_by_pixelspace else 1
            disp_scale = 1
            if generate_from_components:
                loaded_data[output_key] = np.stack([data['DENSE_displacement_field_X'], data['DENSE_displacement_field_Y']], axis=0) * disp_scale
            elif key in data.keys():
                loaded_data[output_key] = data[key] * disp_scale
            elif key not in data.keys() and ('DENSE_displacement_field_X' in data.keys() and 'DENSE_displacement_field_Y' in data.keys()):
                loaded_data[output_key] = np.stack([data['DENSE_displacement_field_X'], data['DENSE_displacement_field_Y']], axis=0) * disp_scale
        elif key == 'DENSE_myo_contour':
            loaded_data[output_key] = data['ROIInfo']['Contour']
        elif key in ['PositionA', 'PositionB']:
            loaded_data[output_key] = data['AnalysisInfo'][key]
        elif key in ['ori_H']:
            loaded_data[output_key] = data['StrainInfo']['X'].shape[0]
        elif key in ['ori_W']:
            loaded_data[output_key] = data['StrainInfo']['X'].shape[1]
        else:
            loaded_data[output_key] = data[key]

        # select part of the data if needed
        if loading_config.get('use_only_original', False) and 'interp_frame_indicatior' in loading_config.keys():
            interp_frame_indicatior = data[loading_config['interp_frame_indicatior']]
            loaded_data[output_key] = loaded_data[output_key][..., np.where(interp_frame_indicatior==0)[0]]
        if loading_config.get('scale_by_pixelspace', False):
            loaded_data[output_key] *= data['DENSEInfo']['PixelSpacing'][0]
            # print(f'times disp with {data["DENSEInfo"]["PixelSpacing"][0]}')
    return loaded_data

import copy
def try_merge_displacements(datum: dict):
    """
    Try to merge the displacement X and Y to a single displacement field
    
    Specially, for any key of the dictionary that (1) includes "disp" and (2) have data ends with both "X" and "Y"
    e.g. "DENSE_displacement_field_X" and "DENSE_displacement_field_Y"
    The new key should be named as the original key without the "X" or "Y" at the end
    Also, is the end of the new key is "-" or "_", it should be also removed
    """
    ori_datum_keys = copy.copy(list(datum.keys()))
    for key in ori_datum_keys:
        if 'disp' in key and key.endswith('X'):
            key_Y = key[:-1] + 'Y'
            if key_Y in datum.keys():
                new_key = key[:-1]
                if new_key.endswith('_') or new_key.endswith('-'):
                    new_key = new_key[:-1]
                datum[new_key] = np.stack([datum[key], datum[key_Y]], axis=0)
                datum.pop(key)
                datum.pop(key_Y)
    return datum

def load_slices_from_npy_file(npy_filename, data_config=None, slices_data_list=None):
    LMA_threshold = data_config.get('LMA_threshold', 25)
    print('loading cine data from npy file...', end='')
    slices_data_list = np.load(npy_filename, allow_pickle=True).tolist()
    print('DONE!')
    print(f'{len(slices_data_list)} slices loaded')
    for datum in slices_data_list:
        datum['augmented'] = False
    
    n_read = data_config.get('n_read', -1)
    n_read = len(slices_data_list) if n_read == -1 else n_read
    slices_data_list = slices_data_list[:n_read]

    # augmentation
    print('augmenting data...', end='')
    augmented_data = augment_all_data(slices_data_list, data_config)
    print('DONE!')
    print(type(augmented_data))
    print('len(augmented): ', len(augmented_data))
    slices_data_list += augmented_data
    print(f'# of data after augmentation: {len(slices_data_list)}')

    # normalization
    normalize_interpolated_cine_key = data_config['loading'].get('normalize_interpolated_cine_key', False)
    def normalize_img(img):
        # normalize the range of image to [0, 1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img

    
    data_to_feed = data_config['loading'].get('data_to_feed', [{'key': 'LMA_label', 'LMA_threshold': 25}])
    try_merge_displacements_flag = data_config['loading'].get('try_merge_displacements', True)
    # for additional_data_keys in ['augmented', 'cine_slice_idx', 'cine_slice_location', 'DENSE_slice_mat_filename', 'DENSE_slice_location']:
    #     data_to_feed.append({'key': additional_data_keys})
    loaded_data_list = []
    for slice_idx, datum in enumerate(slices_data_list):
        # if 'TOSAnalysis' not in datum.keys():
        #     print('Warning: TOSAnalysis not found in slice_data of patient', datum['patient_id'])
        #     continue
        loaded_datum = get_data_from_slice(datum, data_to_feed)        
        loaded_datum['augmented'] = datum['augmented']
        # loaded_datum['cine_slice_idx'] = int(datum['cine_slice_idx'])
        # loaded_datum['cine_slice_location'] = float(datum['cine_slice_location'])
        loaded_datum['DENSE_slice_mat_filename'] = str(datum.get('DENSE_slice_mat_filename', 'None'))
        loaded_datum['DENSE_slice_location'] = float(datum.get('DENSE_slice_location', -1e-3))
        
        loaded_datum['subject_id'] = datum['patient_id']
        loaded_datum['slice_idx'] = slice_idx
        loaded_datum['slice_full_id'] = f'{datum["patient_id"]}-{slice_idx}'
        # loaded_datum['SequenceInfo'] = datum['SequenceInfo']
        

        if try_merge_displacements_flag:
            loaded_datum = try_merge_displacements(loaded_datum)

        loaded_data_list.append(loaded_datum)
    return loaded_data_list
        

