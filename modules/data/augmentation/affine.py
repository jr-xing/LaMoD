import numpy as np
from skimage.transform import rotate as skrotate


def affine(datum: dict, config=None):
    if config is None:
        config = {
            'scale': {},
            'rotate': {
                'n_rotate_sectors': 1,

            },
            'shear': {},
            'translate': {},            
        }

default_augment_data_keys = [
    'cine_lv_myo_masks_merged', 'DENSE_displacement_field_merged', 'CCmid', 'TOSfullRes_Jerry',
    'cine_lv_myo_masks_merged_disp_S_T_phi', 'DENSE_displacement_field',
    'cine_images_merged']
# default_augment_data_keys = ['cine_lv_myo_masks_merged', 'DENSE_displacement_field_merged', 'cine_lv_myo_masks_displacement_field', 'CCmid', 'TOSfullRes_Jerry']
# default_augment_data_keys = [
#     'CCmid', 'TOSfullRes_Jerry', 'DENSE_displacement_field']
def translate(datum: dict, translate_y, translate_x, translate_data_keys=None):
    if translate_data_keys is None:
        translate_data_keys = default_augment_data_keys
    datum_translated = {}
    for key in translate_data_keys:
        try:
            if key == 'cine_lv_myo_masks_merged':
                translated_cine_lv_myo_masks = np.roll(datum[key], (translate_y, translate_x), axis=(0, 1))
                datum_translated[key] = translated_cine_lv_myo_masks
            elif key in ['DENSE_displacement_field_merged', 'cine_lv_myo_masks_displacement_field', 'cine_lv_myo_masks_merged_disp_S_T_phi', 'DENSE_displacement_field']:
                translated_displacement_field_X = np.roll(datum[key+'_X'], (translate_y, translate_x), axis=(0, 1))
                translated_displacement_field_Y = np.roll(datum[key+'_Y'], (translate_y, translate_x), axis=(0, 1))
                datum_translated[key+'_X'] = translated_displacement_field_X
                datum_translated[key+'_Y'] = translated_displacement_field_Y
            elif key == 'CCmid':
                # the strain matrix should not change
                datum_translated['StrainInfo'] = {key: datum['StrainInfo'][key]}
            elif key == 'TOSfullRes_Jerry':
                # the TOS curve should not change
                datum_translated['TOSAnalysis'] = {key: datum['TOSAnalysis'][key]}
            else:
                raise ValueError('Unsupport data key: {}'.format(key))
        except Exception as e:
            # print(e)
            # print('key {} not in datum'.format(key))
            continue
    return datum_translated

def rotate(datum: dict, n_rotate_sectors, n_total_sectors=126, rotate_data_keys=None):
    if rotate_data_keys is None:
        rotate_data_keys = default_augment_data_keys

    rotate_angle_degree =  - n_rotate_sectors * 360 / n_total_sectors

    # Rotate the data
    datum_rorated = {}
    for key in rotate_data_keys:
        # if key not in datum.keys():
        #     print('key {} not in datum'.format(key))
        #     continue
        try:
            if key == 'cine_lv_myo_masks_merged':
                rotated_cine_lv_myo_masks = skrotate(datum[key], rotate_angle_degree, resize=False, preserve_range=True, order=0)
                datum_rorated[key] = rotated_cine_lv_myo_masks
            elif key in ['DENSE_displacement_field_merged', 'cine_lv_myo_masks_displacement_field', 'cine_lv_myo_masks_merged_disp_S_T_phi', 'DENSE_displacement_field']:
                rotated_displacement_field_X = skrotate(datum[key+'_X'], rotate_angle_degree, resize=False, preserve_range=True, order=0)
                rotated_displacement_field_Y = skrotate(datum[key+'_Y'], rotate_angle_degree, resize=False, preserve_range=True, order=0)
                datum_rorated[key+'_X'] = rotated_displacement_field_X
                datum_rorated[key+'_Y'] = rotated_displacement_field_Y
            elif key == 'CCmid':
                strain_matric_rolled = np.roll(datum['StrainInfo'][key], n_rotate_sectors, axis=0)
                datum_rorated['StrainInfo'] = {key: strain_matric_rolled}
                # datum_rorated['StrainInfo']['CCmid'] = strain_matric_rolled
            elif key == 'TOSfullRes_Jerry':
                TOS_curve_rolled = np.roll(datum['TOSAnalysis'][key], n_rotate_sectors, axis=0)
                datum_rorated['TOSAnalysis'] = {key: TOS_curve_rolled}
            else:
                raise ValueError('Unsupport data key: {}'.format(key))
        except Exception as e:
            # print(e)
            # print('key {} not in datum'.format(key))
            continue
        
    return datum_rorated