import numpy as np
import torch
def check_dict(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                print('{:<60} {:<20}'.format(key, str(value)))
            else:
                print('{:<60} {:<20}'.format(key, str(value.shape)))
        elif isinstance(value, torch.Tensor):
            print('{:<60} {:<20}'.format(key, str(value.shape)))
        elif isinstance(value, dict):
            print('{:<60} {:<20}'.format(key, str(value.keys())))
        elif isinstance(value, list):
            print('{:<60} {:<20}'.format(key, 'list: (' + str(len(value))+')'))
        else:
            print('{:<60} {:<20}'.format(key, str(value)))


def split_vol_to_registration_pairs(vol, split_method:str='Lagrangian', output_dim=3):
    """
    Split the input volumes into pair of images for registraion

    vol: [batch_size, 2, n_frames, height, width]
    return: [batch_size*n_frames, 2, height, width]

    split_method: 'Lagrangian' or 'Eulerian'
    if split_method == 'Lagrangian', then the the pairs should be the first frame and each of the following frames.
    if split_method == 'Eulerian', then the pairs should be the every pair of adjacent frames.
    """

    batch_size, n_channels, n_frames, height, width = vol.shape
    assert n_frames > 1, f'n_frames should be larger than 1, but got {n_frames}'

    if split_method == 'Lagrangian':
        src = vol[:, :, :1, :, :].repeat(1, 1, n_frames-1, 1, 1)#.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = vol[:, :, 1:, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
    elif split_method == 'Eulerian':
        src = vol[:, :, :-1, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = vol[:, :, 1:, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
    else:
        raise ValueError(f'Unrecognized split_method: {split_method}')

    if output_dim == 2:
        src = src.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = tar.reshape(batch_size*(n_frames-1), n_channels, height, width)

    return src, tar

import scipy.io as sio
import numpy as np
def loadmat(filename, as_dict=True, ndarray_to_list=False):
    data = sio.loadmat(str(filename), struct_as_record=False, squeeze_me=True)
    return mat2dict(data, as_dict=as_dict, ndarray_to_list=ndarray_to_list)

def mat2dict(matobj, as_dict=True, ndarray_to_list=False):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d, multidim_ndarray_to_list=False):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key], multidim_ndarray_to_list)
            if isinstance(d[key], np.ndarray):
                if d[key].ndim == 1 or multidim_ndarray_to_list:
                    d[key] = _tolist(d[key], multidim_ndarray_to_list)
                else:
                    pass            
        return d
    
    def _todict(matobj, multidim_ndarray_to_list=False):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        # print('matobj._fieldnames', matobj._fieldnames)
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                # print(elem.ndim)
                if multidim_ndarray_to_list and elem.ndim > 1:
                    # if the element has more than 1 dim and we allow converting multidim ndarray into list
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            else:
                d[strg] = elem
        return d
    
    def _tolist(ndarray, multidim_ndarray_to_list=False):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem, multidim_ndarray_to_list))
            elif isinstance(sub_elem, np.ndarray):
                # print('sub_elem.ndim')
                if sub_elem.ndim == 1 or multidim_ndarray_to_list:
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            else:
                elem_list.append(sub_elem)
        return elem_list
  
    # data = sio.loadmat(str(filename), struct_as_record=False, squeeze_me=True)
    
    # if as_dict:
    if type(matobj) == sio.matlab.mio5_params.mat_struct:
        return _check_keys(_todict(matobj), ndarray_to_list)
    else:
        return _check_keys(matobj, ndarray_to_list)
    # else:
    #     return data