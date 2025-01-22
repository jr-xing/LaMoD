# https://stackoverflow.com/questions/12818146/python-argparse-ignore-unrecognised-arguments
import argparse
import copy
def get_args():
    parser = argparse.ArgumentParser(description='DENSE-Guided Cine Registration')
    # Uncategorized
    # parser.add_argument('--repeat_times', '-rp', type=int, default=-1, help='Repeat times')
    # Info
    parser.add_argument('--exp-name', dest = 'exp_name', type=str, default=argparse.SUPPRESS, help='Name of experiment')
    parser.add_argument('--use-exp-name', dest = 'use_exp_name', action = 'store_true', help='Use Name of experiment')
    parser.add_argument('--use_exp_name', dest = 'use_exp_name', action = 'store_true', help='Use Name of experiment')
    # Data
    # loading
    parser.add_argument('--n-read', dest = 'n_read', type=int, default=argparse.SUPPRESS, help='Number of data to read for each dataset. Default: -1 (read all available)')
    parser.add_argument('--n_read', dest = 'n_read', type=int, default=argparse.SUPPRESS, help='Number of data to read for each dataset. Default: -1 (read all available)')
    # train-test-split
    parser.add_argument('--no-repeat-data', dest='no_repeat_data', action = 'store_true', help='Whether allow repeating data for class imbalance')
    # preprocessing
    parser.add_argument('--mask-out', dest = 'mask_out', type=str, default=argparse.SUPPRESS, help='Whether mask out. False or mask data type')
    parser.add_argument('--crop-to-myocardium-size', dest = 'crop_to_myocardium_size', type=str, default=argparse.SUPPRESS, help='Crop size, e.g. 120,120')
    parser.add_argument('--resize-img-size', dest = 'resize_img_size', type=str, default=argparse.SUPPRESS, help='target size, e.g. 224,224')
    # train-transform
    # validate-transform
    # network
    parser.add_argument('--load-pretrained-model', dest = 'load_pretrained_model', type=str, default=argparse.SUPPRESS, help='Whether load pretrained model')
    parser.add_argument('--load-pretrained-transformer', dest = 'load_pretrained_transformer', type=str, default=argparse.SUPPRESS, help='Whether load pretrained transformer')
    parser.add_argument('--pretrained-model-path',dest='pretrained_model_path', type=str, default=argparse.SUPPRESS, help='Path of pretrained model')
    # training    
    parser.add_argument('--epochs', '-e', type=int, default=argparse.SUPPRESS, help='Number of epochs (default -1, i.e. not specified)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=argparse.SUPPRESS, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=argparse.SUPPRESS,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=argparse.SUPPRESS,
                        help='Weight decay', dest='weight_decay')
    parser.add_argument('--optimizer', '-o', dest='optimizer', type=str, default=argparse.SUPPRESS, help='optimizer')
    parser.add_argument('--mixed-precision', '-amp', dest = 'amp', type=bool, default=argparse.SUPPRESS, help='Whether use mixed precision')
    parser.add_argument('--pre-load-data', dest='pre_load_data', type=bool, default=argparse.SUPPRESS, help='If load all data in memory')    
    parser.add_argument('--early-stop-patience', dest='early_stop_patience', type=int, default=argparse.SUPPRESS, help='defalut: 20')
    parser.add_argument('--early-stop-metric', dest='early_stop_metric', type=str, default=argparse.SUPPRESS, help='what metric to decide early stop')

    # Test
    parser.add_argument('--test', dest='test', type=str, default=argparse.SUPPRESS, help='Whether test network (default: False)')    
    parser.add_argument('--test-config-file', dest='test_config_file', type=str, default=argparse.SUPPRESS, help='Config file for test')    

    # loss
    parser.add_argument('--loss-1-weight', dest='loss_1_weight', type=float, default=argparse.SUPPRESS, help='Weight of loss 1')
    parser.add_argument('--loss-2-weight', dest='loss_2_weight', type=float, default=argparse.SUPPRESS, help='Weight of loss 2')

    # saving
    parser.add_argument('--save-nothing', dest='save_nothing', type=str, default='false', help='If true, save nothing')
    parser.add_argument('--saving_dir', dest='saving_dir', type=str, default=argparse.SUPPRESS, help='saving dir')
    # others
    parser.add_argument('--config-file', dest='config_file', help='config file relative path', type=str, default='./configs/training_configs/2023-10/2023-10-08-registraion-and-LMA-sector-classification.json')
    parser.add_argument('--script-file', dest='script_file', help='script file relative path', type=str, default='training_scripts/2023-03/2023-03-15-Quickdraw-func-wandb.py')
    parser.add_argument('--use-wandb', dest='use_wandb', help='whether using wandb to log or parameter tuning', type=str, default='False')
    parser.add_argument('--wandb-sweep', dest='wandb_sweep', help='whether using wandb sweep hyperparameter tuning', type=str, default='False')
    parser.add_argument('--wandb-sweep-file', dest='wandb_sweep_file', help='config file relative path', type=str, default='./configs/test_wandb_sweep.yaml')
    parser.add_argument('--print-config', dest='print_config', help='whether print config. Disable for debug', type=str, default='true')
    parser.add_argument('--valid-period', dest='valid_period', help='whether print config. Disable for debug', type=int, default=argparse.SUPPRESS)
    
    # parser.add_argument('--scale', '-s', type=float, default=argparse.SUPPRESS, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=argparse.SUPPRESS,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    args, undefined_args = parser.parse_known_args()
    return args, undefined_args
    # return parser.parse_args()

def update_config_by_args(config_ori, args):
    # config = config_ori.copy()
    config = copy.deepcopy(config_ori)
    for arg, value in vars(args).items():
        if arg == 'repeat_times': pass
        # Info
        elif arg == 'config_file': pass
        elif arg == 'script_file': pass
        elif arg == 'exp_name': config['info']['experiment_name'] = value
        elif arg == 'use_exp_name': config['info']['use_experiment_name'] = value
        # Data
        # loading
        elif arg == 'n_read':
            for dataset_config in config['data']:
                dataset_config['loading']['n_read'] = value
        # train-test-split
        elif arg == 'no_repeat_data': 
            if value:
                for data_split_dict_idx, data_split_dict in enumerate(config['data_split']['paras']):
                    config['data_split']['paras'][data_split_dict_idx]['repeat_times'] = 0
        # preprocessing
        elif arg == 'mask_out':
            if value.lower() in ['false', 'f']:
                pass
            else:
                config['preprocessing'].insert(0, {'method': 'maskout', 'mask_type': value})
        elif arg == 'crop_to_myocardium_size':
            # print("crop_to_myocardium_size!")
            # print(value)
            # print(config['preprocessing'])
            # convert value from string (e.g. '120,120') to list ([120,120])
            size = [int(val) for val in value.strip('(*)').split(',')]
            # print(size)
            # get the index of crop_to_myocardium method
            for preprocessing_dict_idx, preprocessing_dict in enumerate(config['preprocessing']):
                if preprocessing_dict['method'] == 'crop_to_myocardium':
                    preprocessing_dict['size'] = size
                    break
            print(config['preprocessing'])
        elif arg == 'resize_img_size':
            shape = [int(val) for val in value.strip('(*)').split(',')]
            preprocessing_terms = [prep['method'] for prep in config['preprocessing']]
            try:
                resize_idx = preprocessing_terms.index('resize')
                config['preprocessing']['shape'] = shape
            except:
                config['preprocessing'].insert(len(config['preprocessing']), {'method': 'resize', 'shape': shape})
        # train-transform
        # validate-transform
        # network
        elif arg == 'load_pretrained_model': config['network']['load pretrained model'] = False if value.lower() in ['false', 'f'] else True
        elif arg == 'load_pretrained_transformer': config['network']['load pretrained transformer'] = True if value.lower() in ['true', 't'] else False
        elif arg == 'pretrained_model_path': config['network']['pretrained model path'] = value
        # training
        elif arg == 'epochs': config['training']['epochs'] = value
        elif arg == 'batch_size': config['training']['batch_size'] = value
        elif arg == 'learning_rate': 
            for optimizer_name in config['training']['optimizers'].keys():
                config['training']['optimizers'][optimizer_name]['learning_rate'] = value
        elif arg == 'weight_decay': 
            for optimizer_name in config['training']['optimizers'].keys():
                config['training']['optimizers'][optimizer_name]['weight_decay'] = value
        elif arg == 'amp': config['training']['mixed Precision'] = value
        elif arg == 'pre_load_data': config['training']['preload data'] = value        
        elif arg == 'early_stop_patience': config['training']['early stop']['patience'] = value
        elif arg == 'early_stop_metric': config['training']['early stop']['metric'] = value
        # test
        elif arg == 'test': config['training']['test'] = False if value.lower() in ['false', 'f'] else True
        elif arg == 'test_config_file': config['training']['test config file'] = value
        # loss
        elif arg == 'loss_1_weight': config['loss']['input_GT_pred_role_pairs'][0]['weight'] = value
        elif arg == 'loss_2_weight': config['loss']['input_GT_pred_role_pairs'][1]['weight'] = value
        # saving
        elif arg == 'save_nothing': 
            if value.lower() in ['true', 't']:
                for k in ['save final model', 'save checkpoint', 'save prediction', 'save KeyboardInterrupt', 'save_pred_images']:
                    config['saving'][k] = False
        elif arg == 'saving_dir':
            config['saving']['saving_dir'] = value
        # others        
        elif arg == 'use_wandb': config['others']['use_wandb'] = True if value.lower() in ['true', 't', 'yes', 'y'] else False
        elif arg == 'wandb_sweep': config['others']['wandb_sweep'] = True if value.lower() in ['true', 't', 'yes', 'y'] else False
        elif arg == 'wandb_sweep_file': config['others']['wandb sweep file'] = value        
        elif arg == 'enable_wandb_upload': config['others']['enable_wandb_upload'] = True if value.lower() in ['true', 't', 'yes', 'y'] else False
        elif arg == 'print_config':
            config['others']['print_config'] = True if value.lower() in ['true', 't', 'yes', 'y'] else False
        elif arg == 'valid_period':
            config['others']['valid_period'] = value
        elif arg in ['optimizer']:
            pass
        elif arg.startswith('__'):
            pass
        else:
            raise ValueError(f'Unsupported argument: {arg}')
    
    return config

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def try_convert_to_number(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        try: 
            float(s)
            return float(s)
        except ValueError:
            return s

def try_convert_to_bool(s):
    if type(s) is str:
        if s.lower() == 'false':
            return False
        elif s.lower() == 'true':
            return True
        else:
            return s
    else:
        return s

def update_config_by_undefined_args(config_ori, undefined_args: list):
    # undefined_args should be a list of strings
    # it should follow the structure as config dict
    # the hierarchy is splitted by --
    # and the key-value is splitted by =
    # e.g. ['config--training--learning_rate=0.5'] makes config['training]['learning_rate'] = 0.5
    config = copy.deepcopy(config_ori)
    for arg_value in undefined_args:
        # print('arg_value=', arg_value)
        arg_value = arg_value.strip().lstrip('--')
        arg, value = arg_value.split('=')
        arg_split = arg.split('--')
        subconfig = config
        for layer, key in enumerate(arg_split):
            # key = try_convert_to_number(key)
            if key.startswith('INDEX'):
                key = int(key.split('INDEX')[-1])
            if layer < len(arg_split) - 1:
                # print('subconfig', subconfig)
                subconfig = subconfig[key]
            else:
                value = try_convert_to_number(value)
                value = try_convert_to_bool(value)
                subconfig[key] = value
    return config

# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
import collections.abc
def update_config_by_another_config(config_ori: dict, config_new: dict):
    config = copy.deepcopy(config_ori)
    # config = config.update(config_new)
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update_dict(config, config_new)

import json
def load_config_from_json(json_filename=None):
    if json_filename is None:
        json_filename = './configs/training_configs/2023-10/2023-10-08-registraion-and-LMA-sector-classification.json'
    config = json.load(open(json_filename))
    return config


if __name__ == '__main__':
    base_config_file_path = './tmp/test_segmentation_config.json'
    wandb_metadata_path = './tmp/wandb-metadata.json'
    base_config = load_config_from_json(base_config_file_path)
    wandb_metadata = load_config_from_json(wandb_metadata_path)
    parser = get_arg_parser()
    # correct args
    wandb_metadata_args = wandb_metadata['args']
    for arg_idx, arg in enumerate(wandb_metadata_args):
        if arg.startswith('-') and not arg.startswith('--'):
            wandb_metadata_args[arg_idx] = '-'+ arg
    for arg_idx, arg in enumerate(wandb_metadata_args):        
        wandb_metadata_args[arg_idx] = arg.replace('_', '-')


    args, undefined_args = parser.parse_known_args(wandb_metadata['args'])
    config = update_config_by_args(base_config, args)
    config = update_config_by_undefined_args(config, undefined_args)
    # for arg in wandb_metadata['args']:
    #     parser.parse_args(arg)