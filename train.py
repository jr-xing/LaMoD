import random, string, datetime
import numpy as np
from pathlib import Path
import wandb

# 1. load config
from modules.config.config import load_config_from_json
from modules.config.config import get_args, update_config_by_args, update_config_by_undefined_args
args, undefined_args = get_args()
print('args: \n', args, '\n')

config = load_config_from_json(args.config_file)
# override config file wandb setting if given command line parameter
config = update_config_by_args(config, args)
config = update_config_by_undefined_args(config, undefined_args)
# config = load_config_from_json('/p/mmcardiac/Jerry/code/DENSE-guided-Cine-Registration/configs/training_configs/2023-10/2023-10-26-registraion-with-supervision-and-TOS-regression.json')

# config['training']['batch_size'] = 70
import json
print(json.dumps(config, indent=4))
# quit()

# 2. load all data
from modules.data import load_data
all_data = load_data(config['data'])

# append new data
from modules.data.processing.create_additional_data import create_addition_data
all_data = create_addition_data(all_data, config.get('additional_data', {}))

# 3. data splitting
# from modules.data.data_split import split_data
# data_splits = split_data(all_data, config['data_split'])
train_split = {
    'info': {},
    'data': [d for d in all_data if d['subject_id'].replace('-DENSE', '').replace('-Cine', '') not in config['data_split']['splits']['train']['exclude_patterns']]
}
val_split = {
    'info': {},
    'data': [d for d in all_data if d['subject_id'].replace('-DENSE', '').replace('-Cine', '') in config['data_split']['splits']['val']['patterns']]
}
test_split = {
    'info': {},
    'data': [d for d in all_data if d['subject_id'].replace('-DENSE', '').replace('-Cine', '') in config['data_split']['splits']['test']['patterns']]
}

# remove the "overflow" data
# n_non_zeros = 0
# myo_mask_top_row_sum_non_zero_indices = []
# for datum_idx, datum in enumerate(train_split['data']):
#     myo_mask_top_row_sum = np.sum(np.abs(datum['DENSE_Lag_displacement_field'][0,0,:,0]))
#     if myo_mask_top_row_sum > 2:
#         n_non_zeros += 1
#         # print(datum_idx, myo_mask_top_row_sum)
#         myo_mask_top_row_sum_non_zero_indices.append(datum_idx)
# # pop the data in myo_mask_top_row_sum_non_zero_indices
# for i in reversed(myo_mask_top_row_sum_non_zero_indices):
#     train_split['data'].pop(i)
# print(f'popped {n_non_zeros} data with non-zero top row sum of myo mask: {myo_mask_top_row_sum_non_zero_indices}')

data_splits = {
    'train': train_split,
    'val': val_split,
    'test': test_split
}

n_train_data_to_use = config['data_split']['splits']['train'].get('n_data_to_use', -1)
if n_train_data_to_use > 0:
    data_splits['train']['data'] = data_splits['train']['data'][-n_train_data_to_use:]

# for split_name, split in data_splits.items():
#     print(split_name, len(split['data']))

# 4. Building dataset
from modules.data.dataset import build_datasets
datasets = build_datasets(config['datasets'], data_splits)
# print the length of each dataset
for dataset_name, dataset in datasets.items():
    print(f'{dataset_name} dataset length: {len(dataset)}')

# 6. Building model
from models import build_model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
networks = {}
for model_name, model_config in config['networks'].items():
    networks[model_name] = build_model(model_config)
    networks[model_name] = networks[model_name].to(device)
# model = build_model(config['network'])

# if have more than 1 GPU, use DataParallel
# if torch.cuda.device_count() > 1:
#     for model_name, model in networks.items():
#         networks[model_name] = torch.nn.DataParallel(model)
#         print(f"model {model_name} is using DataParallel")

# 7. Training
from modules.trainer import build_trainer
# import numpy as np
# from pathlib import Path
# if config['others'].get('use_wandb', True):
#     if hasattr(wandb_experiment, 'run'):
#         saving_dir = Path(wandb_experiment.run.dir)
#     else:
#         saving_dir = Path(wandb_experiment.dir)
# else:
#     saving_dir = Path(config['saving'].get('saving_dir', './test_results'))
#     saving_dir.mkdir(parents=True, exist_ok=True)
# print('experiment results saving dir: ', saving_dir)
use_wandb=config['others'].get('use_wandb', True)
if use_wandb:    
    # wandb_base_dir = './exp_results'
    # wandb_base_dir = '/p/miauva/data/Jerry/exp-results'
    wandb_base_dir = '/scratch/jx8fh/exp-results'
    # wandb_base_dir = wandb_base_dir
    # exp_random_ID = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    # exp_name = full_config['info'].get('experiment_name', 'unnamed') + f'-{exp_random_ID}'
    exp_name = config['info'].get('experiment_name', 'unnamed')
    exp_name_with_date = datetime.datetime.now().strftime('%Y-%m-%d')+ '-' + exp_name
    # use_wandb = full_config['others'].get('use_wandb', False)
    wandb_path = Path(wandb_base_dir, f"{datetime.datetime.now().strftime('%Y-%m')}/{exp_name_with_date}-wandb")
    wandb_path.mkdir(parents=True, exist_ok=True)
    
    wandb_visualize_interval = config['others'].get('wandb_visualize_interval', -1) # -1 means no visualization
    # if wandb_visualize_interval is a float number, it means the interval is a fraction of the total number of epochs
    # convert it to an integer
    # if wandb_visualize_interval > 0 and isinstance(wandb_visualize_interval, float):
    #     wandb_visualize_interval = int(wandb_visualize_interval * used_train_config['epochs'])
    
    wandb_experiment = wandb.init( 
        project = config['info'].get('project_name', 'trials'),
        entity = "jrxing", 
        save_code = True,
        name = config['info']['experiment_name'] if config['info'].get('use_experiment_name', False) else None,
        dir = wandb_path,
        resume = 'allow', 
        anonymous = 'must',
        mode='online' if use_wandb else 'disabled')
    exp_save_dir = wandb.run.dir
else:
    # exp_save_dir = Path(used_full_config['others']['save_dir'])
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # exp_save_dir = Path('./test_results') / f'{current_time}-{config["info"]["experiment_name"]}'
    # exp_save_dir = Path('/p/miauva/data/Jerry/exp-trial-results') / f'{current_time}-{config["info"]["experiment_name"]}'
    exp_save_dir = Path('/scratch/jx8fh/exp-trial-results') / f'{current_time}-{config["info"]["experiment_name"]}'
    wandb_experiment = None

exp_save_dir = Path(exp_save_dir)
exp_save_dir.mkdir(parents=True, exist_ok=True)
# wandb_base_dir = '/p/miauva/data/Jerry/exp-results'
trainer = build_trainer(config['training'], device, config)
# LMA_task = trainer.LMA_task
training_seed = config['training'].get('seed', 2434)
torch.manual_seed(training_seed)
exp_dict, trained_models, wandb_experiment = trainer.train(
    models=networks, 
    datasets=datasets, 
    trainer_config=config['training'], 
    full_config=config, 
    device=device,
    use_tensorboard=config['others'].get('use_tensorboard', True),
    tensorboard_log_dir='tensorboard',
    use_wandb=config['others'].get('use_wandb', True),
    wandb_experiment=wandb_experiment,
    enable_wandb_upload=config['others'].get('enable_wandb_upload', True),
    exp_save_dir=exp_save_dir)

# 8. Test
val_pred, val_performance_dict, _ = trainer.test(
    models=trained_models, 
    datasets=datasets, 
    trainer_config=config['training'], 
    full_config=config, 
    device=device,
    wandb_experiment=wandb_experiment,
    target_dataset='val',
    mode='test')
print('done')

test_pred, test_performance_dict, _ = trainer.test(
    models=trained_models, 
    datasets=datasets, 
    trainer_config=config['training'], 
    full_config=config, 
    device=device,
    wandb_experiment=wandb_experiment,
    target_dataset='test',
    mode='test')

# if config['training'].get('full_test_inference', False):
#     inference_result_dicts = trainer.test_inference(data_splits['test']['data'], 
#                            trained_models, 
#                            test_config=config['training'], 
#                            wandb_experiment=wandb_experiment)

# 9. Save results
# save (val and) test predictions as npy file

# save_preds = config['saving'].get('save_prediction', False)
# if save_preds:
import numpy as np
data_keys_to_pop = config['saving'].get('data_keys_to_pop', [])
for test_datum in test_pred:
    for key in data_keys_to_pop:
        if key in test_datum:
            test_datum.pop(key)
# val_save_filename = config['saving'].get('val_save_filename', 'val_pred.npy')
test_save_filename = config['saving'].get('test_save_filename', 'test_pred.npy')
# np.save(Path(saving_dir, val_save_filename), val_pred)
np.save(Path(exp_save_dir, test_save_filename), test_pred)

# if config['training'].get('full_test_inference', False):
#     inference_save_filename = config['saving'].get('inference_save_filename', 'test_inference_full.npy')
#     np.save(Path(exp_save_dir, inference_save_filename), inference_result_dicts)

# 10. save all models
for model_name, model in trained_models.items():
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), Path(exp_save_dir, model_name + '.pth'))
        print(f'model {model_name} saved to {Path(exp_save_dir, model_name + ".pth")}')

# save the config file
import json
with open(Path(exp_save_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4, sort_keys=False)
print('config file saved to: ', Path(exp_save_dir, 'config.json'))

# save performance
final_epoch_loss_dict = exp_dict['epoch_loss_dict']
final_epoch_loss_dict.update(val_performance_dict)
final_epoch_loss_dict.update(test_performance_dict)
# save as json
with open(Path(exp_save_dir, 'performance.json'), 'w') as f:
    json.dump(final_epoch_loss_dict, f, indent=4, sort_keys=False)

# plot
try:
    trainer.plot_results(test_pred, save_fig_dir=exp_save_dir, save_plot=True)
except Exception as e:
    print('plotting failed: ', e)