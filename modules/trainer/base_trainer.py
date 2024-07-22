import wandb
import torch
import numpy as np
import datetime, json, copy
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from modules.loss import LossCalculator

# Custom function to get a subset sampler
def get_subset_random_sampler(size, subset_size):
    indices = torch.randperm(size).tolist()
    subset_indices = indices[:subset_size]
    return SubsetRandomSampler(subset_indices)

class DummyLrScheduler:
    """
    Dummy learning rate scheduler that does nothing
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

class BaseTrainer:
    def __init__(self, trainer_config, device=None, full_config=None):
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')

    def batch_forward(self, batch, models, epoch_loss_dict, loss_name_prefix, mode='train', train_config={}, full_config={}):
        # reg_model = models[0]
        reg_model = models['registration']

        src, tar = batch['source_img'], batch['target_img']
        src = src.to(self.device)
        tar = tar.to(self.device)
        reg_pred_dict = reg_model(src, tar)

        pred_dict = {
            'displacement': reg_pred_dict['displacement'],
            'velocity': pred_dict['velocity'],
            'momentum': pred_dict['momentum'],
            'deformed_source': pred_dict['deformed_source']
        }
        target_dict = {
            'registration_target': tar,
        }
        total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)

        # update the epoch loss dict
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            if record_name not in epoch_loss_dict.keys():
                epoch_loss_dict[record_name] = loss_value
            else:
                epoch_loss_dict[record_name] += loss_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, losses_values_dict, pred_dict, target_dict

    def get_optimizers(self, models_dict, optimizers_config):
        optimizers = {}
        # for model_name, model in models_dict.items():
        #     optimizers[model_name] = torch.optim.Adam(model.parameters(), lr=optimizers_config[model_name]['learning_rate'])
        for optimizer_name, optimizer_config in optimizers_config.items():
            target_model_name = optimizer_config['target']
            optimizers[optimizer_name] = torch.optim.Adam(models_dict[target_model_name].parameters(), lr=optimizer_config['learning_rate'])
        return optimizers
    
    
    def get_lr_scheduler(self, optimizer, lr_scheduler_config):
        # lr_scheduler_config = optimization_config['DENSE_disp_pred']['lr_scheduler']
        lr_scheduler_type = lr_scheduler_config['type']
        lr_scheduler_enabled = lr_scheduler_config['enable']
        if not lr_scheduler_enabled:
            lr_scheduler = DummyLrScheduler(optimizer)
        elif lr_scheduler_type in ['CosineAnnealingLR']:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=lr_scheduler_config['T_max'])
        elif lr_scheduler_type in ['StepLR', 'steplr']:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=lr_scheduler_config.get('step_size', 30),
                gamma=lr_scheduler_config.get('gamma', 0.5))
        else:
            raise NotImplementedError(f'Learning rate scheduler {lr_scheduler_type} not implemented')
        return lr_scheduler

    def get_lr_schedulers(self, optimizers_dict, optimization_config):
        lr_schedulers = {}
        for optimizer_name, optimizer in optimizers_dict.items():
            lr_schedulers[optimizer_name] = self.get_lr_scheduler(optimizer, optimization_config[optimizer_name].get('lr_scheduler', {}))
        return lr_schedulers
    

    def initialize_wandb_experiment(self, full_config, wandb_base_dir, use_wandb):
        import random
        import string

        # wandb_base_dir = './exp_results'
        # wandb_base_dir = wandb_base_dir
        # exp_random_ID = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
        # exp_name = full_config['info'].get('experiment_name', 'unnamed') + f'-{exp_random_ID}'
        exp_name = full_config['info'].get('experiment_name', 'unnamed')
        exp_name_with_date = datetime.datetime.now().strftime('%Y-%m-%d')+ '-' + exp_name
        # use_wandb = full_config['others'].get('use_wandb', False)
        wandb_path = Path(wandb_base_dir, f"{datetime.datetime.now().strftime('%Y-%m')}/{exp_name_with_date}-wandb")
        wandb_path.mkdir(parents=True, exist_ok=True)
        
        wandb_visualize_interval = full_config['others'].get('wandb_visualize_interval', -1)

        if wandb_visualize_interval > 0 and isinstance(wandb_visualize_interval, float):
            wandb_visualize_interval = int(wandb_visualize_interval * used_train_config['epochs'])
        
        wandb_experiment = wandb.init( 
            project = full_config['info'].get('project_name', 'trials'),
            entity = "jrxing", 
            save_code = True,
            name = full_config['info']['experiment_name'] if full_config['info'].get('use_experiment_name', False) else None,
            dir = wandb_path,
            resume = 'allow', 
            anonymous = 'must',
            mode='online' if use_wandb else 'disabled')
        return wandb_experiment
    
    def models_mode_switch(self, models, mode):
        if mode == 'train':
            for model_name, model in models.items():
                model.train()
        elif mode == 'eval':
            for model_name, model in models.items():
                model.eval()

    def train(self, models: dict, datasets: dict, trainer_config=None, full_config=None, device=None, use_tensorboard=False, tensorboard_log_dir=None, early_stop=True, use_wandb=False, wandb_experiment = None, exp_save_dir = './test_results', enable_wandb_upload=True, collate_fn=None):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # unpack datasets
        test_as_val = used_train_config.get('test_as_val', False)
        train_dataset = datasets['train']
        if test_as_val:
            val_dataset = datasets['test']
        else:
            val_dataset = datasets['val']
        test_dataset = datasets['test']

        # Build dataloaders
        train_subset_ratio = used_train_config.get('train_subset_ratio', 1.0)
        if train_subset_ratio is None or train_subset_ratio > 0.99:
            train_dataloader = DataLoader(train_dataset, batch_size=used_train_config['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)
        else:
            subset_sampler = get_subset_random_sampler(len(train_dataset), int(train_subset_ratio * len(train_dataset)))
            train_dataloader = DataLoader(train_dataset, batch_size=used_train_config['batch_size'], sampler=subset_sampler, num_workers=0, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)


        # loss calculator
        # create loss caluator if not exists
        if not hasattr(self, 'loss_calculator'):
            self.loss_calculator = LossCalculator(used_full_config['losses'])
        if not hasattr(self, 'evaluator'):
            self.evaluator = LossCalculator(used_full_config.get('evaluation', {}))

        # optimizers
        self.optimizers = self.get_optimizers(models, used_train_config['optimization'])
        self.lr_schedulers = self.get_lr_schedulers(self.optimizers, used_train_config['optimization'])
        # print the lr schedulers info
        for lr_scheduler_name, lr_scheduler in self.lr_schedulers.items():
            print(f'{lr_scheduler_name} lr_scheduler: {lr_scheduler}')

        # progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))
        # early stop parameters
        if early_stop:
            best_models = {}
            best_val_loss = float('inf')
            best_epoch = 0
            # best_models = {}
            best_epoch_loss_dict = {}
            epochs_without_improvement = 0
            epochs_without_improvement_tolerance = used_train_config.get('epochs_without_improvement_tolerance', 10)

        # Experiment tracking
        if use_tensorboard:
            # current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=exp_save_dir)

        # evaluation metrics
        # self.batch_train_loss_dict = {}
        self.training_epoch_loss_dict_list = []
        # self.batch_val_loss_dict = {}
        # self.epoch_val_loss_dict = {}
        save_best_model_interval = used_train_config.get('save_best_model_interval', -1)
        # Training Loop
        epoch = -1
        for epoch in progress_bar:
            try:
                # initialize epoch_loss_dict
                epoch_loss_dict = {}
                # Train
                self.models_mode_switch(models, 'train')
                
                for train_batch_idx, train_batch in tqdm(enumerate(train_dataloader)):
                    train_loss, training_batch_loss_dict, training_batch_error_values_dict, train_pred_dict, train_target_dict = self.batch_forward(train_batch, models, loss_name_prefix='train', train_config=used_train_config, full_config=used_full_config, mode='train', curr_epoch=epoch)

                    # backward pass
                    for optimizer_name, optimizer in self.optimizers.items():
                        optimizer.zero_grad()

                    train_loss.backward()
                    for optimizer_name, optimizer in self.optimizers.items():
                        optimizer.step()

                    batch_eval_dict = self.batch_eval(train_batch, train_batch)

                    # Update the epoch_loss_dict by adding the loss values
                    for loss_name, loss_value in training_batch_loss_dict.items():
                        if loss_name in epoch_loss_dict.keys():
                            epoch_loss_dict[loss_name] += loss_value
                        else:
                            epoch_loss_dict[loss_name] = loss_value

                    # update the epoch_loss_dict by adding the error values
                    for error_name, error_value in training_batch_error_values_dict.items():
                        if error_name in epoch_loss_dict.keys():
                            epoch_loss_dict[error_name] += error_value
                        else:
                            epoch_loss_dict[error_name] = error_value
                            

                # update learning rate
                for lr_scheduler_name, lr_scheduler in self.lr_schedulers.items():
                    lr_scheduler.step()
                
                # update progress bar
                progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_loss.item():.3e}')  

                # Validate
                epoch_total_val_loss = 0
                self.models_mode_switch(models, 'eval')
                with torch.no_grad():
                    for val_batch_idx, val_batch in tqdm(enumerate(val_dataloader)):
                        val_loss, val_batch_loss_dict, val_batch_error_values_dict, _, _ = self.batch_forward(val_batch, models, loss_name_prefix='val', train_config=used_train_config, full_config=used_full_config)
                        epoch_total_val_loss += val_loss.item()
                        batch_eval_dict = self.batch_eval(val_batch, val_batch)

                        # update the epoch_loss_dict by adding the loss values
                        for loss_name, loss_value in val_batch_loss_dict.items():
                            if loss_name in epoch_loss_dict.keys():
                                epoch_loss_dict[loss_name] += loss_value
                            else:
                                epoch_loss_dict[loss_name] = loss_value

                        # update the epoch_loss_dict by adding the error values
                        for error_name, error_value in val_batch_error_values_dict.items():
                            if error_name in epoch_loss_dict.keys():
                                epoch_loss_dict[error_name] += error_value
                            else:
                                epoch_loss_dict[error_name] = error_value
                
                # update the epoch_loss_dict by averaging the loss values if the loss value is a torch.Tensor
                for loss_name, loss_value in epoch_loss_dict.items():
                    if loss_name.startswith('train'):
                        if isinstance(loss_value, torch.Tensor):
                            epoch_loss_dict[loss_name] = loss_value.item() / len(train_dataloader)
                        else:
                            epoch_loss_dict[loss_name] = loss_value / len(train_dataloader)
                    elif loss_name.startswith('val'):
                        if isinstance(loss_value, torch.Tensor):
                            epoch_loss_dict[loss_name] = loss_value.item() / len(val_dataloader)
                        else:
                            epoch_loss_dict[loss_name] = loss_value / len(val_dataloader)
            
                # print epoch_loss_dict with indentation
                # convert the data in epoch_loss_dict to json serializable format
                # import numpy as np
                for key, value in epoch_loss_dict.items():
                    if isinstance(value, np.ndarray):
                        epoch_loss_dict[key] = value.tolist()
                    elif isinstance(value, torch.Tensor):
                        epoch_loss_dict[key] = value.item()
                print(json.dumps(epoch_loss_dict, indent=4))

                # append epoch_loss_dict to epoch_loss_dict_list
                self.training_epoch_loss_dict_list.append(epoch_loss_dict)

                if use_wandb and enable_wandb_upload:
                    wandb_experiment.log(epoch_loss_dict, step=epoch)                
                
                if use_tensorboard:
                    for loss_name, loss_value in epoch_loss_dict.items():
                        writer.add_scalar(loss_name, loss_value, epoch)
                    # writer.add_scalar('validation?total_loss', epoch_total_val_loss, epoch)
                    # writer.close()
                    writer.flush()

                if early_stop:
                    # save best model
                    if epoch_total_val_loss < best_val_loss:
                        best_epoch = epoch
                        best_val_loss = epoch_total_val_loss
                        # best_strain_model = copy.deepcopy(strain_model)
                        # best_LMA_model = copy.deepcopy(LMA_model)
                        for model_name, model in models.items():
                            best_models[model_name] = copy.deepcopy(model)
                        best_epoch_loss_dict = copy.deepcopy(epoch_loss_dict)
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        print(f'Epochs without improvement: {epochs_without_improvement} / {epochs_without_improvement_tolerance}')

                    if use_wandb and enable_wandb_upload:
                        best_epoch_loss_dict_wandb = {}
                        for key, value in best_epoch_loss_dict.items():
                            # append 'best-' to the second level of the key
                            best_key = '/'.join(key.split('/')[:1] + ['best-' + key.split('/')[1]])
                            best_epoch_loss_dict_wandb[best_key] = value
                        best_epoch_loss_dict_wandb['best_epoch'] = best_epoch
                        wandb_experiment.log(best_epoch_loss_dict_wandb, step=epoch)

                    # early stopping
                    if epochs_without_improvement >= epochs_without_improvement_tolerance:
                        print(f'Early stopping at epoch {epoch}')
                        break
                # save the current best model regularly
                if save_best_model_interval > 0 and (epoch+0) % save_best_model_interval == 0:
                    # save the best model
                    print(f'Saving the best model at epoch {epoch}')
                    for model_name, model in best_models.items():
                        # torch.save(model.state_dict(), Path(exp_save_dir) / f'epoch_{epoch}_best_{model_name}_model.pth')
                        torch.save(model.state_dict(), Path(exp_save_dir) / f'{model_name}_best_checkpoint.pth')
                    # torch.save(best_strain_model.state_dict(), exp_save_dir / 'best_strain_model.pth')
                    # torch.save(best_LMA_model.state_dict(), exp_save_dir / 'best_LMA_model.pth')
            except KeyboardInterrupt:
                print(f'KeyboardInterrupt at epoch {epoch}')
                break
        
        if use_tensorboard:
            writer.close()

        if early_stop:
            # strain_model = best_strain_model
            # LMA_model = best_LMA_model
            for model_name, model in best_models.items():
                models[model_name] = model
            epoch_loss_dict = best_epoch_loss_dict
        
        self.epoch_loss_dict = epoch_loss_dict
        # convert the data in epoch_loss_dict to json serializable format
        import numpy as np
        for key, value in self.epoch_loss_dict.items():
            if isinstance(value, np.ndarray):
                self.epoch_loss_dict[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                self.epoch_loss_dict[key] = value.item()

        exp_dict = {
            'epoch': epoch,
            'epoch_loss_dict': self.epoch_loss_dict,
            'best_epoch': best_epoch,
            'epoch_loss_dict_list': self.training_epoch_loss_dict_list,            
        }
        # exp_dict.update(models)
        trained_models = models

        return exp_dict, trained_models, wandb_experiment    
    
    # function inference: alias to test
    def inference(self, model, datasets, trainer_config=None, full_config=None, device=None):
        return self.test(model, datasets, trainer_config, full_config, device)

    def test(self, models, datasets, trainer_config=None, full_config=None, device=None, wandb_experiment=None, target_dataset='test',mode='test',collate_fn=None):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # task-related parameters
        self.LMA_modality = used_train_config.get('LMA_modality', 'myocardium_mask')
        self.LMA_task = used_train_config.get('LMA_task', 'TOS_regression')

        # unpack models
        LMA_task = used_train_config.get('LMA_task', 'TOS_regression')
        # strain_model = models['strain_model']
        # LMA_model = models['LMA_model']


        # unpack datasets
        test_dataset = datasets[target_dataset]
        # Build dataloaders
        batch_size = used_train_config['batch_size']
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        # loss calculator
        # create loss caluator if not exists
        if not hasattr(self, 'loss_calculator'):
            self.loss_calculator = LossCalculator(used_full_config['losses'])
        if not hasattr(self, 'evaluator'):
            self.evaluator = LossCalculator(used_full_config.get('evaluation', {}))

        test_preds = []
        # test_performance_dict = {}
        test_loss_dict = {}
        # test_total_error_dict = {}
        with torch.no_grad():
            # for model_name, model in models.items():
            #     model.eval()
            self.models_mode_switch(models, 'eval')
            for test_batch_idx, test_batch in enumerate(test_dataloader):
                # if test_batch_idx == 1:
                #     break
                # forward pass
                test_loss, test_batch_loss_dict, test_batch_error_values_dict, test_batch_pred_dict, test_batch_target_dict = self.batch_forward(test_batch, models, loss_name_prefix=target_dataset,mode=mode, train_config=used_train_config, full_config=used_full_config)
                # return test_loss, test_batch_loss_dict, test_batch_error_values_dict, test_batch_pred_dict, test_batch_target_dict
                # batch_eval_dict = self.batch_eval(test_batch, test_batch)
                
                # break the batch into individual images and append to test_preds
                if 'src' in test_batch.keys():
                    curr_batch_size = test_batch['src'].shape[0]
                else:
                    curr_batch_size = test_batch['vol'].shape[0]
                
                for i in range(curr_batch_size):
                    total_datum_idx = test_batch_idx * batch_size + i
                    # if total_datum_idx == 50:
                    #     debug = True
                    test_pred_dict = {}
                    # copy all key-value from batch to test_batch_pred_dict and test_batch if
                    # (1) the value is not a torch.Tensor or np.ndarray, or
                    # (2) the value is a torch.Tensor or np.ndarray and the shape of the value is the same as the shape of the batch
                    for k, v in test_batch_pred_dict.items():
                        if v is None:
                            test_pred_dict[k+'_pred'] = None
                        elif not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k+'_pred'] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == test_batch_pred_dict[k].shape:
                            test_pred_dict[k+'_pred'] = v[i].cpu().numpy()
                    
                    for k, v in test_batch.items():
                        if not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == test_batch[k].shape:
                            test_pred_dict[k] = v[i].cpu().numpy()

                    for k, v in test_batch_target_dict.items():
                        if k in test_batch.keys():
                            pass
                        else:
                            if not isinstance(v, (torch.Tensor, np.ndarray)):
                                test_pred_dict[k] = v[i]
                            elif isinstance(v, torch.Tensor) and v.shape == test_batch_target_dict[k].shape:
                                test_pred_dict[k] = v[i].cpu().numpy()
                                
                    
                    # Add the test_pred_dict to test_preds
                    test_preds.append(test_pred_dict)
                    # if i == 0:
                    #     break
                    # if total_datum_idx == 50:
                    #     return test_pred_dict, test_batch_pred_dict, target_dict, test_preds
                
                # update the test_loss_dict by adding the loss values
                for loss_name, loss_value in test_batch_loss_dict.items():
                    if loss_name in test_loss_dict.keys():
                        test_loss_dict[loss_name] += loss_value
                    else:
                        test_loss_dict[loss_name] = loss_value
                
                # update test_total_error_dict based on test_batch_error_values_dict
                for error_name, error_value in test_batch_error_values_dict.items():
                    if error_name in test_loss_dict.keys():
                        test_loss_dict[error_name] += error_value
                    else:
                        test_loss_dict[error_name] = error_value

        # update the test_loss_dict by averaging the loss values if the loss value is a torch.Tensor
        for loss_name, loss_value in test_loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                test_loss_dict[loss_name] = loss_value.item() / len(test_dataloader)
            else:
                test_loss_dict[loss_name] = loss_value / len(test_dataloader)
                
        if wandb_experiment is not None:
            # wandb_experiment.log(test_loss_dict)
            # also upload test_performance_dict, but remember to add f'final-{target_dataset}/' to each key
            test_loss_dict_wandb = {}
            # update losses
            for key, value in test_loss_dict.items():
                # append f'final-{target_dataset}/' to the second level of the key
                final_key = f'final-' + key
                test_loss_dict_wandb[final_key] = value
            
            # for key, value in test_total_error_dict.items():
            #     # append f'final-{target_dataset}/' to the second level of the key
            #     final_key = f'final-error-' + key
            #     test_loss_dict_wandb[final_key] = value
            # update evaluation metrics
            # for key, value in test_performance_dict.items():
            #     # append f'final-{target_dataset}/' to the second level of the key
            #     final_key = f'final-' + key
            #     test_loss_dict_wandb[final_key] = value
            wandb_experiment.log(test_loss_dict_wandb)
            print('test_loss_dict_wandb: ', test_loss_dict_wandb)
        # print('inference_performance_dict: ', test_performance_dict)        
        print('test_loss_dict: ', test_loss_dict)
        # print('test_total_error_dict: ', test_total_error_dict)
        return test_preds, test_loss_dict, wandb_experiment

    def visualize_pred_regression(self, preds, n_vis=5, vis_indices=None, save_plots=False, save_dir=None, save_name=''):
        # plot the strain matrices and TOS curves
        if vis_indices is not None:
            n_vis = len(vis_indices)
        else:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            axs[plot_idx].pcolor(preds[vis_test_idx]['strain_mat'], cmap='jet', vmin=-0.3, vmax=0.3)
            axs[plot_idx].plot(preds[vis_test_idx]['TOS']/17+1, np.arange(126), color='black')
            axs[plot_idx].plot(preds[vis_test_idx]['TOS_pred']/17+1, np.arange(126), color='red', linestyle='--')
            axs[plot_idx].set_xlim([0, 50])
        if save_plots:
            if save_dir is None:
                save_dir = Path(self.full_config['others']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / save_name)
        return fig, axs

    def merge_data_by_patient(self, data_list):
        return data_list
    
    def plot_results(self, data, **kwargs):
        pass