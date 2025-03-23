import wandb
import torch
import numpy as np
import datetime, json, copy
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from modules.loss import LossCalculator
from modules.trainer.base_trainer import BaseTrainer
import random

class DummyLrScheduler:
    """
    Dummy learning rate scheduler that does nothing
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass
    
from modules.data.processing.rotation_module import ImageSeqRotator, DisplacementFieldSeqRotator
class RegTrainer(BaseTrainer):
    def __init__(self, trainer_config, device=None, full_config=None):        
        super().__init__(trainer_config, device=device, full_config=full_config)        
        self.init_rotation_modules(trainer_config, device)

    def init_rotation_modules(self, trainer_config, device):
        # Pre-compute the rotation utils 
        self.rotation_angles = trainer_config.get('rotation_angles', [0, np.pi/2, np.pi, -np.pi/2])
        self.img_seq_rotators = []
        self.disp_seq_rotators = []
        for theta in self.rotation_angles:
            img_seq_rotator = ImageSeqRotator(theta, device, torch.float32, torch.Size((2, 1, 49, 128, 128)))
            disp_seq_rotator = DisplacementFieldSeqRotator(theta, device, torch.float32, torch.Size((2, 2, 49, 48, 48)))
            self.img_seq_rotators.append(img_seq_rotator)
            self.disp_seq_rotators.append(disp_seq_rotator)
    
    # def get_optimizers(self, models_dict, optimization_config):
    #     optimizers_config = optimization_config['registration']
    #     optimizers = {}
    #     for model_name, model in models_dict.items():
    #         optimizers[model_name] = torch.optim.Adam(model.parameters(), lr=optimizers_config['learning_rate'])
    #     return optimizers
    
    # def get_lr_scheduler(self, optimizer, lr_scheduler_config):
    #     # print(f'{optimization_config=}')
    #     # lr_scheduler_config = optimization_config['registration']['lr_scheduler']
    #     lr_scheduler_type = lr_scheduler_config['type']
    #     lr_scheduler_enabled = lr_scheduler_config['enable']
    #     if not lr_scheduler_enabled:
    #         lr_scheduler = DummyLrScheduler(optimizer)
    #     elif lr_scheduler_type in ['CosineAnnealingLR']:
    #         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer, 
    #             T_max=lr_scheduler_config['T_max'])
    #     else:
    #         raise NotImplementedError(f'Learning rate scheduler {lr_scheduler_type} not implemented')
    #     return lr_scheduler

    def batch_forward(self, batch, models, loss_name_prefix, mode='train', train_config={}, full_config={}, curr_epoch=-1):
        # reg_model = models[0]
        reg_model = models['registration']

        src, tar = batch['src'], batch['tar']
        src = src.to(self.device, dtype=torch.float32)
        tar = tar.to(self.device, dtype=torch.float32)

        if train_config.get('enable_random_rotate', False) and mode in ['training', 'train']:
            random_rotate_prob_thres = train_config.get('random_rotate_prob_thres', 0.3)
            random_rotate_prob = np.random.uniform(0, 1)
            # print(f'random rotate prob: {random_rotate_prob}')
            if random_rotate_prob < random_rotate_prob_thres:
                pass
            else:
                # rotator_idx = 1
                rotator_idx = random.choice(range(len(self.rotation_angles)))                
                src = self.img_seq_rotators[rotator_idx](src)
                tar = self.img_seq_rotators[rotator_idx](tar)

        reg_pred_dict = reg_model(src, tar)
        
        # DENSE_disp_GT = batch['DENSE_disp']
        pred_dict = {
            'displacement': reg_pred_dict['displacement'],
            'velocity': reg_pred_dict['velocity'],
            'momentum': reg_pred_dict['momentum'],
            'deformed_source': reg_pred_dict['deformed_source'],
            'latent': reg_pred_dict['latent'],
        }
        target_dict = {
            'registration_target': tar,
            "src": src,
            'tar': tar,
            # 'DENSE_disp': batch['DENSE_disp']
        }        

        # self.loss_calculator.losses['registration_reconstruction']['regularization_weight'] = 5    
        total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)
        with torch.no_grad():
            total_error, error_values_dict = self.evaluator(pred_dict, target_dict)

        # update the epoch loss dict
        batch_loss_dict = {}
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            batch_loss_dict[record_name] = loss_value

        # update the epoch error dict
        batch_error_dict = {}
        for error_name, error_value in error_values_dict.items():
            record_name = f'{loss_name_prefix}/{error_name}'
            batch_error_dict[record_name] = error_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, batch_loss_dict, batch_error_dict, pred_dict, target_dict
    
    def batch_eval(self, batch_pred, batch):
        return {}