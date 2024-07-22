# class lossCalculator
# Description: This class is used to calculate the loss of the model
import torch
import copy
from modules.loss.registration_losses import RegistrationReconstructionLoss, RegistrationReconstructionLossSVF
from modules.loss.losses import CrossEntropyLoss, MSELoss, GradientMagnitudeLoss, VGGPerceptualLoss, EPELoss, NormDifferenceLoss, L1Loss, SSIMLoss
from modules.loss.bio_info_motion_tracking_losses import MotionVAELoss
from modules.loss.vae_losses import KDELoss
# import json
import pprint

def get_loss_function(loss_conf, full_config=None):
    if loss_conf['criterion'] in ['cross_entropy', 'CrossEntropyLoss']:
        # return torch.nn.CrossEntropyLoss()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # class_weights_list = loss_conf.get('class_weights', [1, 1])
        # class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
        
        # print('cross entropy class_weights: {}'.format(class_weights))
        # return CrossEntropyLoss(loss_conf, weight=class_weights)
        return CrossEntropyLoss(loss_conf)
    elif loss_conf['criterion'] in ['mse', 'MSELoss', "MSE"]:
        # return torch.nn.MSELoss()
        return MSELoss(loss_conf)
    elif loss_conf['criterion'] in ['registration_reconstruction']:
        return RegistrationReconstructionLoss(sigma=loss_conf['sigma'], regularization_weight=loss_conf['regularization_weight'])
    elif loss_conf['criterion'] in ['registration_reconstruction_SVF']:
        return RegistrationReconstructionLossSVF(
            sim_penalty=loss_conf.get('sim_penalty', 'l2'),
            sim_weight=loss_conf.get('sim_weight', 1),
            reg_penalty=loss_conf.get('reg_penalty', 'l1'),
            reg_weight=loss_conf.get('reg_weight', 1),
            reg_multi=loss_conf.get('reg_multi', None)
            )
    elif loss_conf['criterion'] in ['gradient_magnitude']:
        return GradientMagnitudeLoss(loss_conf)
    elif loss_conf['criterion'] in ['vgg_perceptual']:
        return VGGPerceptualLoss(loss_conf)
    elif loss_conf['criterion'].lower() in ['epe']:
        return EPELoss(loss_conf)
    elif loss_conf['criterion'] in ['norm_difference', 'NormDifferenceLoss']:
        return NormDifferenceLoss(loss_conf)
    elif loss_conf['criterion'] in ['l1', 'L1Loss']:
        return L1Loss(loss_conf)
    elif loss_conf['criterion'] in ['ssim', 'SSIMLoss']:
        return SSIMLoss(loss_conf)
    elif loss_conf['criterion'] in ['motion_vae']:
        return MotionVAELoss(loss_conf)
    elif loss_conf['criterion'] in ['kde', 'KDE', 'KDELoss']:
        return KDELoss(loss_conf)
    else:
        raise NotImplementedError("Loss function {} not implemented".format(loss_conf['criterion']))

class LossCalculator:
    def __init__(self, losses_confs: dict, full_config: dict = None, device=None):
        self.losses = copy.deepcopy(losses_confs)
        self.full_config = copy.deepcopy(full_config)
        self.device = device if device is not None else torch.device('cpu')

        # self.losses_functions = {}
        for loss_name, loss_conf in self.losses.items():
            self.losses[loss_name]['function'] = get_loss_function(loss_conf, self.full_config)
            if loss_conf['weight'] < 0 or self.losses[loss_name]['enable'] is False:
                self.losses[loss_name]['enable'] = False
                print('Loss {} is disabled'.format(loss_name))
    
    def __str__(self):
        return str(pprint.pformat(self.losses))

    def __call__(self, outputs, targets):
        total_loss = 0
        losses_values = {}
        for loss_name, loss_conf in self.losses.items():
            if loss_conf['enable'] is False:
                continue
            # prediction = outputs[loss_conf['prediction']]
            # target = targets[loss_conf['target']]
            loss = loss_conf['function'](outputs, targets)
            losses_values[loss_name] = loss.item()
            total_loss += loss_conf['weight'] * loss
        losses_values['total_loss'] = total_loss.item()
        return total_loss, losses_values