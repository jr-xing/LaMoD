import torch
import torch.nn.functional as F

def MotionVAELossFn(recon_x, x, mu, logvar, beta=1e-2):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD


class MotionVAELoss:
    def __init__(self, config, **kwargs):
        self.beta = config.get('beta', 1e-2)
        self.loss_function = MotionVAELossFn
        self.prediction = config['prediction']
        self.target = config['target']
        
    def __call__(self, outputs, targets):
        loss = self.loss_function(outputs[self.prediction], targets[self.target], outputs['mu'], outputs['logvar'], beta=self.beta)
        return loss