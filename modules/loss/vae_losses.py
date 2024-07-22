import torch
# for reference:
# class RegistrationReconstructionLoss:
#     def __init__(self, sigma, regularization_weight=1):
#         self.sigma = sigma
#         self.regularization_weight = regularization_weight

#     def __call__(self, prediction, target):
#         Sdef = prediction['deformed_source']
#         tar = target['registration_target']
#         recon_loss = torch.nn.MSELoss()(tar, Sdef)
#         regularization = (prediction['velocity']*prediction['momentum']).sum() / (tar.numel())
#         loss = 0.5 * recon_loss/(self.sigma*self.sigma) + regularization * self.regularization_weight
#         return loss
    
class KDELoss:
    def __init__(self, config, **kwargs):
        self.log_var_key = config.get('log_var', 'log_var')
        self.mu_key = config.get('mu', 'mu')
        # self.ndims = config.get('ndims', 3)
        # self.sum_dims = tuple(range(1, self.ndims + 1))
    
    def __call__(self, prediction, target):
        log_var = prediction[self.log_var_key].flatten(1)
        mu = prediction[self.mu_key].flatten(1)
        
        # kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss        