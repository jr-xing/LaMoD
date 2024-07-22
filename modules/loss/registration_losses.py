import torch

class RegistrationReconstructionLoss:
    def __init__(self, sigma, regularization_weight=1):
        self.sigma = sigma
        self.regularization_weight = regularization_weight

    def __call__(self, prediction, target):
        Sdef = prediction['deformed_source']
        tar = target['registration_target']
        recon_loss = torch.nn.MSELoss()(tar, Sdef)
        regularization = (prediction['velocity']*prediction['momentum']).sum() / (tar.numel())
        loss = 0.5 * recon_loss/(self.sigma*self.sigma) + regularization * self.regularization_weight
        return loss


class RegistrationReconstructionLossSVF:
    def __init__(self, 
                 sim_penalty ='l2', sim_weight=1,
                 reg_penalty='l1', reg_weight=1, reg_multi=None):
        self.sim_penalty = sim_penalty
        self.sim_weight = sim_weight
        
        self.reg_penalty = reg_penalty
        self.reg_weight = reg_weight
        self.grad_reg_fn = Grad(penalty=self.reg_penalty, loss_mult=reg_multi)

    def __call__(self, prediction, target):
        Sdef = prediction['deformed_source']
        tar = target['registration_target']
        sim_loss = torch.nn.MSELoss()(tar, Sdef)
        reg_loss = self.grad_reg_fn(None, prediction['velocity'])
        loss = self.sim_weight * sim_loss + reg_loss * self.reg_weight
        return loss
        
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = 1 if loss_mult is None else loss_mult

    def loss(self, _, y_pred):
        if(len(y_pred.shape) == 5):
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

            # if self.loss_mult is not None:
            grad *= self.loss_mult
            
            return grad
        elif(len(y_pred.shape) == 4):
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

            # if self.loss_mult is not None:
            grad *= self.loss_mult

            return grad
        
    def __call__(self, _, y_pred):
        return self.loss(_, y_pred)