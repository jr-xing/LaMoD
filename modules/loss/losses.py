import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchmetrics

class GradientMagnitudeLayer(nn.Module):
    def __init__(self, device='cuda'):
        super(GradientMagnitudeLayer, self).__init__()
        # Define Sobel filters for x and y direction
        self.sobel_x = nn.Parameter(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view((1, 1, 3, 3)), requires_grad=False).to(device)
        self.sobel_y = nn.Parameter(torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view((1, 1, 3, 3)), requires_grad=False).to(device)

    def forward(self, x):
        # Check input dimensions
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Expected input shape (N, 1, H, W)")

        # Compute gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude

class GradientMagnitudeLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super(GradientMagnitudeLoss, self).__init__()
        self.offset = config.get('offset', 0)
        self.prediction = config['prediction']
        self.target = config['target']
        self.gradient_magnitude_layer = GradientMagnitudeLayer()

    def forward(self, x, y):
        # y is not used
        # Check input dimensions
        x = x[self.prediction]
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Expected input shape (N, 1, H, W)")

        # Compute gradient magnitude
        grad_magnitude = self.gradient_magnitude_layer(x)

        # Calculate loss: Sum over each image's pixels, then mean over the batch
        loss_per_image = torch.abs(torch.sum(torch.abs(grad_magnitude), dim=[1, 2, 3]) - self.offset)  # Sum over H, W dimensions
        loss = torch.mean(loss_per_image)  # Mean over N (batch dimension)
        return loss

class MSELoss:
    """
    Custom MSE loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.loss_function = torch.nn.MSELoss(**kwargs)
        
    def __call__(self, outputs, targets):
        prediction = outputs[self.prediction]
        target = targets[self.target]
        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target)
        loss = self.loss_function(prediction, target)
        return loss

class L1Loss:
    """
    Custom L1 loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.loss_function = torch.nn.L1Loss(**kwargs)
        
    def __call__(self, outputs, targets):
        loss = self.loss_function(outputs[self.prediction], targets[self.target])
        return loss

from torchmetrics.image import StructuralSimilarityIndexMeasure
class SSIMLoss:
    """
    Custom SSIM loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.device = config.get('device', 'cuda')
        self.loss_function = StructuralSimilarityIndexMeasure(**kwargs).to(self.device)
        
    def __call__(self, outputs, targets):
        SSIM_value = self.loss_function(outputs[self.prediction], targets[self.target])
        return 1 - SSIM_value

class CrossEntropyLoss:
    """
    Custom CrossEntropy loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.loss_function = torch.nn.CrossEntropyLoss(**kwargs)
        
    def __call__(self, outputs, targets):
        loss = self.loss_function(outputs[self.prediction], targets[self.target])
        return loss
    
import torchvision
from torchvision.transforms import Normalize
from torchvision.models.vgg import VGG16_Weights
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, config):
        self.resize = config.get('resize', True)
        self.prediction = config['prediction']
        self.target = config['target']
        self.device = config.get('device', 'cuda')
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(self.device)
        self.transform = torch.nn.functional.interpolate
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # def normalize(self, data):

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = input[self.prediction]
        target = target[self.target]
        if input.shape[1] == 1:
            # if input has only one channel, repeat it 3 times
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        elif input.shape[1] == 2:
            # if input has only two channels, copy the second channel to the third channel
            input = torch.cat([input, input[:, 1:2, :, :]], dim=1)
            target = torch.cat([target, target[:, 1:2, :, :]], dim=1)
        elif input.shape[1] > 3:
            # if input has more than 3 channels, raise an error
            raise ValueError("Input has more than 3 channels")

        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        input = self.normalize(input)
        target = self.normalize(target)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    

class EPELoss:
    def __init__(self, config):
        # super(EPELoss, self).__init__()
        self.prediction = config['prediction']
        self.target = config['target']
        self.use_mask = config.get('use_mask', True)
        self.mask = config.get('mask', 'myo_union')
        # self.loss_function = torch.nn.L1Loss()
        
    def __call__(self, outputs, targets):
        output = outputs[self.prediction]
        target = targets[self.target]
        mask = targets[self.mask] if self.use_mask else 1
        mask_sum = torch.sum(mask) if self.use_mask else output.shape[0] * output.shape[2] * output.shape[3] * output.shape[4]
        # epe_tv = torch.sum(torch.sqrt(((output[:, 0, :, :, :] - target[:, 0, :, :, :])**2.0
        #                          + (output[:, 1, :, :, :] - target[:, 1, :, :, :])**2.0)) * mask[:, 0, :, :, :])\
        #    / torch.sum(mask[:, 0, :, :, :]) \
        #    + 0.000 * torch.sum(torch.abs(output[:, :, :, :, :-1] - output[:, :, :, :, 1:]) * mask[:, :, :, :, 1:])\
        #    / torch.sum(mask[:, :, :, :, :])
        # epe_loss = torch.sum(
        #     torch.sqrt((
        #         (output[:, 0, :, :] - target[:, 0, :, :])**2.0 +
        #         (output[:, 1, :, :] - target[:, 1, :, :])**2.0
        #     )) * mask
        # ) / mask_sum
        # epe_loss = torch.sum(
        #     torch.sqrt((
        #         (output[:, 0]*mask[:, 0] - target[:, 0]*mask[:, 0])**2.0 +
        #         (output[:, 1]*mask[:, 1] - target[:, 1]*mask[:, 1])**2.0
        #     )) * mask
        # ) / mask_sum
        
        epe_loss = torch.sum(torch.sqrt(((output[:, 0, :, :, :] - target[:, 0, :, :, :])**2.0
                                 + (output[:, 1, :, :, :] - target[:, 1, :, :, :])**2.0)) * mask[:, 0, :, :, :])\
           / torch.sum(mask[:, 0, :, :, :])
        # epe_loss = torch.sqrt(
        #     nn.functional.mse_loss(output[:, 0]*mask[:, 0], target[:, 0]*mask[:, 0], reduce='sum') + \
        #     nn.functional.mse_loss(output[:, 1]*mask[:, 1], target[:, 1]*mask[:, 1], reduce='sum')
        # ) / mask_sum
        # check the gradient of d epeloss / d output
        # print(epe_loss)
        # epe_loss.backward()
        # print(output.grad)
        return epe_loss

class NormDifferenceLoss:
    """Measure the difference between two tensors by their norms
    """
    def __init__(self, config):
        self.prediction = config['prediction']
        self.target = config['target']
        self.norm = config.get('norm', 2)
        self.loss_function = torch.nn.L1Loss()

    def __call__(self, outputs, targets):
        output = outputs[self.prediction]
        target = targets[self.target]
        loss = self.loss_function(torch.norm(output, self.norm), torch.norm(target, self.norm))
        return loss