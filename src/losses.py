import torch
from torch import nn

from utils import label_smoothing

σ = torch.sigmoid


class DiceLoss(nn.Module):
    def __init__(self, logit=True):
        super(DiceLoss, self).__init__()
        self.logit = logit

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        if self.logit:
            inputs = σ(inputs)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
    
class ComboLoss(nn.Module):
    def __init__(self, α=0., bce_weight=0.5):
        super(ComboLoss, self).__init__()
        self.dice = DiceLoss(logit=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.α = α
        self.bce_weight = bce_weight
        self.dice_weight = 1 - bce_weight
        
    def forward(self, inputs, targets, ϵ=1):
        loss_dice = self.dice(inputs, targets)
        
        inputs = label_smoothing(targets, self.α)
        loss_bce = self.bce(inputs.view(-1), targets.view(-1)) 
        
        assert not torch.isnan(loss_dice)
        assert not torch.isnan(loss_bce)
        
        loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return loss
