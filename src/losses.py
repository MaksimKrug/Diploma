from contextlib import redirect_stderr
from turtle import forward
import segmentation_models_pytorch as smp
from torch import nn
import torch


def get_loss(loss_name):
    # BCE
    if loss_name == "BCE":
        loss = smp.losses.SoftBCEWithLogitsLoss()
    # Focal
    elif loss_name == "FocalLoss":
        loss = smp.losses.FocalLoss("binary")
    # Dice
    elif loss_name == "DiceLoss":
        loss = smp.losses.DiceLoss("binary")
    # TverskyLoss
    elif loss_name == "TverskyLoss":
        loss= smp.losses.TverskyLoss("binary")
    # DiceFocal
    elif loss_name == "DiceFocal":
        loss = DiceFocal()
    # # BiasLoss
    # elif loss_name == "BiasLoss":
    #     loss = BiasLoss()

    return loss

class DiceFocal(nn.Module):
    def __init__(self):
        super(DiceFocal, self).__init__()
        self.dice_loss = smp.losses.DiceLoss("binary")
        self.focal_loss = smp.losses.FocalLoss("binary")

    def forward(self, output, target):
        dice_loss = self.dice_loss(output, target)
        focal_loss = self.focal_loss(output, target)
        loss = dice_loss + focal_loss

        return loss
        

class BiasLoss(nn.Module):
    """
    Paper: https://arxiv.org/abs/2107.11170
    """
    def __init__(self, alpha=0.3, beta=0.3, normalisation_mode="global"):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.norm_mode = normalisation_mode
        self.global_min = 100000

    def norm_global(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        if min < self.global_min:
            self.global_min = min
        normalised = (tensor - self.global_min) / (max - min)
        return normalised

    def norm_local(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        normalised = (tensor - min) / (max - min)

        return normalised

    def forward(self, features, output, target):
        features_copy = features.clone().detach()
        features_dp = features_copy.reshape(features_copy.shape[0], -1)

        features_dp = torch.var(features_dp, dim=1)
        if self.norm_mode == "global":
            variance_dp_normalised = self.norm_global(features_dp)
        else:
            variance_dp_normalised = self.norm_local(features_dp)

        weights = (
            (torch.exp(variance_dp_normalised * self.beta) - 1.0) / 1.0
        ) + self.alpha
        loss = weights * self.ce(output, target)

        loss = loss.mean()

        return loss
