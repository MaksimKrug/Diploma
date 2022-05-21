import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def get_loss(loss_name, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
    # get loss
    loss = globals()[loss_name](weight, num_classes)

    return loss


class BiasLoss(nn.Module):
    # https://arxiv.org/pdf/2107.11170.pdf
    def __init__(
        self,
        weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        num_classes=11,
        alpha=0.3,
        beta=0.3,
        normalisation_mode="global",
    ):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.num_classes = num_classes
        self.ce = BCEWeighted(self.weight)
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

        features_dp = torch.var(features_dp, dim=0)
        if self.norm_mode == "global":
            variance_dp_normalised = self.norm_global(features_dp)
        else:
            variance_dp_normalised = self.norm_local(features_dp)

        weights = (
            (torch.exp(variance_dp_normalised * self.beta) - 1.0) / 1.0
        ) + self.alpha

        loss = weights.mean() * self.ce(output, target)

        return loss


class BCEWeighted(nn.Module):
    def __init__(self, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
        super(BCEWeighted, self).__init__()
        if torch.cuda.is_available():
            self.weight = torch.Tensor(weight).to("cuda")
        else:
            self.weight = torch.Tensor(weight)

        self.BCELoss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        val = self.BCELoss(inputs, targets)  # [B, C, H, W]
        val = (
            torch.permute(val, (1, 0, 2, 3)).reshape(self.num_classes, -1).mean(dim=1)
        )  # [C, -1]
        val = (val * self.weight).mean()

        return val


class DiceLoss(nn.Module):
    # https://arxiv.org/pdf/1802.05098.pdf
    def __init__(self, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
        super(DiceLoss, self).__init__()
        if torch.cuda.is_available():
            self.weight = torch.Tensor(weight).to("cuda")
        else:
            self.weight = torch.Tensor(weight)
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        # sigmoid
        inputs = F.sigmoid(inputs)

        # calculate intersection
        intersection = inputs * targets
        intersection = (
            torch.permute(intersection, (1, 0, 2, 3))
            .reshape(self.num_classes, -1)
            .sum(1)
        )

        # calculate dice
        inputs_sum = torch.sum(inputs, dim=(0, 2, 3))
        targets_sum = torch.sum(targets, dim=(0, 2, 3))

        dice = (2.0 * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        dice = 1 - dice

        return (dice * self.weight).mean()


class FocalLoss(nn.Module):
    # https://arxiv.org/pdf/1708.02002.pdf
    def __init__(self, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
        super(FocalLoss, self).__init__()
        if torch.cuda.is_available():
            self.weight = torch.Tensor(weight).to("cuda")
        else:
            self.weight = torch.Tensor(weight)
        self.num_classes = num_classes

    def forward(self, inputs, targets, alpha=0.8, gamma=2):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs
        targets = targets

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="none")
        BCE = torch.permute(BCE, (1, 0, 2, 3)).reshape(self.num_classes, -1).sum(1)

        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return (focal_loss * self.weight).mean()


class DiceFocal(nn.Module):
    def __init__(self, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
        super(DiceFocal, self).__init__()
        self.dice_loss = DiceLoss(weight, num_classes)
        self.focal_loss = FocalLoss(weight, num_classes)

    def forward(self, output, target):
        dice_loss = self.dice_loss(output, target)
        focal_loss = self.focal_loss(output, target)
        loss = dice_loss + focal_loss

        return loss


class DiceBCE(nn.Module):
    def __init__(self, weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=11):
        super(DiceBCE, self).__init__()
        self.dice_loss = DiceLoss(weight, num_classes)
        self.bce_loss = BCEWeighted(weight, num_classes)

    def forward(self, output, target):
        dice_loss = self.dice_loss(output, target)
        bce_loss = self.bce_loss(output, target)
        loss = dice_loss + bce_loss

        return loss


class TverskyLoss(nn.Module):
    # https://arxiv.org/pdf/1706.05721.pdf
    def __init__(
        self,
        weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        num_classes=11,
        alpha=0.5,
        beta=0.5,
    ):
        super(TverskyLoss, self).__init__()
        if torch.cuda.is_available():
            self.weight = torch.Tensor(weight).to("cuda")
        else:
            self.weight = torch.Tensor(weight)
        self.num_classes = num_classes
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # True Positives, False Positives & False Negatives
        TP = (
            torch.permute(inputs * targets, (1, 0, 2, 3))
            .reshape(self.num_classes, -1)
            .sum(1)
        )
        FP = (
            torch.permute(inputs * (1 - targets), (1, 0, 2, 3))
            .reshape(self.num_classes, -1)
            .sum(1)
        )
        FN = (
            torch.permute((1 - inputs) * targets, (1, 0, 2, 3))
            .reshape(self.num_classes, -1)
            .sum(1)
        )
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        Tversky = 1 - Tversky

        return (Tversky * self.weight).mean()

