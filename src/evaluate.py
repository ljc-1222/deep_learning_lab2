import torch
import numpy as np
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)


@torch.no_grad()
def dice_sum_batch(preds: torch.Tensor, target: torch.Tensor, soft: bool = True) -> float:

    prob = torch.sigmoid(preds).float()
    if soft:
        pred_use = prob
    else:
        pred_use = (prob > 0.5).float()

    total = 0.0
    for i in range(pred_use.shape[0]):
        total += float(dice_score(pred_use[i].squeeze(0), target[i].squeeze(0)))

    return total


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return torch.log(torch.cosh(1.0 - dice_score(pred, target)))

def lovasz_softmax_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return smp.losses.LovaszLoss(mode = "binary", ignore_index = None)(preds, target)

def combined_loss(preds: torch.Tensor, target: torch.Tensor, weight0: float = 1.0, weight1: float = 1.0, weight2: float = 1.0) -> torch.Tensor:

    bce  = F.binary_cross_entropy_with_logits(preds, target)
    prob = torch.sigmoid(preds)
    dice = dice_loss(prob, target)
    lovasz = lovasz_softmax_loss(prob, target)
    
    return weight0 * bce + weight1 * dice + weight2 * lovasz
