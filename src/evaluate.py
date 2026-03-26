import torch
import torch.nn as nn


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)


@torch.no_grad()
def dice_sum_batch(preds: torch.Tensor, target: torch.Tensor) -> float:

    pred_bin = (torch.sigmoid(preds)).float()
    total = 0.0
    
    for i in range(pred_bin.shape[0]):
        total += float(dice_score(pred_bin[i].squeeze(0), target[i].squeeze(0)))
        
    return total


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return 1.0 - dice_score(pred, target)

def combined_loss(preds: torch.Tensor, target: torch.Tensor, weight: float = 0.5) -> torch.Tensor:

    preds = torch.sigmoid(preds)
    bce   = nn.BCELoss()(preds, target)
    dice  = dice_loss(preds, target)
        
    return weight * bce + (1 - weight) * dice
