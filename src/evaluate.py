import torch

def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    
    return 2 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)