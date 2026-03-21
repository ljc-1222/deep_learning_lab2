import torch
import torch.nn.functional as F


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    pred_prob   = torch.softmax(pred_logits, dim=1)[:, 1]
    target_f    = target.float()
    numerator   = 2.0 * (pred_prob * target_f).sum()
    denominator = pred_prob.sum() + target_f.sum() + 1e-8
    
    return 1.0 - numerator / denominator


def tv_loss(pred_logits: torch.Tensor) -> torch.Tensor:
    
    pred_prob = torch.softmax(pred_logits, dim=1)[:, 1]
    diff_h    = (pred_prob[:, 1:, :] - pred_prob[:, :-1, :]).abs()
    diff_w    = (pred_prob[:, :, 1:] - pred_prob[:, :, :-1]).abs()
    
    return diff_h.mean() + diff_w.mean()


def combined_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, beta: float  = 0.1) -> torch.Tensor:

    ce   = F.cross_entropy(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    tv   = tv_loss(pred_logits)
    
    return ce + alpha * dice + beta * tv
