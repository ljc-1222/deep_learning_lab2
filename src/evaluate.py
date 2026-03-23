import torch
import torch.nn.functional as F
from kornia.losses import lovasz_softmax_loss

def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, gamma: float = 2) -> torch.Tensor:
    
    log_p = F.log_softmax(pred_logits, dim=1)
    p = log_p.exp()
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
    ce = F.nll_loss(log_p, target, reduction="none")
    
    return (((1.0 - p_t) ** gamma) * ce).mean()


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    return 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    pred_prob = torch.softmax(pred_logits, dim=1)[:, 1]
    target_f = target.float()
    numerator = 2.0 * (pred_prob * target_f).sum()
    denominator = pred_prob.sum() + target_f.sum() + 1e-8
    
    return 1.0 - numerator / denominator


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2) -> torch.Tensor:

    fl = focal_loss(pred_logits, target, gamma = gamma)
    dice = dice_loss(pred_logits, target)

    return fl + alpha * dice


def combined_loss_lovasz(pred_logits: torch.Tensor, target: torch.Tensor, beta: float = 1.0, gamma: float = 2) -> torch.Tensor:

    probas = torch.softmax(pred_logits, dim=1)
    lov = lovasz_softmax_loss(probas, target)
    fl = focal_loss(pred_logits, target, gamma = gamma)

    return fl + beta * lov
