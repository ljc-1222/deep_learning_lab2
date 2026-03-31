import torch
import torch.nn.functional as F


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

    return 1.0 - dice_score(pred, target)


def combined_loss(preds: torch.Tensor, target: torch.Tensor, weight: float = 0.5) -> torch.Tensor:

    bce  = F.binary_cross_entropy_with_logits(preds, target)
    prob = torch.sigmoid(preds)
    dice = dice_loss(prob, target)

    return weight * bce + (1 - weight) * dice
