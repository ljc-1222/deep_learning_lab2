import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-8)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    pred_prob = torch.softmax(pred_logits, dim = 1)[:, 1]
    target_f  = target.float()
    numerator   = 2.0 * (pred_prob * target_f).sum()
    denominator = pred_prob.sum() + target_f.sum() + 1e-8
    return 1.0 - numerator / denominator


def combined_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, ce_lambda: float = 0.5) -> torch.Tensor:

    ce   = torch.nn.functional.cross_entropy(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return ce_lambda * ce + (1.0 - ce_lambda) * dice


if __name__ == "__main__":
    
    print("=" * 50)
    print("dice_score / dice_loss unit tests")
    print("=" * 50)

    # ── dice_score ─────────────────────────────────────────────────────────

    pred   = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    score  = dice_score(pred, target).item()
    assert abs(score - 1.0) < 1e-5, f"Expected 1.0, got {score:.6f}"
    print(f"[PASS] dice_score — perfect:         {score:.4f}")

    pred   = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    target = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    score  = dice_score(pred, target).item()
    assert abs(score - 0.0) < 1e-5, f"Expected 0.0, got {score:.6f}"
    print(f"[PASS] dice_score — no overlap:      {score:.4f}")

    pred   = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    score  = dice_score(pred, target).item()
    assert abs(score - 0.5) < 1e-5, f"Expected 0.5, got {score:.6f}"
    print(f"[PASS] dice_score — partial overlap: {score:.4f}")

    # ── dice_loss ──────────────────────────────────────────────────────────

    B, C, H, W = 2, 2, 4, 4

    # Perfect foreground prediction → loss ≈ 0
    logits = torch.full((B, C, H, W), 0.0)
    logits[:, 1] = 10.0
    target_ones = torch.ones(B, H, W, dtype=torch.long)
    loss_val = dice_loss(logits, target_ones).item()
    assert abs(loss_val) < 1e-4, f"Expected ~0.0, got {loss_val:.6f}"
    print(f"[PASS] dice_loss  — perfect fg pred: {loss_val:.4f}")

    # All-background prediction against all-fg target → loss ≈ 1
    logits_bg = torch.full((B, C, H, W), 0.0)
    logits_bg[:, 0] = 10.0
    loss_val = dice_loss(logits_bg, target_ones).item()
    assert abs(loss_val - 1.0) < 1e-4, f"Expected ~1.0, got {loss_val:.6f}"
    print(f"[PASS] dice_loss  — all-bg pred:     {loss_val:.4f}")

    # Exact half overlap → loss ≈ 0
    # Both halves need explicit high logits so softmax gives near-0/1 probs;
    # zero logits → 0.5 fg prob everywhere, which inflates the denominator.
    logits_half = torch.zeros(B, C, H, W)
    logits_half[:, 1, :H//2, :] = 10.0   # top half:    predict fg
    logits_half[:, 0, H//2:, :] = 10.0   # bottom half: predict bg
    target_half = torch.zeros(B, H, W, dtype=torch.long)
    target_half[:, :H//2, :] = 1
    loss_val = dice_loss(logits_half, target_half).item()
    assert abs(loss_val) < 1e-4, f"Expected ~0.0, got {loss_val:.6f}"
    print(f"[PASS] dice_loss  — exact half:      {loss_val:.4f}")

    # Gradient must flow
    logits_grad = torch.randn(B, C, H, W, requires_grad=True)
    target_rand = torch.randint(0, 2, (B, H, W), dtype=torch.long)
    dice_loss(logits_grad, target_rand).backward()
    grad_norm = logits_grad.grad.norm().item()
    assert grad_norm > 0, "Gradient is zero — loss is not differentiable!"
    print(f"[PASS] dice_loss  — gradient norm:   {grad_norm:.4f}")

    print()
    print("All tests passed.")
