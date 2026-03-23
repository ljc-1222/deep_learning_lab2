import os
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm import tqdm
from typing import Callable, Optional


def print_training_config(
    run_timestamp: str,
    p1_best_model_path: str,
    p2_best_model_path: str,
    device: str,
    model_name: str,
    phase1_epochs: int,
    phase2_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dice_alpha: float,
    lovasz_beta: float,
    p1_warmup_epochs: int,
    p2_warmup_epochs: int,
    p1_warmup_start_factor: float,
    p2_warmup_start_factor: float,
    p1_cosine_eta_min_factor: float,
    p2_cosine_eta_min_factor: float,
    patience_phase1: int,
    p2_lr_divisor: int,
    patience_phase2: int,
    min_width: int = 72,
) -> None:
    p2_lr = learning_rate / p2_lr_divisor

    LABEL_W = 22

    def fmt_sched(warmup_ep: int, start_f: float, eta_f: float) -> str:
        return (
            f"LinearLR warmup {warmup_ep} ep "
            f"({start_f}x → 1.0)  +  CosineAnnealingLR ({eta_f} min factor)"
        )

    header_rows: list[tuple[str, object]] = [
        ("Start Time",   run_timestamp),
        ("Device",       device),
        ("Model",        model_name),
        ("Phase 1 Save", p1_best_model_path),
        ("Phase 2 Save", p2_best_model_path),
    ]
    p1_rows: list[tuple[str, object]] = [
        ("Epochs (max)",        phase1_epochs),
        ("Batch Size",          batch_size),
        ("Learning Rate",       learning_rate),
        ("Weight Decay",        weight_decay),
        ("Loss",                f"Focal(γ=0.0)  +  {dice_alpha}× Dice"),
        ("Scheduler",           fmt_sched(p1_warmup_epochs, p1_warmup_start_factor, p1_cosine_eta_min_factor)),
        ("Early Stop Patience", patience_phase1),
    ]
    p2_rows: list[tuple[str, object]] = [
        ("Epochs (max)",        phase2_epochs),
        ("Batch Size",          batch_size),
        ("Learning Rate",       p2_lr),
        ("Weight Decay",        weight_decay / 20),
        ("Loss",                f"Focal(γ=0.0)  +  {lovasz_beta}× Lovász"),
        ("Scheduler",           fmt_sched(p2_warmup_epochs, p2_warmup_start_factor, p2_cosine_eta_min_factor)),
        ("Early Stop Patience", patience_phase2),
    ]

    # Auto-fit width so no value ever wraps
    all_rows = header_rows + p1_rows + p2_rows
    inner_w  = max(min_width, max(LABEL_W + 4 + len(str(v)) + 2 for _, v in all_rows))

    def hline(left: str = "╠", fill: str = "═", right: str = "╣") -> None:
        tqdm.write(left + fill * inner_w + right)

    def section(title: str) -> None:
        hline("╠", "─", "╣")
        tqdm.write("║" + title.center(inner_w) + "║")
        hline("╠", "─", "╣")

    def row(label: str, value: object) -> str:
        body = f"  {label:<{LABEL_W}}  {value}"
        return f"║{body:<{inner_w}}║"

    def blank() -> None:
        tqdm.write("║" + " " * inner_w + "║")

    hline("╔", "═", "╗")
    tqdm.write("║" + " ✦  Training Configuration  ✦ ".center(inner_w) + "║")
    hline("╠", "═", "╣")
    blank()
    for label, value in header_rows:
        tqdm.write(row(label, value))
    blank()

    section("  Phase 1  —  Focal + Dice  ")
    blank()
    for label, value in p1_rows:
        tqdm.write(row(label, value))
    blank()

    section("  Phase 2  —  Focal + Lovász  ")
    blank()
    for label, value in p2_rows:
        tqdm.write(row(label, value))
    blank()

    hline("╚", "═", "╝")



def plot_training_curves(
    train_losses:    list[float],
    val_losses:      list[float],
    val_dice_scores: list[float],
    save_dir:        str = ".",
) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, val_losses,   color="tomato",    linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curves.png")
    fig1.savefig(loss_path, dpi=150)
    print(f"Saved → {loss_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, val_dice_scores, color="seagreen", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    dice_path = os.path.join(save_dir, "dice_scores.png")
    fig2.savefig(dice_path, dpi=150)
    print(f"Saved → {dice_path}")

    plt.show()


def rle_encode(mask: np.ndarray) -> str:
    """RLE encode a binary mask in column-major (Fortran) order, 1-indexed."""
    flat = mask.flatten(order="F").astype(bool)
    padded = np.concatenate([[False], flat, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts  = np.where(diffs ==  1)[0] + 1
    lengths = np.where(diffs == -1)[0] + 1 - starts
    if len(starts) == 0:
        return ""
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def plot_sample(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    image_title: str = "Image",
    mask_title: str = "Mask",
    save_path: Optional[str] = None,
) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].imshow(image, cmap = "gray")
    axes[0].set_title(image_title)
    axes[0].axis("off")
    axes[1].imshow(mask, cmap = "viridis", vmin = 0, vmax = 1)
    axes[1].set_title(mask_title)
    axes[1].axis("off")
    
    if title:
        fig.suptitle(title)
        
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi = 150)
        
    plt.show()


def prepare_five_crops(
    pil_image: Image.Image,
    crop_size: int = 388,
    pad_size: int = 92,
) -> tuple[torch.Tensor, list[tuple[int, int]], int, int, int, int, int, int]:
    
    """Return a [5, 3, 572, 572] batch and stitching metadata for one image.

    Returns ``(batch, positions, H_p, W_p, pad_top, pad_left, H_orig, W_orig)``.
    Pass all return values directly to :func:`stitch_five_crop_results`.
    """
    img_t = TF.to_tensor(pil_image.convert("RGB"))
    img_t = TF.normalize(img_t, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    H_orig, W_orig = img_t.shape[1], img_t.shape[2]

    # TF.five_crop raises if image is smaller than crop_size
    pad_h = max(0, crop_size - H_orig)
    pad_w = max(0, crop_size - W_orig)
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        img_t = TF.pad(img_t, [pad_left, pad_top, pad_right, pad_bottom], padding_mode="constant")

    H_p, W_p = img_t.shape[1], img_t.shape[2]
    tl, tr, bl, br, center = TF.five_crop(img_t, [crop_size, crop_size])

    # Mirrors five_crop / center_crop position arithmetic exactly
    c_top  = int(round((H_p - crop_size) / 2.0))
    c_left = int(round((W_p - crop_size) / 2.0))
    positions: list[tuple[int, int]] = [
        (0, 0),                              # TL
        (0, W_p - crop_size),                # TR
        (H_p - crop_size, 0),                # BL
        (H_p - crop_size, W_p - crop_size),  # BR
        (c_top, c_left),                     # Center
    ]

    batch = torch.stack([TF.pad(crop, pad_size, padding_mode = "constant") for crop in (tl, tr, bl, br, center)], dim = 0)
    
    return batch, positions, H_p, W_p, pad_top, pad_left, H_orig, W_orig


def stitch_five_crop_results(
    crop_maps: torch.Tensor,
    positions: list[tuple[int, int]],
    H_p: int,
    W_p: int,
    pad_top: int,
    pad_left: int,
    H_orig: int,
    W_orig: int,
    crop_size: int = 388,
) -> torch.Tensor:
    
    """Accumulate 5 crop maps onto a canvas and return the original-resolution result."""
    C = crop_maps.shape[1]
    accum = torch.zeros(C, H_p, W_p, dtype=crop_maps.dtype)
    count = torch.zeros(H_p, W_p, dtype=crop_maps.dtype)

    for prob, (top, left) in zip(crop_maps, positions):
        accum[:, top:top + crop_size, left:left + crop_size] += prob
        count[top:top + crop_size, left:left + crop_size]    += 1.0

    result = accum / count.unsqueeze(0).clamp(min = 1.0)
    
    return result[:, pad_top:pad_top + H_orig, pad_left:pad_left + W_orig]


def predict_full_image(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    img_normalized: torch.Tensor,
    crop_size: int = 388,
    pad_size: int = 92,
) -> torch.Tensor:
    """Full-image inference: resize → pad → forward → upsample back to original size.

    The normalized image is resized to ``crop_size × crop_size``, padded by
    ``pad_size`` on all sides (producing a ``(crop_size + 2*pad_size)²`` model
    input), passed through ``forward_fn``, and the result is bilinearly
    upsampled back to the original spatial resolution.

    ``forward_fn`` receives a ``[1, 3, H_in, W_in]`` batch and must return a
    ``[1, 2, crop_size, crop_size]`` softmax-probability tensor.  Device
    placement is the caller's responsibility (move the batch inside
    ``forward_fn``).

    Args:
        forward_fn: Callable that takes a CPU batch and returns probabilities
            (e.g. ``lambda b: tta_predict(model, b.to(device))`` or
            ``lambda b: torch.softmax(model(b.to(device)), dim=1)``).
        img_normalized: ImageNet-normalised image tensor of shape
            ``[3, H_orig, W_orig]``.
        crop_size: Spatial size to rescale the image to before padding.
        pad_size: Padding added on each side to reach the full model input.

    Returns:
        Softmax probability tensor of shape ``[2, H_orig, W_orig]`` on CPU.
    """
    H_orig, W_orig = img_normalized.shape[1], img_normalized.shape[2]

    img_resized = TF.resize(img_normalized, [crop_size, crop_size], antialias=True)
    img_padded  = TF.pad(img_resized, pad_size, padding_mode="constant")
    batch       = img_padded.unsqueeze(0)  # [1, 3, crop+2*pad, crop+2*pad]

    probs = forward_fn(batch).cpu()  # [1, 2, crop_size, crop_size]
    return TF.resize(probs.squeeze(0), [H_orig, W_orig], antialias=True)  # [2, H_orig, W_orig]
