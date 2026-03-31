from __future__ import annotations

import csv
import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Union

from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.utils import rle_encode
from src.oxford_pet import OxfordPetDataset
from src.models.resnet34_unet import ResNet34UNet

# ---------------------------------------------------------------------
# Independent utility block (safe to delete later)
# ---------------------------------------------------------------------
def dice_scores_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:

    # Normalize shapes to (N, H, W)
    if probs.ndim == 2:
        probs_nhw = probs.unsqueeze(0)
    elif probs.ndim == 3:
        probs_nhw = probs
    else:
        # (N, 1, H, W) -> (N, H, W)
        if probs.shape[1] != 1:
            raise ValueError(f"Expected probs shape (N,1,H,W) for 4D input, got {tuple(probs.shape)}")
        probs_nhw = probs.squeeze(1)

    if targets.ndim == 2:
        targets_nhw = targets.unsqueeze(0)
    elif targets.ndim == 3:
        targets_nhw = targets
    else:
        if targets.shape[1] != 1:
            raise ValueError(
                f"Expected targets shape (N,1,H,W) for 4D input, got {tuple(targets.shape)}"
            )
        targets_nhw = targets.squeeze(1)

    if probs_nhw.shape != targets_nhw.shape:
        raise ValueError(
            f"probs and targets shapes must match after squeezing, got probs={tuple(probs_nhw.shape)} "
            f"targets={tuple(targets_nhw.shape)}"
        )

    probs_f = probs_nhw.to(dtype=torch.float32)
    targets_f = targets_nhw.to(dtype=torch.float32)

    reduce_dims = (1, 2)  # sum over H, W
    intersection_soft = (probs_f * targets_f).sum(dim=reduce_dims)
    union_soft = probs_f.sum(dim=reduce_dims) + targets_f.sum(dim=reduce_dims)
    soft_dice = (2.0 * intersection_soft + eps) / (union_soft + eps)

    preds_f = (probs_f >= float(threshold)).to(dtype=torch.float32)
    intersection_hard = (preds_f * targets_f).sum(dim=reduce_dims)
    union_hard = preds_f.sum(dim=reduce_dims) + targets_f.sum(dim=reduce_dims)
    hard_dice = (2.0 * intersection_hard + eps) / (union_hard + eps)

    return soft_dice, hard_dice

# MODEL_NAME = "UNet"
# TIMESTAMP = "20260327-110204"

MODEL_NAME = "ResNet34UNet"
TIMESTAMP = "20260327-144436"

BATCH_SIZE = 32
DATASET_ROOT = "dataset/oxford-iiit-pet"
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIZE_MAP = {
    "UNet": {
        "IMAGE_SIZE": (572, 572),
        "MASK_SIZE": (572 - 184, 572 - 184),
    },
    "ResNet34UNet": {
        "IMAGE_SIZE": (512, 512),
        "MASK_SIZE": (512, 512),
    }
}

if MODEL_NAME == "UNet":
    TEST_SPLIT = "test_unet.txt"
elif MODEL_NAME == "ResNet34UNet":
    TEST_SPLIT = "test_res_unet.txt"
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

def load_model(model_path: str, device: torch.device) -> Union[UNet, ResNet34UNet]:

    raw_state = torch.load(model_path, map_location = device, weights_only = True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw_state.items()}

    if MODEL_NAME == "UNet":
        model = UNet().to(device)
    elif MODEL_NAME == "ResNet34UNet":
        model = ResNet34UNet().to(device)
    else:
        raise ValueError(f"Invalid model name: {MODEL_NAME}")

    model.load_state_dict(state_dict)
    model.eval()
    return model

def resize_prob_to_original(prob: np.ndarray, height: int, width: int) -> np.ndarray:

    t = torch.from_numpy(prob.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size = (height, width), mode = "bilinear", align_corners = False)
    return out.squeeze(0).squeeze(0).numpy()

@torch.no_grad()
def collect_sigmoid_probs(
    model: Union[UNet, ResNet34UNet],
    dataset: OxfordPetDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> tuple[list[np.ndarray], list[int]]:

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = True
    )
    all_probs: list[np.ndarray] = []
    all_idx: list[int] = []
    
    for images, _, idxs in tqdm(loader, desc = desc, leave = False):
        
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        
        for i in range(probs.shape[0]):
            
            all_probs.append(probs[i])
            all_idx.append(int(idxs[i].item() if torch.is_tensor(idxs[i]) else idxs[i]))
            
    return all_probs, all_idx


def write_submission_csv(
    model: Union[UNet, ResNet34UNet],
    model_name: str,
    dataset_root: str,
    test_split: str,
    save_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    csv_name: str,
) -> str:

    os.makedirs(save_dir, exist_ok=True)
    test_ds = OxfordPetDataset(root       = dataset_root, 
                               split_file = test_split, 
                               is_train   = False, 
                               model_name = model_name,
                               resize_map = RESIZE_MAP
                               )
    
    probs, idxs = collect_sigmoid_probs(
        model, test_ds, device, batch_size, num_workers, desc="Test forward"
    )

    rows: list[dict[str, str]] = []
    for prob, idx in tqdm(
        list(zip(probs, idxs)), total=len(idxs), desc="Resize & RLE"
    ):
        w, h = test_ds.load_image(idx).size
        prob_hw = resize_prob_to_original(prob, h, w)
        binary = (prob_hw > 0.5).astype(np.uint8)
        image_id = test_ds.image_list[idx].stem
        rows.append(
            {
                "image_id": image_id,
                "encoded_mask": rle_encode(binary),
            }
        )

    csv_path = os.path.join(save_dir, csv_name)
    with open(csv_path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "encoded_mask"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    return csv_path


if __name__ == "__main__":

    # Print Dice on your test split (two numbers: soft/hard).
    save_dir = os.path.join("saved_models", MODEL_NAME, TIMESTAMP)
    ckpt = os.path.join(save_dir, f"{MODEL_NAME}.pth")

    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt} — update TIMESTAMP or path")

    model = load_model(ckpt, DEVICE)
    print(f"Checkpoint : {ckpt}")

    test_ds = OxfordPetDataset(
        root       = DATASET_ROOT,
        split_file = TEST_SPLIT,
        is_train   = False,
        model_name = MODEL_NAME,
        resize_map = RESIZE_MAP
    )
    test_loader = DataLoader(
        test_ds,
        batch_size         = BATCH_SIZE,
        shuffle            = False,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = True,
    )

    soft_sum = 0.0
    hard_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc = "Test Dice", leave = False):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            probs = torch.sigmoid(model(images))
            soft, hard = dice_scores_from_probs(probs, masks, threshold=0.5)
            soft_sum += float(soft.sum().item())
            hard_sum += float(hard.sum().item())
            n += int(soft.numel())

    print(f"Soft Dice: {soft_sum / max(n, 1):.6f}")
    print(f"Hard Dice: {hard_sum / max(n, 1):.6f}")

    write_submission_csv(
        model        = model,
        model_name   = MODEL_NAME,
        dataset_root = DATASET_ROOT,
        test_split   = TEST_SPLIT,
        save_dir     = save_dir,
        device       = DEVICE,
        batch_size   = BATCH_SIZE,
        num_workers  = NUM_WORKERS,
        csv_name     = f"submission_{MODEL_NAME}.csv"
    )
    
