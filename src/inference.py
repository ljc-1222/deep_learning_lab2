"""UNet inference: full-resolution binary masks as Fortran-order RLE submission CSV."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.oxford_pet import OxfordPetDataset
from src.utils import rle_encode

TIMESTAMP = "20260323-064056"
BATCH_SIZE = 32
DATASET_ROOT = "dataset/oxford-iiit-pet"
TEST_SPLIT = "test_unet.txt"
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, device: torch.device) -> UNet:

    raw_state = torch.load(model_path, map_location = device, weights_only = True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw_state.items()}

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def resize_prob_to_original(prob: np.ndarray, height: int, width: int) -> np.ndarray:

    t = torch.from_numpy(prob.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size = (height, width), mode = "bilinear", align_corners = False)
    return out.squeeze(0).squeeze(0).numpy()


@torch.no_grad()
def collect_sigmoid_probs(
    model: UNet,
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
        persistent_workers = True,
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
    model: UNet,
    dataset_root: str,
    test_split: str,
    save_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    csv_name: str = "submission.csv",
) -> str:

    os.makedirs(save_dir, exist_ok=True)
    test_ds = OxfordPetDataset(root=dataset_root, split_file=test_split, is_train=False)
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
    save_dir = os.path.join("saved_models", "unet", TIMESTAMP)
    ckpt = os.path.join(save_dir, "unet_p2.pth")

    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt} — update TIMESTAMP or path")

    model = load_model(ckpt, DEVICE)
    print(f"Checkpoint : {ckpt}")
    print(f"Device     : {DEVICE}")

    write_submission_csv(
        model,
        DATASET_ROOT,
        TEST_SPLIT,
        save_dir,
        DEVICE,
        BATCH_SIZE,
        NUM_WORKERS,
    )
