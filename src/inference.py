from __future__ import annotations

import argparse
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

DEFAULT_MODEL_NAME = "ResNet34UNet"
DEFAULT_TIMESTAMP = "20260331-094415"
DEFAULT_USE_SWA = True

BATCH_SIZE = 32
DATASET_ROOT = "dataset/oxford-iiit-pet"
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIZE_MAP = {
    "UNet": (388, 388),
    "ResNet34UNet": (384, 384),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference and generate submission csv.")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["UNet", "ResNet34UNet"],
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--timestamp", type=str, default=DEFAULT_TIMESTAMP)
    parser.add_argument(
        "--use-swa",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SWA,
        help="Use SWA checkpoint (<model>_swa.pth) instead of base checkpoint.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def get_test_split(model_name: str) -> str:
    if model_name == "UNet":
        return "test_unet.txt"
    if model_name == "ResNet34UNet":
        return "test_res_unet.txt"
    raise ValueError(f"Invalid model name: {model_name}")


def load_model(
    model_path: str,
    model_name: str,
    device: torch.device,
) -> Union[UNet, ResNet34UNet]:

    raw_state = torch.load(model_path, map_location = device, weights_only = True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw_state.items()}

    if model_name == "UNet":
        model = UNet().to(device)
    elif model_name == "ResNet34UNet":
        model = ResNet34UNet().to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

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
    use_persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = use_persistent_workers,
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
    args = parse_args()

    model_name = args.model_name
    timestamp = args.timestamp
    use_swa = args.use_swa
    test_split = get_test_split(model_name)

    save_dir = os.path.join("saved_models", model_name, timestamp)
    ckpt_name = f"{model_name}_swa.pth" if use_swa else f"{model_name}.pth"
    ckpt = os.path.join(save_dir, ckpt_name)

    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt} -- update timestamp/path or SWA option")

    model = load_model(ckpt, model_name, DEVICE)
    print(f"Checkpoint : {ckpt}")

    write_submission_csv(
        model        = model,
        model_name   = model_name,
        dataset_root = DATASET_ROOT,
        test_split   = test_split,
        save_dir     = save_dir,
        device       = DEVICE,
        batch_size   = BATCH_SIZE,
        num_workers  = NUM_WORKERS,
        csv_name     = f"submission_{model_name}.csv"
    )
    
