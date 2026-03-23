from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

TARGET_SIZE: Tuple[int, int] = (388, 388)  # 572 = 388 + 92*2 (UNet padded input)

GAMMA_MIN: float = 0.8
GAMMA_MAX: float = 1.2


def _pad_to_min_hw(
    image: Image.Image,
    trimap: Image.Image,
    min_h: int,
    min_w: int,
) -> Tuple[Image.Image, Image.Image]:
    """Pad image (reflect) and trimap (constant 0) so both are at least min_h × min_w."""
    w, h = image.size
    pad_left = pad_top = pad_right = pad_bottom = 0
    if w < min_w:
        diff = min_w - w
        pad_left = diff // 2
        pad_right = diff - pad_left
    if h < min_h:
        diff = min_h - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
    if pad_left or pad_top or pad_right or pad_bottom:
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        image = F.pad(image, padding, fill=0, padding_mode="reflect")
        trimap = F.pad(trimap, padding, fill=0, padding_mode="constant")
    return image, trimap


def _random_crop_pair(
    image: Image.Image,
    trimap: Image.Image,
    crop_h: int,
    crop_w: int,
) -> Tuple[Image.Image, Image.Image]:
    """Apply the same spatial crop to RGB image and trimap."""
    w, h = image.size
    top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
    left = int(torch.randint(0, w - crop_w + 1, (1,)).item())
    image = F.crop(image, top, left, crop_h, crop_w)
    trimap = F.crop(trimap, top, left, crop_h, crop_w)
    return image, trimap


class OxfordPetDataset(Dataset):
    def __init__(self, root, split_file, transform = None, is_train = False):
        self.root = Path(root)
        self.split_file = self.root / split_file
        self.is_train = is_train
        
        with open(self.split_file, "r") as f:
            sample_ids = [line.strip() for line in f if line.strip()]

        self.image_list = [self.root / "images" / f"{sample_id}.jpg" for sample_id in sample_ids]
        self.trimap_list = [self.root / "annotations" / "trimaps" / f"{sample_id}.png" for sample_id in sample_ids]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def load_trimap(self, idx):
        return Image.open(self.trimap_list[idx])
    
    def load_image(self, idx):
        return Image.open(self.image_list[idx])
    
    def __getitem__(self, idx):
        image = self.load_image(idx).convert("RGB")
        trimap = self.load_trimap(idx)

        crop_h, crop_w = TARGET_SIZE
        if self.is_train:
            image, trimap = _pad_to_min_hw(image, trimap, crop_h, crop_w)
            image, trimap = _random_crop_pair(image, trimap, crop_h, crop_w)

            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)
                trimap = F.hflip(trimap)

            if torch.rand(1).item() < 0.5:
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
                image = jitter(image)

        image = F.rgb_to_grayscale(image)
        image = F.to_tensor(image)

        if self.is_train:
            image = F.pad(image, padding=92, fill=0, padding_mode="reflect")

        trimap = np.array(trimap)
        trimap = (trimap == 1).astype(np.uint8)
        trimap = torch.from_numpy(trimap).long()

        return image, trimap, idx

    
if __name__ == "__main__":

    from src.utils import plot_sample

    dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", split_file = "train.txt", is_train = False)
    image, trimap, idx = dataset[0]

    size = dataset.load_image(idx).size[::-1]

    image = image[:, 92:-92, 92:-92].float()
    image = F.resize(image, size = size).squeeze(0)
    trimap = F.resize(trimap.unsqueeze(0).float(), size = size, interpolation =F.InterpolationMode.NEAREST).squeeze(0)

    plot_sample(image.numpy(), trimap.numpy(), image_title="Image", mask_title="Trimap", save_path="sample.png")