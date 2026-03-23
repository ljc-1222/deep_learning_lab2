from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

TARGET_SIZE:    Tuple[int, int] = (388, 388)
IMAGENET_MEAN: list[float]     = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float]     = [0.229, 0.224, 0.225]

def _random_resized_crop_pair(
    image: Image.Image,
    trimap: Image.Image,
    target_size: Tuple[int, int],
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Tuple[Image.Image, Image.Image]:
    """Apply the same random resized crop to RGB image and trimap."""
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)

    image = F.crop(image, i, j, h, w)
    trimap = F.crop(trimap, i, j, h, w)

    image = F.resize(image, target_size, interpolation=F.InterpolationMode.BILINEAR)
    trimap = F.resize(trimap, target_size, interpolation=F.InterpolationMode.NEAREST)

    return image, trimap


class OxfordPetDataset(Dataset):
    def __init__(self, root, split_file, transform=None, is_train=False):
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

        if self.is_train:
            image, trimap = _random_resized_crop_pair(image, trimap, TARGET_SIZE)

            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)
                trimap = F.hflip(trimap)

            if torch.rand(1).item() < 0.5:
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                image = jitter(image)
            
            image = F.pad(image, padding=92, fill=0, padding_mode="reflect")

        image = F.to_tensor(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        trimap = np.array(trimap)
        trimap = (trimap == 1).astype(np.uint8)
        trimap = torch.from_numpy(trimap).long()

        return image, trimap, idx

    
if __name__ == "__main__":

    from src.utils import plot_sample

    dataset = OxfordPetDataset(root="dataset/oxford-iiit-pet", split_file="train.txt", is_train=False)
    image, trimap, idx = dataset[0]

    size = dataset.load_image(idx).size[::-1]

    image = image[:, 92:-92, 92:-92].float()
    image = F.resize(image, size=size).permute(1, 2, 0).numpy()  # H×W×3 for plot_sample
    trimap = F.resize(trimap.unsqueeze(0).float(), size=size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)

    plot_sample(image, trimap.numpy(), image_title="Image", mask_title="Trimap", save_path="sample.png")