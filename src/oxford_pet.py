from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import numpy as np
import albumentations as A
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE: int = 572
MASK_SIZE:  int = 388

def train_transform() -> A.Compose:

    return A.Compose(
        [
            A.HorizontalFlip(p = 0.5),
            A.RandomResizedCrop(size = (IMAGE_SIZE, IMAGE_SIZE), scale = (0.8, 1)),
            A.ShiftScaleRotate(
                shift_limit  = 0.2,
                scale_limit  = 0.2,
                rotate_limit = 30,
                p            = 0.5,
            ),
            A.ColorJitter(
                brightness = 0.2,
                contrast   = 0.2,
                saturation = 0.2,
                hue        = 0.2,
                p          = 0.5,
            ),
            A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )


def eval_transform() -> A.Compose:

    return A.Compose(
        [
            A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )
class OxfordPetDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        split_file: Union[str, Path],
        is_train: bool = False,
        transform: Optional[A.Compose] = None,
    ) -> None:
        
        self.root = Path(root)
        self.split_file = self.root / split_file
        self.is_train = is_train
        if transform is not None:
            self.transform = transform
        else:
            self.transform = train_transform() if is_train else eval_transform()

        with open(self.split_file, "r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]

        self.image_list = [
            self.root / "images" / f"{sample_id}.jpg" for sample_id in sample_ids
        ]
        self.trimap_list = [
            self.root / "annotations" / "trimaps" / f"{sample_id}.png"
            for sample_id in sample_ids
        ]

    def __len__(self) -> int:
        
        return len(self.image_list)

    def load_trimap(self, idx: int) -> Image.Image:

        return Image.open(self.trimap_list[idx])

    def load_image(self, idx: int) -> Image.Image:

        return Image.open(self.image_list[idx])

    @staticmethod
    def _trimap_to_binary_mask(trimap: np.ndarray) -> np.ndarray:
        
        return (trimap == 1).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:

        image = np.array(self.load_image(idx).convert("RGB"))
        trimap = np.array(self.load_trimap(idx))
        mask = self._trimap_to_binary_mask(trimap)

        transformed = self.transform(image = image, mask = mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].unsqueeze(0)
        
        image_t = F.resize(image_t, size = (IMAGE_SIZE, IMAGE_SIZE), interpolation = F.InterpolationMode.NEAREST)
        mask_t = F.resize(mask_t, size = (MASK_SIZE, MASK_SIZE), interpolation = F.InterpolationMode.NEAREST)

        return image_t, mask_t, idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = OxfordPetDataset(
        root = "dataset/oxford-iiit-pet",
        split_file = "train.txt",
        is_train = False,
    )
    
    image, trimap, idx = dataset[0]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    vis = (image * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    size = dataset.load_image(idx).size[::-1]

    image_up = F.resize(torch.from_numpy(vis).permute(2, 0, 1), size = size)
    trimap_up = F.resize(
        trimap,
        size = size,
        interpolation = F.InterpolationMode.NEAREST,
    ).squeeze(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_up.permute(1, 2, 0).numpy())
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(trimap_up.numpy(), cmap = "viridis", vmin = 0, vmax = 1)
    axes[1].set_title("Mask")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
