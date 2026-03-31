from pathlib import Path
from typing import Tuple, Union

import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def train_transform() -> A.Compose:

    return A.Compose(
        [
            A.HorizontalFlip(p = 0.5),
            A.SafeRotate(limit = 20, p = 0.5),
            A.RandomBrightnessContrast(p = 0.5),
            A.Normalize(mean = [0.5070823478322863, 0.474229456630694, 0.4202200043649814], 
                        std = [0.2662919044873643, 0.26026855187836595, 0.26879510227266507]),
            ToTensorV2(),
        ],
    )


def eval_transform() -> A.Compose:

    return A.Compose(
        [
            A.Normalize(mean = [0.5070823478322863, 0.474229456630694, 0.4202200043649814], 
                        std = [0.2662919044873643, 0.26026855187836595, 0.26879510227266507]),
            ToTensorV2(),
        ],
    )
    
def apply_clahe_rgb(image: np.ndarray) -> np.ndarray:

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply CLAHE only on the L channel in LAB space.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    l_channel = clahe.apply(l_channel)
    
    lab = cv2.merge((l_channel, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return rgb_enhanced

class OxfordPetDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        split_file: Union[str, Path],
        is_train: bool = False,
        model_name: str = "UNet",
        resize_map: dict = None,
    ) -> None:
        
        self.root = Path(root)
        self.split_file = self.root / split_file
        self.is_train = is_train
        self.transform = train_transform() if is_train else eval_transform()
        self.model_name = model_name
        self.resize_map = resize_map
        
        with open(self.split_file, "r", encoding = "utf-8") as f:
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

        image = apply_clahe_rgb(image)
        transformed = self.transform(image = image, mask = mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].unsqueeze(0)
        
        image_t = F.resize(image_t, 
                           size = (self.resize_map[self.model_name]), 
                           interpolation = F.InterpolationMode.BILINEAR
                           )
        
        # Pad the image to IMAGE_SIZE if model is UNet
        if self.model_name == "UNet":
            image_t = F.pad(image_t, 
                            padding = 92,
                            padding_mode = "constant"
                            )
            
        mask_t = F.resize(mask_t, 
                          size = (self.resize_map[self.model_name]), 
                          interpolation = F.InterpolationMode.NEAREST
                          )

        # Ensure worker collation uses standalone, contiguous storages.
        image_t = image_t.contiguous().clone()
        mask_t = mask_t.contiguous().clone()

        return image_t, mask_t, idx

        return image, trimap, idx

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    dataset = OxfordPetDataset(
        root = "dataset/oxford-iiit-pet",
        split_file = "train.txt",
        is_train = False,
        model_name = "UNet",
        resize_map = {
            "UNet": (388, 388),
            "ResNet34UNet": (384, 384),
        }
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

