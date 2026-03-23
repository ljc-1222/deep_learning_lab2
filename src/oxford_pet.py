import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


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
        image = self.load_image(idx).convert("RGB") # (H, W, 3)
        trimap = self.load_trimap(idx) # (H, W)
        
        target_size = (388, 388)
        image = F.resize(image, size = target_size) # (target_size, 3)
        trimap = F.resize(trimap, size = target_size, interpolation =F.InterpolationMode.NEAREST) # (target_size)

        if self.is_train:
            jitter = transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05)
            image = jitter(image)

            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)
                trimap = F.hflip(trimap)
            
        image = F.to_tensor(image)
        image = F.pad(image, padding = 92, fill = 0, padding_mode = "reflect")

        trimap = np.array(trimap)
        trimap = (trimap == 1).astype(np.uint8)
        trimap = torch.from_numpy(trimap).long()
        
        return image, trimap, idx
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", split_file = "train.txt", is_train = False)
    image, trimap, idx = dataset[0]
    
    size = dataset.load_image(idx).size[::-1]
    
    image = image[:, 92:-92, 92:-92].float()
    image = F.resize(image, size = size).squeeze(0)
    trimap = F.resize(trimap.unsqueeze(0).float(), size = size, interpolation =F.InterpolationMode.NEAREST).squeeze(0)
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].imshow(image, cmap = "gray")
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(trimap, cmap = "viridis", vmin = 0, vmax = 1)
    axes[1].set_title("Trimap")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()