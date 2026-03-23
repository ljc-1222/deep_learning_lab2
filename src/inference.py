import os
import csv
import torch
import numpy as np
import torchvision.transforms.functional as TF

from tqdm import tqdm
from scipy import ndimage
from skimage import morphology

from src.models.unet import UNet
from src.evaluate import dice_score
from src.oxford_pet import OxfordPetDataset
from src.utils import prepare_five_crops, rle_encode, stitch_five_crop_results

TIMESTAMP      = "20260323-040637"
BATCH_SIZE     = 16
DATASET_ROOT   = "dataset/oxford-iiit-pet"
VAL_SPLIT      = "val.txt"
TEST_SPLIT     = "test_unet.txt"
THRESHOLD_MIN  = 0.30
THRESHOLD_MAX  = 0.70
THRESHOLD_STEP = 0.05

NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device) -> UNet:
    """Load a UNet checkpoint and set it to eval mode.

    Args:
        model_path: Path to the `.pth` checkpoint file.
        device: Target device.

    Returns:
        Loaded UNet model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    raw_state = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw_state.items()}

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


_TTA_JITTER = (1.0, 1.2, 0.8)


def _jitter(x: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return x
    return TF.adjust_contrast(TF.adjust_brightness(x, factor), factor)


@torch.no_grad()
def tta_predict(model: UNet, image_batch: torch.Tensor) -> torch.Tensor:
    """6-pass TTA: identity + hflip for each of 3 brightness/contrast jitters.

    Args:
        model: Trained UNet model.
        image_batch: Input batch tensor.

    Returns:
        Averaged softmax probability tensor.
    """
    def fwd(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(x), dim=1)

    probs = []
    for jf in _TTA_JITTER:
        img = _jitter(image_batch, jf)
        probs.append(fwd(img))
        probs.append(TF.hflip(fwd(TF.hflip(img))))

    return torch.stack(probs).mean(dim=0)


def collect_probs(
    model: UNet,
    dataset: OxfordPetDataset,
    desc: str = "Collecting probs",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Run TTA inference over a dataset and collect per-image fg probabilities.

    Args:
        model: Trained UNet model.
        dataset: Dataset to iterate over.
        desc: tqdm progress bar label.

    Returns:
        Tuple of (fg_probs, targets, dataset_indices).
    """
    all_probs, all_targets, all_idxs = [], [], []

    for idx in tqdm(range(len(dataset)), desc=desc):
        pil_image = dataset.load_image(idx).convert("RGB")
        batch, positions, H_p, W_p, pad_top, pad_left, H_orig, W_orig = prepare_five_crops(pil_image)

        probs = tta_predict(model, batch.to(DEVICE)).cpu()
        stitched = stitch_five_crop_results(probs, positions, H_p, W_p, pad_top, pad_left, H_orig, W_orig)

        trimap_np = np.array(dataset.load_trimap(idx))
        all_probs.append(stitched[1].numpy().astype(np.float32))
        all_targets.append((trimap_np == 1).astype(np.uint8))
        all_idxs.append(idx)

    return all_probs, all_targets, all_idxs


def find_optimal_threshold(
    all_probs: list[np.ndarray],
    all_targets: list[np.ndarray],
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Grid-search the threshold that maximises mean Dice on a validation set.

    Args:
        all_probs: Per-image foreground probability maps.
        all_targets: Per-image binary ground-truth masks.
        thresholds: 1-D array of candidate threshold values.

    Returns:
        Tuple of (best_threshold, best_mean_dice).
    """
    best_thresh = float(thresholds[len(thresholds) // 2])
    best_dice   = -1.0

    for t in tqdm(thresholds, desc="Threshold search", leave=False, unit="t"):
        mean_dice = np.mean([
            2.0 * np.sum((p > t) * tgt) / (np.sum(p > t) + np.sum(tgt) + 1e-8)
            for p, tgt in zip(all_probs, all_targets)
        ])
        if mean_dice > best_dice:
            best_dice, best_thresh = float(mean_dice), float(t)

    return best_thresh, best_dice


def _apply_postprocess(mask: np.ndarray) -> np.ndarray:
    """Remove small objects and fill holes in a binary mask."""
    mask = morphology.remove_small_objects(mask, min_size = 500)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask

def run_val(
    model: UNet,
    dataset_root: str,
    val_split: str,
    threshold_min: float = THRESHOLD_MIN,
    threshold_max: float = THRESHOLD_MAX,
    threshold_step: float = THRESHOLD_STEP,
) -> tuple[float, float, list[np.ndarray], list[np.ndarray], list[int], OxfordPetDataset]:
    """Validate with TTA: find optimal threshold and compute Dice on the val set.

    Args:
        model: Trained UNet model.
        dataset_root: Root directory of the dataset.
        val_split: Filename of the validation split list.
        threshold_min: Lower bound for threshold search.
        threshold_max: Upper bound for threshold search.
        threshold_step: Step size for threshold search.

    Returns:
        Tuple of (best_thresh, val_dice, probs, targets, idxs, dataset).
    """
    dataset    = OxfordPetDataset(root=dataset_root, split_file=val_split, is_train=False)
    thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)

    all_probs, all_targets, all_idxs = collect_probs(model, dataset, desc="Val (TTA)")
    best_thresh, _ = find_optimal_threshold(all_probs, all_targets, thresholds)

    per_dice = [
        float(dice_score(
            torch.from_numpy(_apply_postprocess(p > best_thresh).astype(np.float32)),
            torch.from_numpy(t.astype(np.float32)),
        ))
        for p, t in zip(all_probs, all_targets)
    ]
    val_dice = float(np.mean(per_dice))

    W = 44
    print(f"\n{'─'*W}")
    print(" Validation — TTA (6-pass)".center(W))
    print(f"{'─'*W}")
    print(f"  Threshold : {best_thresh:.2f}")
    print(f"  Val Dice  : {val_dice:.6f}")
    print(f"{'─'*W}")

    return best_thresh, val_dice, all_probs, all_targets, all_idxs, dataset


def generate_submission(
    model: UNet,
    dataset_root: str,
    test_split: str,
    best_thresh: float,
    save_dir: str,
) -> str:
    """Run TTA inference on the test set and write a submission CSV.

    Args:
        model: Trained UNet model.
        dataset_root: Root directory of the dataset.
        test_split: Filename of the test split list.
        best_thresh: Binarisation threshold (from validation).
        save_dir: Directory to write ``submission.csv`` into.

    Returns:
        Path to the written CSV file.
    """
    dataset = OxfordPetDataset(root=dataset_root, split_file=test_split, is_train=False)
    all_probs, _, all_idxs = collect_probs(model, dataset, desc="Test (TTA)")

    rows = [
        {
            "image_id": dataset.image_list[idx].stem,
            "encoded_mask": rle_encode(_apply_postprocess(fg_prob > best_thresh)),
        }
        for fg_prob, idx in tqdm(zip(all_probs, all_idxs), total=len(all_probs), desc="Binarize & RLE")
    ]

    csv_path = os.path.join(save_dir, "submission.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "encoded_mask"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def plot_worst_predictions(
    probs: list[np.ndarray],
    per_dice: list[float],
    idxs: list[int],
    thresh: float,
    dataset: OxfordPetDataset,
    save_dir: str,
    n: int = 3,
) -> None:
    """Save visualisations of the n worst-Dice predictions.

    Args:
        probs: Per-image foreground probability maps.
        per_dice: Per-image Dice scores.
        idxs: Dataset indices corresponding to each entry in *probs*.
        thresh: Binarisation threshold.
        dataset: Dataset used for loading images.
        save_dir: Directory to write the PNG files into.
        n: Number of worst samples to visualise.
    """
    from src.utils import plot_sample

    for rank, i in enumerate(sorted(range(len(per_dice)), key=lambda i: per_dice[i])[:n], start=1):
        int_idx  = idxs[i]
        image_np = np.array(dataset.load_image(int_idx).convert("L"), dtype=np.float32) / 255.0
        pred_mask = (probs[i] > thresh).astype(np.uint8)
        plot_sample(
            image_np,
            pred_mask,
            title = f"Worst {rank} — Dice: {per_dice[i]:.6f}",
            mask_title = "Predicted Mask",
            save_path = os.path.join(save_dir, f"worst_{rank}.png"),
        )


if __name__ == "__main__":
    
    save_dir = os.path.join("saved_models", "unet", TIMESTAMP)
    ckpt     = os.path.join(save_dir, "unet_p2.pth")

    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}  — update TIMESTAMP")

    model = load_model(ckpt, DEVICE)
    print(f"Checkpoint : {ckpt}")
    print(f"Device     : {DEVICE}")

    best_thresh, val_dice, probs, targets, idxs, val_dataset = run_val(model, DATASET_ROOT, VAL_SPLIT)

    per_dice = [
        float(dice_score(
            torch.from_numpy(_apply_postprocess(p > best_thresh).astype(np.float32)),
            torch.from_numpy(t.astype(np.float32)),
        ))
        for p, t in zip(probs, targets)
    ]
    plot_worst_predictions(probs, per_dice, idxs, best_thresh, val_dataset, save_dir)

    # csv_path = generate_submission(model, DATASET_ROOT, TEST_SPLIT, best_thresh, save_dir)
    # print(f"\nSubmission saved → {csv_path}")
