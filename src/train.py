import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet
from torch.utils.data import DataLoader
from src.oxford_pet import OxfordPetDataset
from src.evaluate import dice_score, combined_loss, combined_loss_lovasz
from src.utils import (
    predict_full_image,
    print_training_config,
    plot_training_curves,
)

# Number of workers
NUM_WORKERS = 6

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PyTorch SequentialLR internally passes `epoch` to sub-scheduler.step(),
# which triggers a deprecation UserWarning that is not actionable from user code.
warnings.filterwarnings(
    "ignore",
    message = "The epoch parameter in `scheduler.step\\(\\)`",
    category = UserWarning,
    module = "torch.optim.lr_scheduler",
)

# Seed
SEED = 42

# Model selection
MODEL_NAME = "UNet"

# Training hyperparameters
BATCH_SIZE        = 16
P1_EPOCHS         = 50
P2_EPOCHS         = 100
DICE_WEIGHT       = 2.0
LOVASZ_WEIGHT     = 0.8
P2_LR_DIVISOR     = 8
FOCAL_GAMMA       = 0.0

# Optimizer hyperparameters
WEIGHT_DECAY  = 5e-4
LEARNING_RATE = 1e-4

# Scheduler hyperparameters
P1_WARMUP_EPOCHS      = 5
P2_WARMUP_EPOCHS      = 3
P1_WARMUP_START_FACTOR = 0.01
P2_WARMUP_START_FACTOR = 0.05
COSINE_ETA_MIN_FACTOR  = 0.01

# Early stopping hyperparameters
P1_PATIENCE = 5
P2_PATIENCE = 12

torch.manual_seed(SEED)

train_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                                 split_file = "train.txt", 
                                 is_train = True)

val_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                               split_file = "val.txt", 
                               is_train = False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
    pin_memory = True,
    persistent_workers = True,
)

def train_one_epoch(model, optimizer, loss, train_dataloader, scheduler = None) -> float:

    training_loss = 0.0
    model.train()
    train_pbar = tqdm(train_dataloader, desc="Training  ")

    for image, trimap, _ in train_pbar:

        image  = image.to(DEVICE)
        trimap = trimap.to(DEVICE)

        output     = model(image)
        loss_value = loss(output, trimap)

        training_loss += loss_value.item()
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_pbar.set_postfix({'loss': f"{loss_value.item():.6f}"})

    return training_loss / len(train_dataloader)
        
def validate_one_epoch(
    model: UNet,
    val_dataset: OxfordPetDataset,
) -> tuple[float, float]:
    """Validate with full-image inference: resize → normalize → pad → model → upsample.

    The image is resized to ``TARGET_SIZE``, padded by 92 px on all sides
    (matching the training input convention), passed through the model, then
    bilinearly upsampled back to the original resolution for loss and Dice
    computation.

    Args:
        model: Trained UNet model.
        val_dataset: Validation dataset (``is_train=False``).

    Returns:
        Tuple of (mean_val_loss, mean_dice_score).
    """
    model.eval()
    validation_loss  = 0.0
    epoch_dice_score = 0.0

    with torch.no_grad():
        val_pbar = tqdm(range(len(val_dataset)), desc="Validation")
        for idx in val_pbar:
            # dataset[idx] returns (normalized_tensor [3,H,W], trimap, idx)
            img_normalized, _, _ = val_dataset[idx]

            stitched = predict_full_image(
                lambda b: torch.softmax(model(b.to(DEVICE)), dim=1),
                img_normalized,
            )  # [2, H_orig, W_orig]

            trimap_np = (np.array(val_dataset.load_trimap(idx)) == 1).astype(np.uint8)
            trimap_t  = torch.from_numpy(trimap_np).long()

            log_probs  = torch.log(stitched.clamp(min=1e-8))
            loss_value = F.nll_loss(log_probs.unsqueeze(0), trimap_t.unsqueeze(0))
            validation_loss += loss_value.item()

            pred = torch.argmax(stitched, dim=0).float()
            epoch_dice_score += dice_score(pred, trimap_t.float())

            val_pbar.set_postfix({"loss": f"{loss_value.item():.6f}"})

    return validation_loss / len(val_dataset), epoch_dice_score / len(val_dataset)

if __name__ == "__main__":
    
    if MODEL_NAME == "UNet":
        model  = torch.compile(UNet().to(DEVICE))
    elif MODEL_NAME == "ResNet34_UNet":
        model  = torch.compile(ResNet34UNet().to(DEVICE))

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    _model_slug   = MODEL_NAME.lower()
    save_dir      = os.path.join("saved_models", _model_slug, run_timestamp)
    os.makedirs(save_dir, exist_ok = True)
    p1_best_model_path = os.path.join(save_dir, f"{_model_slug}_p1.pth")
    p2_best_model_path = os.path.join(save_dir, f"{_model_slug}_p2.pth")

    print_training_config(
        run_timestamp            = run_timestamp,
        p1_best_model_path       = p1_best_model_path,
        p2_best_model_path       = p2_best_model_path,
        device                   = str(DEVICE),
        model_name               = MODEL_NAME,
        phase1_epochs            = P1_EPOCHS,
        phase2_epochs            = P2_EPOCHS,
        batch_size               = BATCH_SIZE,
        learning_rate            = LEARNING_RATE,
        weight_decay             = WEIGHT_DECAY,
        dice_alpha               = DICE_WEIGHT,
        lovasz_beta              = LOVASZ_WEIGHT,
        p1_warmup_epochs         = P1_WARMUP_EPOCHS,
        p2_warmup_epochs         = P2_WARMUP_EPOCHS,
        p1_warmup_start_factor   = P1_WARMUP_START_FACTOR,
        p2_warmup_start_factor   = P2_WARMUP_START_FACTOR,
        p1_cosine_eta_min_factor = COSINE_ETA_MIN_FACTOR,
        p2_cosine_eta_min_factor = COSINE_ETA_MIN_FACTOR,
        patience_phase1          = P1_PATIENCE,
        p2_lr_divisor            = P2_LR_DIVISOR,
        patience_phase2          = P2_PATIENCE,
    )

    train_losses:    list[float] = []
    val_losses:      list[float] = []
    val_dice_scores: list[float] = []

    try:
        # ── Phase 1: Focal + Dice, linear warmup then constant LR ────────────
        tqdm.write(f"\n── Phase 1: Focal(gamma = {FOCAL_GAMMA}) + Dice ({P1_EPOCHS} epochs, warmup {P1_WARMUP_EPOCHS} ep) ──")

        steps_per_epoch = len(train_dataloader)

        # Training settings
        p1_lr = LEARNING_RATE
        p1_loss = lambda output, target: combined_loss(output, target, alpha = DICE_WEIGHT, gamma = FOCAL_GAMMA)    
        p1_optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr           = p1_lr, 
            weight_decay = WEIGHT_DECAY,
        )
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            p1_optimizer,
            start_factor = P1_WARMUP_START_FACTOR,
            end_factor   = 1.0,
            total_iters  = P1_WARMUP_EPOCHS * steps_per_epoch,
        )
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            p1_optimizer,
            T_max   = (P1_EPOCHS - P1_WARMUP_EPOCHS) * steps_per_epoch,
            eta_min = 1e-6, 
        )
        
        p1_scheduler = torch.optim.lr_scheduler.SequentialLR(
            p1_optimizer,
            schedulers = [warmup_scheduler, cosine_scheduler],
            milestones = [P1_WARMUP_EPOCHS * steps_per_epoch]
        )

        best_p1_dice = -1.0
        p1_epochs_no_improve = 0

        # Training loop
        for epoch in tqdm(range(P1_EPOCHS), desc = "Phase 1"):

            training_loss = train_one_epoch(model, p1_optimizer, p1_loss, train_dataloader,
                                            scheduler=p1_scheduler)
            validation_loss, epoch_dice_score = validate_one_epoch(model, val_dataset)

            train_losses.append(training_loss)
            val_losses.append(validation_loss)
            val_dice_scores.append(epoch_dice_score.item() if hasattr(epoch_dice_score, "item") else float(epoch_dice_score))

            tqdm.write(
                f"[P1] Epoch {epoch+1:2d}/{P1_EPOCHS} | "
                f"Train Loss: {training_loss:.6f} | "
                f"Val Loss: {validation_loss:.6f} | "
                f"Val Dice: {epoch_dice_score:.6f}"
            )


            # Early stopping & checkpoint saving
            if epoch_dice_score > best_p1_dice:
                best_p1_dice         = epoch_dice_score
                p1_epochs_no_improve = 0
                torch.save(model.state_dict(), p1_best_model_path)
                tqdm.write(f"  ↑ New best Dice: {best_p1_dice:.6f} — checkpoint saved")
            else:
                p1_epochs_no_improve += 1
                if p1_epochs_no_improve >= P1_PATIENCE:
                    tqdm.write(
                        f"Early stopping at P1 epoch {epoch+1}. Best Dice: {best_p1_dice:.6f}"
                    )
                    break

        tqdm.write(f"Phase 1 complete. Best Dice: {best_p1_dice:.6f}")

        model.load_state_dict(torch.load(p1_best_model_path, weights_only=True))
        tqdm.write(f"Loaded best Phase 1 weights → {p1_best_model_path}")

        # ── Phase 2: Focal + Lovász, LR/10, CosineAnnealing, early stopping ───
        tqdm.write(f"\n── Phase 2: Focal(gamma = {FOCAL_GAMMA}) + Lovász (max {P2_EPOCHS} epochs, CosineAnnealingLR) ──")

        p2_lr = LEARNING_RATE / P2_LR_DIVISOR
        p2_loss = lambda output, target: combined_loss_lovasz(output, target, beta = LOVASZ_WEIGHT, gamma = FOCAL_GAMMA)
        p2_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = p2_lr,
            weight_decay = WEIGHT_DECAY / 20,
        )
        p2_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            p2_optimizer,
            start_factor = P2_WARMUP_START_FACTOR,
            end_factor   = 1.0,
            total_iters  = P2_WARMUP_EPOCHS * steps_per_epoch,
        )

        p2_cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            p2_optimizer,
            T_max   = (P2_EPOCHS - P2_WARMUP_EPOCHS) * steps_per_epoch,
            eta_min = p2_lr * COSINE_ETA_MIN_FACTOR,
        )
        
        p2_scheduler = torch.optim.lr_scheduler.SequentialLR(
            p2_optimizer,
            schedulers = [p2_warmup_scheduler, p2_cosine_scheduler],
            milestones = [P2_WARMUP_EPOCHS * steps_per_epoch]
        )

        best_p2_dice = -1.0
        epochs_without_improve = 0

        # Training loop
        for epoch in tqdm(range(P2_EPOCHS), desc = "Phase 2"):

            training_loss = train_one_epoch(model, p2_optimizer, p2_loss, train_dataloader,
                                            scheduler=p2_scheduler)
            validation_loss, epoch_dice_score = validate_one_epoch(model, val_dataset)

            train_losses.append(training_loss)
            val_losses.append(validation_loss)
            val_dice_scores.append(epoch_dice_score.item() if hasattr(epoch_dice_score, "item") else float(epoch_dice_score))

            tqdm.write(
                f"[P2] Epoch {epoch+1:2d}/{P2_EPOCHS} | "
                f"Train Loss: {training_loss:.6f} | "
                f"Val Loss: {validation_loss:.6f} | "
                f"Val Dice: {epoch_dice_score:.6f}"
            )
            
            # Early stopping & checkpoint saving
            if epoch_dice_score > best_p2_dice:
                best_p2_dice           = epoch_dice_score
                epochs_without_improve = 0
                torch.save(model.state_dict(), p2_best_model_path)
                tqdm.write(f"  ↑ New best Dice: {best_p2_dice:.6f} — checkpoint saved")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= P2_PATIENCE:
                    tqdm.write(
                        f"Early stopping at P2 epoch {epoch+1}. "
                        f"Best Dice: {best_p2_dice:.6f}"
                    )
                    break

    finally:
        # Plot training curves
        if train_losses:
            plot_training_curves(
                train_losses    = train_losses,
                val_losses      = val_losses,
                val_dice_scores = val_dice_scores,
                save_dir        = save_dir,
            )