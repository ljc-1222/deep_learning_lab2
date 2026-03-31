import os
import time
import contextlib
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.oxford_pet import OxfordPetDataset
from src.models.resnet34_unet import ResNet34UNet
from src.evaluate import combined_loss, dice_sum_batch
from src.utils import save_training_config, save_training_results

# Model name
MODEL_NAME = "UNet"
# MODEL_NAME = "ResNet34UNet"

# Device and number of workers
NUM_WORKERS = min(8, os.cpu_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _forward_autocast() -> contextlib.AbstractContextManager:

    if USE_BF16:
        return torch.autocast(device_type = "cuda", dtype = torch.bfloat16)
    return contextlib.nullcontext()

# Seed
SEED = 42

# Dataset hyperparameters
BATCH_SIZE = 16

# Training hyperparameters
NUM_EPOCHS  = 400
COMBINED_LOSS_WEIGHT = 0.4

# Optimizer hyperparameters
WEIGHT_DECAY  = 5e-4
LEARNING_RATE = 1e-5
ETA_MIN_RATE  = 1e-3

# Early stopping hyperparameters
PATIENCE = 40

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

train_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                                 split_file = "train.txt", 
                                 is_train = True,
                                 model_name = MODEL_NAME
                                 )

val_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                               split_file = "val.txt", 
                               is_train = False,
                               model_name = MODEL_NAME)

train_dataloader = DataLoader(train_dataset, 
                                   batch_size = BATCH_SIZE, 
                                   shuffle = True, 
                                   num_workers = NUM_WORKERS,
                                   pin_memory = True,
                                   persistent_workers = True)

val_dataloader = DataLoader(val_dataset, 
                                 batch_size = BATCH_SIZE, 
                                 shuffle = False, 
                                 num_workers = NUM_WORKERS,
                                 pin_memory = True,
                                 persistent_workers = True)

# Scheduler hyperparameters (warmup/cosine lengths are in epoch-equivalents; step() once per train batch)
WARMUP_EPOCHS = 20
WARMUP_FACTOR = 0.1

STEPS_PER_EPOCH = len(train_dataloader)
WARMUP_STEPS = WARMUP_EPOCHS * STEPS_PER_EPOCH
COSINE_STEPS = (NUM_EPOCHS - WARMUP_EPOCHS) * STEPS_PER_EPOCH

if MODEL_NAME == "UNet":
    model = UNet().to(DEVICE)
elif MODEL_NAME == "ResNet34UNet":
    model = ResNet34UNet().to(DEVICE)
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

model = torch.compile(model)

loss = lambda output, target: combined_loss(output, target, weight = COMBINED_LOSS_WEIGHT)

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor = WARMUP_FACTOR, end_factor = 1.0, total_iters = WARMUP_STEPS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = COSINE_STEPS, eta_min = ETA_MIN_RATE * LEARNING_RATE
)

# SequentialLR calls child schedulers with step(0) at milestones, which triggers PyTorch's
# epoch-deprecation warning; stepping warmup then cosine manually avoids that.
_train_batch_step = 0


def train_one_epoch(model, optimizer, loss, train_dataloader) -> tuple[float, float]:

    loss_sum = 0.0
    soft_dice_sum = 0.0
    n_batches = len(train_dataloader)
    n_samples = len(train_dataloader.dataset)
    model.train()
    train_pbar = tqdm(train_dataloader, desc = "Training  ", leave = False)

    global _train_batch_step

    for image, mask, _ in train_pbar:

        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        with _forward_autocast():
            output = model(image)
            loss_value = loss(output, mask)

        loss_sum += loss_value.item()
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()
        if _train_batch_step < WARMUP_STEPS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        _train_batch_step += 1
        
        soft_dice_sum += dice_sum_batch(output, mask, soft = True)
        lr_display = optimizer.param_groups[0]["lr"]
        train_pbar.set_postfix({'lr': f"{lr_display:.4g}"})

    mean_loss = loss_sum / n_batches
    mean_soft_dice = soft_dice_sum / n_samples

    return mean_loss, mean_soft_dice


def validate_one_epoch(model, loss, val_dataloader) -> tuple[float, float]:
    
    model.eval()
    
    with torch.no_grad():
    
        loss_sum = 0.0
        soft_dice_sum = 0.0
        n_batches = len(val_dataloader)
        n_samples = len(val_dataloader.dataset)
        val_pbar = tqdm(val_dataloader, desc = "Validation", leave = False)
    
        for image, mask, _ in val_pbar:
            
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            with _forward_autocast():
                output = model(image)
                loss_value = loss(output, mask)
            loss_sum += loss_value.item()
            
            soft_dice_sum += dice_sum_batch(output, mask, soft = True)
            lr_display = optimizer.param_groups[0]["lr"]
            val_pbar.set_postfix({'lr': f"{lr_display:.4g}"})
        
    mean_loss = loss_sum / n_batches
    mean_soft_dice = soft_dice_sum / n_samples

    return mean_loss, mean_soft_dice


if __name__ == "__main__":

    torch.cuda.empty_cache()

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir      = os.path.join("saved_models", MODEL_NAME, run_timestamp)
    os.makedirs(save_dir, exist_ok = True)
    best_model_path = os.path.join(save_dir, f"{MODEL_NAME}.pth")

    save_training_config(
        save_dir        = save_dir,
        run_timestamp   = run_timestamp,
        best_model_path = best_model_path,
        device          = str(DEVICE),
        model_name      = MODEL_NAME,
        num_epochs      = NUM_EPOCHS,
        batch_size      = BATCH_SIZE,
        learning_rate   = LEARNING_RATE,
        weight_decay    = WEIGHT_DECAY,
        combined_loss_weight = COMBINED_LOSS_WEIGHT,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_factor   = WARMUP_FACTOR,
        eta_min_rate    = ETA_MIN_RATE,
        patience        = PATIENCE,
        use_bf16        = USE_BF16,
        cudnn_benchmark = torch.cuda.is_available() and torch.backends.cudnn.benchmark,
    )
    
    train_losses:      list[float] = []
    train_dice_scores: list[float] = []
    val_losses:        list[float] = []
    val_dice_scores:   list[float] = []

    best_validation_mean_soft_dice = 0.0
    epochs_without_improvement = 0

    try:
        for epoch in tqdm(range(NUM_EPOCHS)):

            training_loss, training_mean_soft_dice = train_one_epoch(
                model, optimizer, loss, train_dataloader
            )
            validation_loss, validation_mean_soft_dice = validate_one_epoch(
                model, loss, val_dataloader
            )

            train_losses.append(training_loss)
            train_dice_scores.append(training_mean_soft_dice)
            val_losses.append(validation_loss)
            val_dice_scores.append(validation_mean_soft_dice)

            tqdm.write(
                f"Epoch {epoch + 1:3d} |"
                f"Train Loss: {training_loss:.6f} | "
                f"Train Soft Dice: {training_mean_soft_dice:.6f} | "
                f"Val Loss: {validation_loss:.6f} | "
                f"Val Soft Dice: {validation_mean_soft_dice:.6f}"
            )

            if validation_mean_soft_dice > best_validation_mean_soft_dice:
                best_validation_mean_soft_dice = validation_mean_soft_dice
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Best validation soft Dice: {best_validation_mean_soft_dice:.6f} → Saved best model → {os.path.basename(best_model_path)}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    tqdm.write(
                        f"Early stopping at epoch {epoch+1}. Best val mean soft Dice: "
                        f"{best_validation_mean_soft_dice:.6f}"
                    )
                    break

    finally:
        if train_losses:
            save_training_results(
                train_losses      = train_losses,
                train_dice_scores = train_dice_scores,
                val_losses        = val_losses,
                val_dice_scores   = val_dice_scores,
                save_dir          = save_dir,
            )