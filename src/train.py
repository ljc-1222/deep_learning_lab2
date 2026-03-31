import os
import time
import contextlib
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from src.models.unet import UNet
from src.oxford_pet import OxfordPetDataset
from src.models.resnet34_unet import ResNet34UNet
from src.evaluate import combined_loss, dice_sum_batch
from src.utils import save_training_config, save_training_results

# Model name
# MODEL_NAME = "UNet"
MODEL_NAME = "ResNet34UNet"

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
BATCH_SIZE = 32

# Image size
RESIZE_MAP = {
    "UNet": {
        "IMAGE_SIZE": (524, 524),
        "MASK_SIZE": (524 - 184, 524 - 184),
    },
    "ResNet34UNet": {
        "IMAGE_SIZE": (512, 512),
        "MASK_SIZE": (512, 512),
    }
}

# Training hyperparameters
NUM_EPOCHS  = 500
WEIGHT_BCE = 0.1
WEIGHT_DICE = 0.9
WEIGHT_LOVAZ = 0.001
SWA_START_EPOCH = 300

# Optimizer hyperparameters
WEIGHT_DECAY  = 1e-4
LEARNING_RATE = 1.5 * 1e-4

# OneCycleLR hyperparameters
ONECYCLE_MAX_LR = 10 * LEARNING_RATE
ONECYCLE_PCT_START = 0.45
ONECYCLE_DIV_FACTOR = 10
ONECYCLE_FINAL_DIV_FACTOR = 200
ONECYCLE_THREE_PHASE = True
 
# SWA LR scheduler hyperparameters
SWA_LR = LEARNING_RATE * 0.5
SWA_ANNEAL_EPOCHS = 20

# Early stopping hyperparameters
PATIENCE = 50

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

train_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                                 split_file = "train.txt", 
                                 is_train = True,
                                 model_name = MODEL_NAME,
                                 resize_map = RESIZE_MAP
                                 )

val_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                               split_file = "val.txt", 
                               is_train = False,
                               model_name = MODEL_NAME,
                               resize_map = RESIZE_MAP
                               )

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

STEPS_PER_EPOCH = len(train_dataloader)
ONECYCLE_TOTAL_STEPS = SWA_START_EPOCH * STEPS_PER_EPOCH

if MODEL_NAME == "UNet":
    model = UNet().to(DEVICE)
elif MODEL_NAME == "ResNet34UNet":
    model = ResNet34UNet().to(DEVICE)
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

model = torch.compile(model)

loss = lambda output, target: combined_loss(output, target, weight0 = WEIGHT_BCE, weight1 = WEIGHT_DICE, weight2 = WEIGHT_LOVAZ)

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr = SWA_LR, anneal_epochs = SWA_ANNEAL_EPOCHS)

onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = ONECYCLE_MAX_LR,
    total_steps = ONECYCLE_TOTAL_STEPS,
    pct_start = ONECYCLE_PCT_START,
    div_factor = ONECYCLE_DIV_FACTOR,
    final_div_factor = ONECYCLE_FINAL_DIV_FACTOR,
    three_phase = ONECYCLE_THREE_PHASE,
)

# OneCycleLR is stepped per train batch before SWA starts.
# SWALR is stepped once per epoch during SWA phase.
_train_batch_step = 0


def train_one_epoch(
    model,
    optimizer,
    loss,
    train_dataloader,
    use_swa: bool = False,
) -> tuple[float, float]:

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        if not use_swa and _train_batch_step < ONECYCLE_TOTAL_STEPS:
            onecycle_scheduler.step()
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
        resize_map      = RESIZE_MAP[MODEL_NAME],
        device          = str(DEVICE),
        model_name      = MODEL_NAME,
        num_epochs      = NUM_EPOCHS,
        batch_size      = BATCH_SIZE,
        learning_rate   = LEARNING_RATE,
        weight_decay    = WEIGHT_DECAY,
        weight_bce      = WEIGHT_BCE,
        weight_dice     = WEIGHT_DICE,
        weight_lovasz   = WEIGHT_LOVAZ,
        onecycle_max_lr = ONECYCLE_MAX_LR,
        onecycle_pct_start = ONECYCLE_PCT_START,
        onecycle_div_factor = ONECYCLE_DIV_FACTOR,
        onecycle_final_div_factor = ONECYCLE_FINAL_DIV_FACTOR,
        onecycle_three_phase = ONECYCLE_THREE_PHASE,
        swa_start_epoch = SWA_START_EPOCH,
        swa_lr = SWA_LR,
        swa_anneal_epochs = SWA_ANNEAL_EPOCHS,
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
    swa_updates = 0

    try:
        for epoch in tqdm(range(NUM_EPOCHS)):
            use_swa = (epoch + 1) >= SWA_START_EPOCH

            training_loss, training_mean_soft_dice = train_one_epoch(
                model, optimizer, loss, train_dataloader, use_swa = use_swa
            )
            if use_swa:
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_scheduler.step()
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
                if use_swa:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= PATIENCE:
                        tqdm.write(
                            f"Early stopping at epoch {epoch+1}. Best val mean soft Dice: "
                            f"{best_validation_mean_soft_dice:.6f}"
                        )
                        break

    finally:
        if train_losses:
            if swa_updates > 0:
                if MODEL_NAME == "ResNet34UNet":  # Recompute BN running stats before exporting SWA model.
                    update_bn(train_dataloader, swa_model, device = DEVICE)

                swa_model_path = os.path.join(save_dir, f"{MODEL_NAME}_swa.pth")
                torch.save(swa_model.module.state_dict(), swa_model_path)
                tqdm.write(f"Saved SWA model → {os.path.basename(swa_model_path)}")

            save_training_results(
                train_losses      = train_losses,
                train_dice_scores = train_dice_scores,
                val_losses        = val_losses,
                val_dice_scores   = val_dice_scores,
                save_dir          = save_dir,
            )