import os
import json
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from typing import Callable, Optional


def rle_encode(mask: np.ndarray) -> str:
    
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return ""
    pixels = np.ravel(binary, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def save_training_config(
    save_dir: str,
    run_timestamp: str,
    best_model_path: str,
    resize_map: dict,
    device: str,
    model_name: str,
    phase1_epochs: int,
    phase2_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    weight_bce: float,
    weight_dice: float,
    weight_lovasz: float,
    patience: int,
    onecycle_max_lr: float,
    onecycle_pct_start: float,
    onecycle_div_factor: float,
    onecycle_final_div_factor: float,
    onecycle_three_phase: bool,
    swa_start_epoch: int,
    swa_lr: float,
    swa_anneal_epochs: int,
    use_bf16: bool,
    cudnn_benchmark: bool,
    width: int = 72,
) -> None:
    
    # print training configuration
    W   = width
    pad = W - 26  
    
    def row(label: str, value: object) -> str:
        return f"║  {label:<22} {str(value):>{pad}} ║"

    tqdm.write("╔" + "═" * W + "╗")
    tqdm.write("║" + " Training Configuration ".center(W) + "║")
    tqdm.write("╠" + "═" * W + "╣")
    tqdm.write(row("Start Time",          run_timestamp))
    tqdm.write(row("Image Size",          resize_map["IMAGE_SIZE"]))
    tqdm.write(row("Mask Size",           resize_map["MASK_SIZE"]))
    tqdm.write(row("Device",              device))
    tqdm.write(row("Model",               model_name))
    amp_dtype = "bfloat16" if use_bf16 else "float32"
    tqdm.write(row("AMP dtype",           amp_dtype))
    tqdm.write(row("cuDNN benchmark",     cudnn_benchmark))
    tqdm.write("╠" + "─" * W + "╣")
    tqdm.write(row("Epochs",              num_epochs))
    tqdm.write(row("Batch Size",          batch_size))
    tqdm.write(row("Learning Rate",       learning_rate))
    tqdm.write(row("Weight Decay",        weight_decay))
    tqdm.write(row("OneCycle Max LR",     onecycle_max_lr))
    tqdm.write(row("OneCycle Pct Start",  onecycle_pct_start))
    tqdm.write(row("OneCycle Div Factor", onecycle_div_factor))
    tqdm.write(row("OneCycle Final Div",  onecycle_final_div_factor))
    tqdm.write(row("OneCycle 3-Phase",    onecycle_three_phase))
    tqdm.write(row("SWA Start Epoch",     swa_start_epoch))
    tqdm.write(row("SWA LR",              swa_lr))
    tqdm.write(row("SWA Anneal Epochs",   swa_anneal_epochs))
    tqdm.write(row("Early Stop Patience", patience))
    tqdm.write("╠" + "─" * W + "╣")
    tqdm.write(
        row(
            "Loss",
            f"{weight_bce}×BCE + {weight_dice}×Dice + {weight_lovasz}×Lovasz",
        )
    )
    tqdm.write(row("Checkpoint",          os.path.basename(best_model_path)))
    tqdm.write("╚" + "═" * W + "╝")
    tqdm.write("")
    
    # save training configuration as json
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump({
            "Run Timestamp": run_timestamp,
            "Image Size": resize_map["IMAGE_SIZE"],
            "Mask Size": resize_map["MASK_SIZE"],
            "Device": device,
            "Model": model_name,
            "AMP dtype": "bfloat16" if use_bf16 else "float32",
            "use_bf16": use_bf16,
            "cuDNN benchmark": cudnn_benchmark,
            "Epochs": num_epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Weight Decay": weight_decay,
            "OneCycle Max LR": onecycle_max_lr,
            "OneCycle Pct Start": onecycle_pct_start,
            "OneCycle Div Factor": onecycle_div_factor,
            "OneCycle Final Div Factor": onecycle_final_div_factor,
            "OneCycle Three Phase": onecycle_three_phase,
            "SWA Start Epoch": swa_start_epoch,
            "SWA LR": swa_lr,
            "SWA Anneal Epochs": swa_anneal_epochs,
            "Early Stop Patience": patience,
            "Weight BCE": weight_bce,
            "Weight Dice": weight_dice,
            "Weight Lovasz": weight_lovasz,
            "Loss": (
                f"{weight_bce} * BCE + {weight_dice} * Dice + {weight_lovasz} * Lovasz"
            ),
        }, f, indent = 4)



def save_training_results(
    train_losses:    list[float],
    train_dice_scores: list[float],
    val_losses:      list[float],
    val_dice_scores: list[float],
    save_dir:        str = ".",
) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    # save loss curves
    fig1, ax1 = plt.subplots(figsize = (8, 5))
    ax1.plot(epochs, train_losses, color = "steelblue", linewidth=1.5, label = "Train Loss")
    ax1.plot(epochs, val_losses,   color = "tomato",    linewidth=1.5, label = "Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    fig1.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curves.png")
    fig1.savefig(loss_path, dpi = 150)
    print(f"Saved → {loss_path}")

    # save dice scores curves
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, train_dice_scores, color="steelblue", linewidth=1.5, label = "Train Soft Dice")
    ax2.plot(epochs, val_dice_scores, color="seagreen", linewidth=1.5, label = "Val Soft Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Soft Dice")
    ax2.set_title("Training & Validation Soft Dice")
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    fig2.tight_layout()
    dice_path = os.path.join(save_dir, "dice_scores.png")
    fig2.savefig(dice_path, dpi = 150)
    print(f"Saved → {dice_path}")

    # save training results as csv
    with open(os.path.join(save_dir, "training_results.csv"), "w") as f:
        f.write("Epoch,Train Loss,Train Soft Dice,Val Loss,Val Soft Dice\n")
        for i in range(len(epochs)):
            f.write(f"{epochs[i]},{train_losses[i]},{train_dice_scores[i]},{val_losses[i]},{val_dice_scores[i]}\n")
            
    print(f"Saved → {os.path.join(save_dir, 'training_results.csv')}")
