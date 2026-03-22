import os

import matplotlib.pyplot as plt
from tqdm import tqdm


def print_training_config(
    run_timestamp: str,
    best_model_path: str,
    device: str,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dice_alpha: float,
    tv_beta: float,
    patience: int,
    factor: float,
    width: int = 60,
) -> None:
    
    W   = width
    pad = W - 26  
    
    def row(label: str, value: object) -> str:
        return f"║  {label:<22} {str(value):>{pad}} ║"

    tqdm.write("╔" + "═" * W + "╗")
    tqdm.write("║" + " Training Configuration ".center(W) + "║")
    tqdm.write("╠" + "═" * W + "╣")
    tqdm.write(row("Start Time",          run_timestamp))
    tqdm.write(row("Device",              device))
    tqdm.write(row("Model",               model_name))
    tqdm.write("╠" + "─" * W + "╣")
    tqdm.write(row("Epochs",              num_epochs))
    tqdm.write(row("Batch Size",          batch_size))
    tqdm.write(row("Learning Rate",       learning_rate))
    tqdm.write(row("Weight Decay",        weight_decay))
    tqdm.write(row("LR Factor",            factor))
    tqdm.write(row("LR Patience",          patience))
    tqdm.write("╠" + "─" * W + "╣")
    tqdm.write(row("Loss",                f"CE + {dice_alpha}×Dice + {tv_beta}×TV"))
    tqdm.write(row("Early Stop Patience", patience))
    tqdm.write(row("Checkpoint",          os.path.basename(best_model_path)))
    tqdm.write("╚" + "═" * W + "╝")
    tqdm.write("")


def plot_training_curves(
    train_losses:    list[float],
    val_losses:      list[float],
    val_dice_scores: list[float],
    save_dir:        str = ".",
) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, val_losses,   color="tomato",    linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curves.png")
    fig1.savefig(loss_path, dpi=150)
    print(f"Saved → {loss_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, val_dice_scores, color="seagreen", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    dice_path = os.path.join(save_dir, "dice_scores.png")
    fig2.savefig(dice_path, dpi=150)
    print(f"Saved → {dice_path}")

    plt.show()
