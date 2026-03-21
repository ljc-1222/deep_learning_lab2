import os

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
    step_size: int,
    gamma: float,
    dice_alpha: float,
    tv_beta: float,
    patience: int,
    width: int = 60,
) -> None:
    W   = width
    pad = W - 25

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
    tqdm.write(row("LR Scheduler",        f"StepLR(step={step_size}, γ={gamma})"))
    tqdm.write("╠" + "─" * W + "╣")
    tqdm.write(row("Loss",                f"CE + {dice_alpha}×Dice + {tv_beta}×TV"))
    tqdm.write(row("Early Stop Patience", patience))
    tqdm.write(row("Checkpoint",          os.path.basename(best_model_path)))
    tqdm.write("╚" + "═" * W + "╝")
    tqdm.write("")
