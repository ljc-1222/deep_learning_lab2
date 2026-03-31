import argparse
import contextlib
import os
import time
from collections.abc import Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import combined_loss, dice_sum_batch
from src.models.resnet34_unet import ResNet34UNet
from src.models.unet import UNet
from src.oxford_pet import OxfordPetDataset
from src.utils import save_training_config, save_training_results

DEFAULT_MODEL_NAME = "UNet"
DEFAULT_DATASET_ROOT = "dataset/oxford-iiit-pet"
DEFAULT_TRAIN_SPLIT_FILE = "train.txt"
DEFAULT_VAL_SPLIT_FILE = "val.txt"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 500
DEFAULT_WEIGHT_BCE = 0.1
DEFAULT_WEIGHT_DICE = 0.9
DEFAULT_WEIGHT_LOVASZ = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LEARNING_RATE = 1.5e-4
DEFAULT_ONECYCLE_PCT_START = 0.45
DEFAULT_ONECYCLE_DIV_FACTOR = 10.0
DEFAULT_ONECYCLE_FINAL_DIV_FACTOR = 200.0
DEFAULT_ONECYCLE_THREE_PHASE = True
DEFAULT_SWA_START_EPOCH = 300
DEFAULT_SWA_ANNEAL_EPOCHS = 20
DEFAULT_PATIENCE = 50
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)
UNET_SIZE = 388
RESNET34_UNET_SIZE = 384
RESIZE_MAP = {
    "UNet": (UNET_SIZE, UNET_SIZE),
    "ResNet34UNet": (RESNET34_UNET_SIZE, RESNET34_UNET_SIZE),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train binary semantic segmentation model.")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["UNet", "ResNet34UNet"],
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split-file", type=str, default=DEFAULT_TRAIN_SPLIT_FILE)
    parser.add_argument("--val-split-file", type=str, default=DEFAULT_VAL_SPLIT_FILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--weight-bce", type=float, default=DEFAULT_WEIGHT_BCE)
    parser.add_argument("--weight-dice", type=float, default=DEFAULT_WEIGHT_DICE)
    parser.add_argument("--weight-lovasz", type=float, default=DEFAULT_WEIGHT_LOVASZ)
    parser.add_argument("--onecycle-max-lr", type=float, default=None)
    parser.add_argument("--onecycle-pct-start", type=float, default=DEFAULT_ONECYCLE_PCT_START)
    parser.add_argument("--onecycle-div-factor", type=float, default=DEFAULT_ONECYCLE_DIV_FACTOR)
    parser.add_argument(
        "--onecycle-final-div-factor",
        type=float,
        default=DEFAULT_ONECYCLE_FINAL_DIV_FACTOR,
    )
    parser.add_argument(
        "--onecycle-three-phase",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ONECYCLE_THREE_PHASE,
    )
    parser.add_argument("--swa-start-epoch", type=int, default=DEFAULT_SWA_START_EPOCH)
    parser.add_argument("--swa-lr", type=float, default=None)
    parser.add_argument("--swa-anneal-epochs", type=int, default=DEFAULT_SWA_ANNEAL_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return torch.device(device_arg)


def forward_autocast(device: torch.device, use_bf16: bool) -> contextlib.AbstractContextManager:
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_dataloader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    onecycle_scheduler: LRScheduler,
    onecycle_total_steps: int,
    train_batch_step: int,
    use_swa: bool = False,
) -> tuple[float, float, int]:
    loss_sum = 0.0
    soft_dice_sum = 0.0
    n_batches = len(train_dataloader)
    n_samples = len(train_dataloader.dataset)
    model.train()
    train_pbar = tqdm(train_dataloader, desc="Training  ", leave=False)

    for image, mask, _ in train_pbar:
        image = image.to(device)
        mask = mask.to(device)

        with forward_autocast(device=device, use_bf16=use_bf16):
            output = model(image)
            loss_value = loss_fn(output, mask)

        loss_sum += loss_value.item()
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if not use_swa and train_batch_step < onecycle_total_steps:
            onecycle_scheduler.step()
        train_batch_step += 1

        soft_dice_sum += dice_sum_batch(output, mask, soft=True)
        lr_display = optimizer.param_groups[0]["lr"]
        train_pbar.set_postfix({"lr": f"{lr_display:.4g}"})

    mean_loss = loss_sum / n_batches
    mean_soft_dice = soft_dice_sum / n_samples
    return mean_loss, mean_soft_dice, train_batch_step


def validate_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        loss_sum = 0.0
        soft_dice_sum = 0.0
        n_batches = len(val_dataloader)
        n_samples = len(val_dataloader.dataset)
        val_pbar = tqdm(val_dataloader, desc="Validation", leave=False)

        for image, mask, _ in val_pbar:
            image = image.to(device)
            mask = mask.to(device)
            with forward_autocast(device=device, use_bf16=use_bf16):
                output = model(image)
                loss_value = loss_fn(output, mask)
            loss_sum += loss_value.item()

            soft_dice_sum += dice_sum_batch(output, mask, soft=True)
            lr_display = optimizer.param_groups[0]["lr"]
            val_pbar.set_postfix({"lr": f"{lr_display:.4g}"})

    mean_loss = loss_sum / n_batches
    mean_soft_dice = soft_dice_sum / n_samples
    return mean_loss, mean_soft_dice


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    onecycle_max_lr = (
        args.onecycle_max_lr
        if args.onecycle_max_lr is not None
        else 10.0 * args.learning_rate
    )
    swa_lr = args.swa_lr if args.swa_lr is not None else args.learning_rate * 0.5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = OxfordPetDataset(
        root=args.dataset_root,
        split_file=args.train_split_file,
        is_train=True,
        model_name=args.model_name,
        resize_map=RESIZE_MAP,
    )
    val_dataset = OxfordPetDataset(
        root=args.dataset_root,
        split_file=args.val_split_file,
        is_train=False,
        model_name=args.model_name,
        resize_map=RESIZE_MAP,
    )

    use_persistent_workers = args.num_workers > 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )

    onecycle_total_steps = args.swa_start_epoch * len(train_dataloader)

    if args.model_name == "UNet":
        model = UNet().to(device)
    elif args.model_name == "ResNet34UNet":
        model = ResNet34UNet().to(device)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    model = torch.compile(model)

    loss_fn = lambda output, target: combined_loss(
        output,
        target,
        weight0=args.weight_bce,
        weight1=args.weight_dice,
        weight2=args.weight_lovasz,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_epochs=args.swa_anneal_epochs,
    )
    onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=onecycle_max_lr,
        total_steps=onecycle_total_steps,
        pct_start=args.onecycle_pct_start,
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
        three_phase=args.onecycle_three_phase,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("saved_models", args.model_name, run_timestamp)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{args.model_name}.pth")

    save_training_config(
        save_dir=save_dir,
        run_timestamp=run_timestamp,
        best_model_path=best_model_path,
        resize_map=RESIZE_MAP,
        device=str(device),
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weight_bce=args.weight_bce,
        weight_dice=args.weight_dice,
        weight_lovasz=args.weight_lovasz,
        onecycle_max_lr=onecycle_max_lr,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        onecycle_three_phase=args.onecycle_three_phase,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=swa_lr,
        swa_anneal_epochs=args.swa_anneal_epochs,
        patience=args.patience,
        use_bf16=use_bf16,
        cudnn_benchmark=device.type == "cuda" and torch.backends.cudnn.benchmark,
    )

    train_losses: list[float] = []
    train_dice_scores: list[float] = []
    val_losses: list[float] = []
    val_dice_scores: list[float] = []

    best_validation_mean_soft_dice = 0.0
    epochs_without_improvement = 0
    swa_updates = 0
    train_batch_step = 0

    try:
        for epoch in tqdm(range(args.num_epochs)):
            use_swa = (epoch + 1) >= args.swa_start_epoch

            training_loss, training_mean_soft_dice, train_batch_step = train_one_epoch(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_dataloader=train_dataloader,
                device=device,
                use_bf16=use_bf16,
                onecycle_scheduler=onecycle_scheduler,
                onecycle_total_steps=onecycle_total_steps,
                train_batch_step=train_batch_step,
                use_swa=use_swa,
            )
            if use_swa:
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_scheduler.step()
            validation_loss, validation_mean_soft_dice = validate_one_epoch(
                model=model,
                loss_fn=loss_fn,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                device=device,
                use_bf16=use_bf16,
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
                print(
                    f"Best validation soft Dice: {best_validation_mean_soft_dice:.6f} "
                    f"-> Saved best model -> {os.path.basename(best_model_path)}"
                )
            elif use_swa:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    tqdm.write(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best val mean soft Dice: {best_validation_mean_soft_dice:.6f}"
                    )
                    break
    finally:
        if train_losses:
            if swa_updates > 0:
                if args.model_name == "ResNet34UNet":
                    # Recompute BN stats before exporting SWA model.
                    update_bn(train_dataloader, swa_model, device=device)

                swa_model_path = os.path.join(save_dir, f"{args.model_name}_swa.pth")
                torch.save(swa_model.module.state_dict(), swa_model_path)
                tqdm.write(f"Saved SWA model -> {os.path.basename(swa_model_path)}")

            save_training_results(
                train_losses=train_losses,
                train_dice_scores=train_dice_scores,
                val_losses=val_losses,
                val_dice_scores=val_dice_scores,
                save_dir=save_dir,
            )


if __name__ == "__main__":
    main()