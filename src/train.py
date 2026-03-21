import os
import time
import torch
import numpy as np
import torchvision.transforms.functional as F

from tqdm import tqdm
from src.models.unet import UNet
from torch.utils.data import DataLoader
from src.utils import print_training_config
from src.evaluate import dice_score, combined_loss

# Device and number of workers
NUM_WORKERS = min(4, os.cpu_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
SEED = 42

# Dataset hyperparameters
BATCH_SIZE = 24

# Training hyperparameters
NUM_EPOCHS  = 250
DICE_WEIGHT = 1.3 
TV_WEIGHT   = 0.4

# Optimizer hyperparameters
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3

# Scheduler hyperparameters
GAMMA = 0.1
STEP_SIZE = 10

# Early stopping hyperparameters
PATIENCE = 40
FACTOR = 0.5

# Set seed
torch.manual_seed(SEED)

train_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                                 split_file = "train.txt", 
                                 is_train = True)

val_dataset = OxfordPetDataset(root = "dataset/oxford-iiit-pet", 
                               split_file = "val.txt", 
                               is_train = False)

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

model  = torch.compile(UNet().to(DEVICE))

loss = lambda output, target: combined_loss(output, target, alpha = DICE_WEIGHT, beta = TV_WEIGHT)

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = FACTOR, patience = PATIENCE)

def train_one_epoch(model, optimizer, scheduler, loss, train_dataloader):
    
    training_loss = 0
    model.train()
    train_pbar = tqdm(train_dataloader, desc = "Training  ")
    
    for batch_idx, (image, trimap, _) in enumerate(train_pbar):
        
        image = image.to(DEVICE)
        trimap = trimap.to(DEVICE)

        output = model(image)
        loss_value = loss(output, trimap)

        training_loss += loss_value.item()
        train_pbar.set_postfix({'loss': f"{loss_value.item():.4f}"})
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()
        
    scheduler.step(epoch_dice_score)
        
    training_loss = training_loss / len(train_dataloader)
        
    return training_loss
        
def validate_one_epoch(model, loss, val_dataloader):
    
    model.eval()
    
    with torch.no_grad():
    
        validation_loss = 0
        epoch_dice_score = 0
        val_pbar = tqdm(val_dataloader, desc = "Validation")
    
        for image, trimap, idxs in val_pbar:
            
            image = image.to(DEVICE)
            trimap = trimap.to(DEVICE)
            output = model(image)
            loss_value = loss(output, trimap)
            validation_loss += loss_value.item()
            
            for i in range(len(idxs)):
                idx = idxs[i]
                size_i = val_dataset.load_image(idx).size[::-1]
                output_i = F.resize(output[i], size = size_i).squeeze(0)
                output_i = torch.argmax(output_i, dim = 0).float()
                trimap_i = val_dataset.load_trimap(idx)
                trimap_i = np.array(trimap_i)
                trimap_i = (trimap_i == 1).astype(np.uint8)
                trimap_i = torch.from_numpy(trimap_i).float().to(DEVICE)
                epoch_dice_score += dice_score(output_i, trimap_i)
                
            val_pbar.set_postfix({'loss': f"{loss_value.item():.4f}"})
        
    validation_loss = validation_loss / len(val_dataloader)
    epoch_dice_score = epoch_dice_score / len(val_dataset)
    
    return validation_loss, epoch_dice_score

if __name__ == "__main__":

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir      = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_unet_{run_timestamp}.pth")

    print_training_config(
        run_timestamp   = run_timestamp,
        best_model_path = best_model_path,
        device          = str(DEVICE),
        model_name      = "UNet",
        num_epochs      = NUM_EPOCHS,
        batch_size      = BATCH_SIZE,
        learning_rate   = LEARNING_RATE,
        weight_decay    = WEIGHT_DECAY,
        step_size       = STEP_SIZE,
        gamma           = GAMMA,
        dice_alpha      = DICE_WEIGHT,
        tv_beta         = TV_WEIGHT,
        patience        = PATIENCE,
    )
    
    training_losses = []
    validation_losses = []
    dice_scores = []
    
    best_validation_dice_score = -1.0
    epochs_without_improvement = 0
    
    for epoch in tqdm(range(NUM_EPOCHS)):
        
        # Training
        training_loss = train_one_epoch(model, optimizer, scheduler, loss, train_dataloader)

        # Validation
        validation_loss, epoch_dice_score = validate_one_epoch(model, loss, val_dataloader)
                
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        dice_scores.append(epoch_dice_score)
        
        tqdm.write(f"Epoch {epoch+1:3d} | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f} | Val Dice: {epoch_dice_score:.4f}")
                
        if epoch_dice_score > best_validation_dice_score:
            best_validation_dice_score = epoch_dice_score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                tqdm.write(f"Early stopping at epoch {epoch+1}. Best validation dice score: {best_validation_dice_score:.4f}")
                break