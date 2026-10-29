import torch
import torch.nn as nn
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize, cross_entropy
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler  # Mixed-precision training.
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
from src.multiclass_head import MulticlassHead
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import numpy as np
import random
import os
from datetime import datetime
import json
from PIL import Image

# Configure the PyTorch allocator to reduce memory fragmentation.
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Fix random seeds to improve reproducibility.
DEFAULT_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================== Configuration ====================
# Adjust these parameters for your environment.
CONFIG_FILE = "./config.yaml"  # Configuration file path.
OUTPUT_DIR = "./training_results"  # Training output directory.
SAVE_CHECKPOINT_EVERY = 5  # Save a checkpoint every N epochs.
EARLY_STOPPING_PATIENCE = 5  # Stop after this many epochs without validation loss improvement.
RESUME_CHECKPOINT = None  # Checkpoint path to resume from; None starts a new run.
# RESUME_CHECKPOINT = "./training_results/train_20260117_164947/checkpoints/emergency_checkpoint.pth"  # Uncomment and set the path to resume training.

# Loss configuration for boundary accuracy.
USE_CE_LOSS = True   # Enable CE loss to provide pixelwise classification gradients.
DICE_WEIGHT = 1.0    # Dice loss weight for boundary accuracy.
CE_WEIGHT = 1.0      # CE loss weight for pixelwise classification.
# ==================================================

# Load the configuration file.
with open(CONFIG_FILE, "r", encoding='utf-8') as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Read the random seed from the configuration or use the default.
random_seed = int(config_file.get("TRAIN", {}).get("SEED", DEFAULT_SEED))
set_seed(random_seed)
print(f"Random seed: {random_seed}")

# Create the output directory.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(OUTPUT_DIR, f"train_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val_images"), exist_ok=True)

# Save the configuration.
with open(os.path.join(output_dir, "config.yaml"), "w", encoding='utf-8') as f:
    yaml.dump(config_file, f, allow_unicode=True)

# Dataset paths.
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
val_dataset_path = config_file["DATASET"].get("VAL_PATH", None)
if val_dataset_path is None:
    # Try to infer the validation dataset path.
    if "train" in train_dataset_path:
        val_dataset_path = train_dataset_path.replace("train", "val")
    else:
        val_dataset_path = None

# Load SAM.
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
# Add LoRA to SAM.
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam

# Add the multiclass classification head.
multiclass_head = MulticlassHead(num_classes=4, in_channels=1)
# Use the second visible GPU.
multiclass_head = multiclass_head.to("cuda:1" if torch.cuda.is_available() else "cpu")

# Load datasets.
processor = Samprocessor(model)
try:
    train_ds = DatasetSegmentation(config_file, processor, mode="train")
    if len(train_ds) == 0:
        raise ValueError("The training set is empty. Check the dataset path and file formats.")
except Exception as e:
    print(f"Error: Could not load the training set")
    print(f"  Training set path: {train_dataset_path}")
    print(f"  Error details: {str(e)}")
    raise

# Check whether the validation dataset path exists.
val_ds = None
if val_dataset_path and os.path.exists(val_dataset_path):
    try:
        val_ds = DatasetSegmentation(config_file, processor, mode="val")
        if len(val_ds) == 0:
            print(f"Warning: The validation set is empty; using the training set only")
            val_ds = None
    except Exception as e:
        print(f"Warning: Could not load validation set {val_dataset_path}; using the training set only")
        print(f"  Error details: {str(e)}")
        val_ds = None

# Create data loaders.
# Ensure batch_size is an integer.
batch_size = config_file["TRAIN"]["BATCH_SIZE"]
if isinstance(batch_size, str):
    batch_size = int(batch_size)
else:
    batch_size = int(batch_size)

# Gradient accumulation reduces GPU memory requirements.
gradient_accumulation_steps = config_file["TRAIN"].get("GRADIENT_ACCUMULATION_STEPS", 1)
if isinstance(gradient_accumulation_steps, str):
    gradient_accumulation_steps = int(gradient_accumulation_steps)
else:
    gradient_accumulation_steps = int(gradient_accumulation_steps)
print(f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch_size = {batch_size * gradient_accumulation_steps})")

data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(random_seed)

train_dataloader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    worker_init_fn=seed_worker,
    generator=data_loader_generator
)

if val_ds is not None:
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=data_loader_generator
    )

# Initialize the optimizer and loss functions.
# Optimize SAM LoRA parameters and the multiclass head together.
# Ensure learning rate and weight decay are numeric.
learning_rate = config_file["TRAIN"].get("LEARNING_RATE", 1e-4)
if isinstance(learning_rate, str):
    # Parse scientific-notation strings such as "1e-4".
    learning_rate = float(learning_rate)
else:
    learning_rate = float(learning_rate)

weight_decay = config_file["TRAIN"].get("WEIGHT_DECAY", 0)
if isinstance(weight_decay, str):
    weight_decay = float(weight_decay)
else:
    weight_decay = float(weight_decay)

optimizer = Adam(
    list(model.image_encoder.parameters()) + list(multiclass_head.parameters()), 
    lr=learning_rate, 
    weight_decay=weight_decay
)

# Compute class weights to address class imbalance.
def compute_class_weights(dataset, num_classes=4):
    """
    Compute per-class weights to address class imbalance.
    
    Arguments:
        dataset: Training dataset.
        num_classes: Number of classes.
    
    Return:
        class_weights: List of per-class weights.
    """
    print("Computing class weights...")
    class_counts = {i: 0 for i in range(num_classes)}
    total_pixels = 0
    
    # Use at most the first 1000 samples for large datasets.
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="Computing class distribution"):
        try:
            sample = dataset[idx]
            mask = sample["ground_truth_mask"].numpy()
            for class_id in range(num_classes):
                count = np.sum(mask == class_id)
                class_counts[class_id] += count
                total_pixels += count
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue
    
    # Use moderated inverse-frequency weights to preserve boundary accuracy.
    # weight = total_pixels / (num_classes * class_count)
    class_weights = []
    print("\nClass distribution:")
    for class_id in range(num_classes):
        count = class_counts[class_id]
        if count > 0:
            # Apply inverse-frequency weighting while avoiding excessively large weights.
            weight = total_pixels / (num_classes * count)
            # Clip weights to [0.5, 5.0] to preserve boundary accuracy.
            # The previous range was [0.1, 10.0]; the reduced upper bound limits excessive penalties.
            weight = max(0.5, min(5.0, weight))
        else:
            weight = 1.0
        class_weights.append(weight)
        class_name = utils.TAMPER_NAMES.get(class_id, f"Class {class_id}")
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {class_name} (class {class_id}): {count} pixels ({percentage:.2f}%), weight={weight:.4f}")
    
    return class_weights

# Ensure the epoch count is an integer.
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
if isinstance(num_epochs, str):
    num_epochs = int(num_epochs)
else:
    num_epochs = int(num_epochs)
# Use the second visible GPU.
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Compute class weights after defining the device.
class_weights = compute_class_weights(train_ds, num_classes=4)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\nClass weight tensor: {weights_tensor}")

# Multiclass loss combines Dice for boundaries with CE for pixelwise classification.
# Configure the CE contribution through CE_WEIGHT.
ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
# DiceLoss: to_onehot_y=True converts labels to one-hot; softmax=True applies softmax to predictions.
# include_background=True includes the background class.
# Although background covers over 90%, CE class weights address imbalance; including background in Dice can:
# 1. Provide more stable gradients.
# 2. Help distinguish background from tampered regions.
# 3. Complement CE classification with boundary matching.
dice_loss = monai.losses.DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
# Set loss weights from the configuration above.
dice_weight = DICE_WEIGHT
ce_weight = CE_WEIGHT if USE_CE_LOSS else 0.0
print(f"\nLoss configuration:")
if USE_CE_LOSS:
    print(f"  Dice loss weight: {dice_weight} (boundary matching)")
    print(f"  CE loss weight: {ce_weight} (pixelwise classification)")
    print(f"  Total loss = {dice_weight} * Dice + {ce_weight} * CE")
else:
    print(f"  Using Dice loss only (weight: {dice_weight}) for boundary matching, as in the original code")
    print(f"  Total loss = {dice_weight} * Dice")

# Initialize the mixed-precision gradient scaler to reduce memory use.
use_amp = True  # Enable mixed-precision training to reduce memory use.
scaler = GradScaler() if use_amp else None
if use_amp:
    print("✓ Enabled automatic mixed precision (AMP) to reduce GPU memory use")

# Set training mode and move models to the device.
model.train()
model.to(device)
multiclass_head.train()

# ==================== Helper functions ====================
def visualize_predictions(model, multiclass_head, dataset, processor, device, save_dir, prefix, num_samples=4):
    """Visualize predictions."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    multiclass_head.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        image_path = dataset.img_files[idx]
        mask_path = dataset.mask_files[idx]
        
        # Load the original image and ground-truth mask.
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask_array = np.array(mask)
        gt_class_mask = utils.rgb_mask_to_class_mask(mask_array)
        
        # Predict.
        with torch.no_grad():
            batch = [sample]
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            class_logits = multiclass_head(stk_out)
            pred_class_mask = torch.argmax(class_logits, dim=1)[0].cpu().numpy()
            
            # Resize to the original image size.
            original_size = sample["original_size"]
            pred_class_mask = F.interpolate(
                torch.from_numpy(pred_class_mask).unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode='nearest'
            )[0, 0].numpy().astype(np.int64)
        
        # Visualize.
        image_array = np.array(image)
        filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
        utils.plot_multiclass_prediction(
            image_array, 
            pred_class_mask, 
            gt_class_mask,
            f"{prefix}_{filename}.png",
            save_dir
        )


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves.
    axes[0].plot(history["epoch"], history["train_loss"], label="Training loss", marker='o')
    if any(v > 0 for v in history["val_loss"]):
        axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score curves.
    axes[1].plot(history["epoch"], history["train_dice"], label="Training Dice", marker='o')
    if any(v > 0 for v in history["val_dice"]):
        axes[1].plot(history["epoch"], history["val_dice"], label="Validation Dice", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice score")
    axes[1].set_title("Training and validation Dice scores")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==================== Training history ====================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_dice": [],
    "val_dice": [],
    "epoch": []
}

# Early stopping.
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# Resume training if a checkpoint is specified.
start_epoch = 0
if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"\nResuming training from checkpoint: {RESUME_CHECKPOINT}")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    multiclass_head.load_state_dict(checkpoint['multiclass_head_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'history' in checkpoint:
        history = checkpoint['history']
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    if 'patience_counter' in checkpoint:
        patience_counter = checkpoint.get('patience_counter', 0)
    print(f"✓ Restored training state: starting from epoch {start_epoch}")
    print(f"  Training history: {len(history['epoch'])} epoch records")
    if checkpoint.get('is_emergency', False):
        print(f"  Note: This is an emergency checkpoint; check the training state")

# Emergency checkpoint saving for OOM and similar failures.
def save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir):
    """Save an emergency checkpoint to preserve training progress."""
    emergency_path = os.path.join(output_dir, "checkpoints", "emergency_checkpoint.pth")
    try:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'multiclass_head_state_dict': multiclass_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'is_emergency': True,
        }, emergency_path)
        print(f"✓ Saved emergency checkpoint: {emergency_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to save emergency checkpoint: {e}")
        return False

print(f"Starting training for {num_epochs} epochs")
if start_epoch > 0:
    print(f"Resuming from epoch {start_epoch}")
print(f"Training set size: {len(train_ds)}")
if val_ds is not None:
    print(f"Validation set size: {len(val_ds)}")
print(f"Device: {device}")
print(f"Output directory: {output_dir}")

for epoch in range(start_epoch, num_epochs):
    # ========== Training phase ==========
    model.train()
    multiclass_head.train()
    epoch_train_losses = []
    epoch_train_dice_scores = []

    # Clear gradients at the start of each epoch.
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
        try:
            # Clear the cache.
            torch.cuda.empty_cache()
            
            # Run the forward pass with mixed precision.
            with autocast(enabled=use_amp):
                outputs = model(batched_input=batch, multimask_output=False)
                
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                # Ensure tensors are on the correct device.
                stk_out = stk_out.to(device)
                stk_gt = stk_gt.to(device)
                # stk_out: [B, 1, H, W]
                # stk_gt: [B, H, W] class-index masks with values from 0 to 3.
                
                # Delete unused outputs to free memory.
                del outputs
                torch.cuda.empty_cache()
                
                # Apply the multiclass classification head.
                class_logits = multiclass_head(stk_out)  # [B, 4, H, W]
                
                # Resize ground truth to the low-resolution output size.
                B, C, H, W = class_logits.shape
                stk_gt_resized = F.interpolate(
                    stk_gt.unsqueeze(1).float(), 
                    size=(H, W), 
                    mode='nearest'
                ).squeeze(1).long()
                
                # Compute the loss.
                # Dice loss emphasizes boundary and shape matching.
                dice = dice_loss(class_logits, stk_gt_resized.unsqueeze(1))  # [B, 1, H, W]
                
                # Combine losses.
                if USE_CE_LOSS:
                    # CE loss provides pixelwise classification gradients.
                    ce = ce_loss(class_logits, stk_gt_resized)
                    loss = dice_weight * dice + ce_weight * ce
                else:
                    # Use Dice loss only for boundary accuracy, as in the original code.
                    loss = dice_weight * dice
                    ce = torch.tensor(0.0, device=device)  # Logging only; does not affect training.
                
                # Compute Dice for monitoring inside autocast with no_grad.
                with torch.no_grad():
                    pred_classes = torch.argmax(class_logits, dim=1)
                    dice_score = utils.compute_multiclass_dice(pred_classes.cpu(), stk_gt_resized.cpu(), num_classes=4)
                    epoch_train_dice_scores.append(float(dice_score))
                    del pred_classes
                    torch.cuda.empty_cache()  # Release temporary memory used for the Dice score.
            
            # Delete unused intermediate tensors before backpropagation.
            del stk_out, class_logits  # Free GPU memory early.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for pending CUDA operations to finish.
            
            # Backpropagate with mixed precision and gradient accumulation.
            # Divide the loss by the accumulation steps to average gradients across batches.
            loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update parameters only after reaching the accumulation step count.
            if (i + 1) % gradient_accumulation_steps == 0:
                # Clear the cache before the parameter update to free more memory.
                torch.cuda.empty_cache()
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # Clear the cache immediately after the update.
                torch.cuda.empty_cache()
            # Otherwise, keep accumulating gradients without updating parameters.

            # Record the loss scaled for gradient accumulation, as in train_20260408_132355.
            loss_value = loss.item()
            epoch_train_losses.append(loss_value)
            
            # Delete intermediate tensors; the Dice score was already computed inside autocast.
            # Note: stk_out and class_logits were deleted before backpropagation.
            del stk_gt, stk_gt_resized, ce, dice, loss
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ Error: GPU out of memory (batch {i+1}, epoch {epoch+1})")
            print(f"Suggestions: 1) Wait for other processes to release GPU memory 2) Increase GRADIENT_ACCUMULATION_STEPS 3) Reduce batch_size")
            print(f"Current configuration: batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
            torch.cuda.empty_cache()
            
            # Try to save an emergency checkpoint.
            print("\nAttempting to save an emergency checkpoint...")
            save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir)
            
            print("\nExiting. Resolve the GPU memory issue before restarting training.")
            print("If available, resume from training_results/*/checkpoints/emergency_checkpoint.pth.")
            raise
    
    avg_train_loss = mean(epoch_train_losses)
    # Use numpy.mean or ensure all values have the same numeric type.
    if epoch_train_dice_scores:
        avg_train_dice = float(np.mean(epoch_train_dice_scores))
    else:
        avg_train_dice = 0.0
    
    # ========== Validation phase ==========
    if val_ds is not None:
        model.eval()
        multiclass_head.eval()
        epoch_val_losses = []
        epoch_val_dice_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                try:
                    # Clear the cache.
                    torch.cuda.empty_cache()
                    
                    # Use mixed precision during validation to reduce memory use.
                    with autocast(enabled=use_amp):
                        outputs = model(batched_input=batch, multimask_output=False)
                        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                        # Ensure tensors are on the correct device.
                        stk_out = stk_out.to(device)
                        stk_gt = stk_gt.to(device)
                        
                        # Delete unused outputs.
                        del outputs
                        torch.cuda.empty_cache()
                        
                        class_logits = multiclass_head(stk_out)
                        
                        B, C, H, W = class_logits.shape
                        stk_gt_resized = F.interpolate(
                            stk_gt.unsqueeze(1).float(), 
                            size=(H, W), 
                            mode='nearest'
                        ).squeeze(1).long()
                        
                        # Dice loss emphasizes boundary and shape matching.
                        dice = dice_loss(class_logits, stk_gt_resized.unsqueeze(1))  # [B, 1, H, W]
                        
                        # Combine losses.
                        if USE_CE_LOSS:
                            # Auxiliary CE loss.
                            ce = ce_loss(class_logits, stk_gt_resized)
                            loss = dice_weight * dice + ce_weight * ce
                        else:
                            # Use Dice loss only.
                            loss = dice_weight * dice
                            ce = torch.tensor(0.0, device=device)  # Logging only.
                        
                        # Compute the Dice score.
                        pred_classes = torch.argmax(class_logits, dim=1)
                        dice_score = utils.compute_multiclass_dice(pred_classes.cpu(), stk_gt_resized.cpu(), num_classes=4)
                        epoch_val_dice_scores.append(float(dice_score))
                    
                    epoch_val_losses.append(loss.item())
                    
                    # Delete intermediate tensors.
                    del stk_out, stk_gt, class_logits, stk_gt_resized, pred_classes
                    del ce, dice, loss
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n❌ Error: GPU out of memory (validation batch {i+1}, epoch {epoch+1})")
                    torch.cuda.empty_cache()
                    
                    # Try to save an emergency checkpoint.
                    print("\nAttempting to save an emergency checkpoint...")
                    save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir)
                    
                    print("\nExiting. Resolve the GPU memory issue before restarting training.")
                    raise
        
        avg_val_loss = mean(epoch_val_losses)
        # Use numpy.mean or ensure all values have the same numeric type.
        if epoch_val_dice_scores:
            avg_val_dice = float(np.mean(epoch_val_dice_scores))
        else:
            avg_val_dice = 0.0
    else:
        avg_val_loss = 0.0
        avg_val_dice = 0.0
    
    # Record training history.
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_dice"].append(avg_train_dice)
    history["val_dice"].append(avg_val_dice)
    history["epoch"].append(epoch + 1)
    
    # Print progress.
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print(f'Training loss: {avg_train_loss:.4f}, training Dice: {avg_train_dice:.4f}')
    if val_ds is not None:
        print(f'Validation loss: {avg_val_loss:.4f}, validation Dice: {avg_val_dice:.4f}')
    
    # Save visualizations every five epochs.
    if (epoch + 1) % 5 == 0 or epoch == 0:
        # Training set visualizations.
        visualize_predictions(
            model, multiclass_head, train_ds, processor, device,
            os.path.join(output_dir, "train_images"),
            f"epoch_{epoch+1}_train",
            num_samples=4
        )
        
        # Validation set visualizations.
        if val_ds is not None:
            visualize_predictions(
                model, multiclass_head, val_ds, processor, device,
                os.path.join(output_dir, "val_images"),
                f"epoch_{epoch+1}_val",
                num_samples=4
            )
    
    # Save a checkpoint.
    if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'multiclass_head_state_dict': multiclass_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'history': history,
        }, checkpoint_path)
    
    # Early stopping.
    if val_ds is not None and avg_val_loss > 0:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model.
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'multiclass_head_state_dict': multiclass_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history,
            }
            torch.save(best_model_state, os.path.join(output_dir, "checkpoints", "best_model.pth"))
            print(f"✓ Saved best model (validation loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs")
                if best_model_state:
                    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_model_state['epoch']})")
                break
    
    # Plot training curves.
    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

# Save final model parameters.
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(os.path.join(output_dir, f"lora_rank{rank}.safetensors"))
torch.save(multiclass_head.state_dict(), os.path.join(output_dir, "multiclass_head.pth"))

# Save training history.
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

# Save the training summary.
summary = {
    "total_epochs": len(history["epoch"]),
    "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
    "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
    "best_val_loss": best_val_loss if (val_ds is not None and best_val_loss < float('inf')) else None,
    "best_epoch": best_model_state['epoch'] if best_model_state else len(history["epoch"]),
    "config": config_file
}

with open(os.path.join(output_dir, "summary.json"), "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nTraining complete. Results saved to: {output_dir}")
