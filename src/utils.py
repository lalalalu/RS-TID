import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad

# Colors for tampering classes.
TAMPER_COLORS = {
    0: [0, 0, 0],      # Background - black.
    1: [0, 255, 0],    # copy_move - green.
    2: [255, 0, 0],    # splicing - red.
    3: [0, 0, 255],    # remove - blue.
}

TAMPER_NAMES = {
    0: "Background",
    1: "copy_move",
    2: "splicing",
    3: "remove"
}


def show_mask(mask: np.array, ax, random_color=False):
    """
    Plot the mask

    Arguments:
        mask: Array of the binary mask (or float)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """
    Plot the image and the mask superposed

    Arguments:
        image: PIL original image
        mask: PIL original binary mask
    """
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()
    

def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """
    Take an image from the dataset and plot it

    Arguments:
        dataset: Dataset class loaded with our images
        idx: Index of the data we want
    """
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def rgb_mask_to_class_mask(rgb_mask: np.array) -> np.array:
    """
    Convert an RGB mask to a class-index mask.
    
    Arguments:
        rgb_mask: RGB mask array of shape [H, W, 3].
        
    Return:
        class_mask: Class-index array of shape [H, W], with values from 0 to 3.
    """
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    
    # Match colors with a small tolerance.
    tolerance = 10
    
    # Map white background (255,255,255) to class 0, just like black background (0,0,0).
    white_bg = (rgb_mask[:, :, 0] > (255 - tolerance)) & \
               (rgb_mask[:, :, 1] > (255 - tolerance)) & \
               (rgb_mask[:, :, 2] > (255 - tolerance))
    class_mask[white_bg] = 0
    
    # Match tampering classes (1, 2, 3) before black background (0) to avoid overwriting labels.
    # Priority: green (copy_move), red (splicing), blue (remove), then black (background).
    for class_id in [1, 2, 3, 0]:  # Match tampering classes before background.
        color = TAMPER_COLORS[class_id]
        if class_id == 0:  # Black (background).
            # Black pixels have all channels close to zero.
            # Match only pixels still labeled 0, excluding those already assigned a tampering class.
            mask = (rgb_mask[:, :, 0] < tolerance) & \
                   (rgb_mask[:, :, 1] < tolerance) & \
                   (rgb_mask[:, :, 2] < tolerance) & \
                   (class_mask == 0)  # Match only pixels still labeled as background.
        else:
            # Match the corresponding RGB values for other colors.
            # Nonzero channels must be close to the target; zero channels must be close to zero.
            # This prevents black pixels from matching a tampering class.
            if class_id == 1:  # Green [0, 255, 0].
                mask = (rgb_mask[:, :, 0] < tolerance) & \
                       (rgb_mask[:, :, 1] > (255 - tolerance)) & \
                       (rgb_mask[:, :, 2] < tolerance) & \
                       (class_mask == 0)
            elif class_id == 2:  # Red [255, 0, 0].
                mask = (rgb_mask[:, :, 0] > (255 - tolerance)) & \
                       (rgb_mask[:, :, 1] < tolerance) & \
                       (rgb_mask[:, :, 2] < tolerance) & \
                       (class_mask == 0)
            elif class_id == 3:  # Blue [0, 0, 255].
                mask = (rgb_mask[:, :, 0] < tolerance) & \
                       (rgb_mask[:, :, 1] < tolerance) & \
                       (rgb_mask[:, :, 2] > (255 - tolerance)) & \
                       (class_mask == 0)
        class_mask[mask] = class_id
    
    return class_mask


def class_mask_to_rgb_mask(class_mask: np.array) -> np.array:
    """
    Convert a class-index mask to an RGB mask.
    
    Arguments:
        class_mask: Class-index array of shape [H, W], with values from 0 to 3.
        
    Return:
        rgb_mask: RGB mask array of shape [H, W, 3].
    """
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in TAMPER_COLORS.items():
        mask = class_mask == class_id
        rgb_mask[mask] = color
    
    return rgb_mask


def get_bounding_box(ground_truth_map: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # get bounding box from mask
  idx = np.where(ground_truth_map > 0)
  H, W = ground_truth_map.shape
  
  # If there are no nonzero pixels, return the full-image bounding box.
  if len(idx[0]) == 0:
    return [0, 0, W, H]
  
  x_indices = idx[1]
  y_indices = idx[0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


def stacking_batch(batch, outputs):
    """
    Given the batch and outputs of SAM, stacks the tensors to compute the loss. We stack by adding another dimension.

    Arguments:
        batch(list(dict)): List of dict with the keys given in the dataset file
        outputs: list(dict): List of dict that are the outputs of SAM
    
    Return: 
        stk_gt: Stacked tensor of the ground truth masks in the batch. Shape: [batch_size, H, W]
        stk_out: Stacked tensor of logits mask outputed by SAM. Shape: [batch_size, 1, H, W] 
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
    # Remove extra dimensions: [B, 1, 1, H, W] -> [B, 1, H, W].
    stk_out = stk_out.squeeze(1)
        
    return stk_gt, stk_out


def show_multiclass_mask(class_mask: np.array, ax, alpha=0.6):
    """
    Display a multiclass mask.
    
    Arguments:
        class_mask: Class-index array of shape [H, W], with values from 0 to 3.
        ax: matplotlib axes
        alpha: Overlay opacity.
    """
    rgb_mask = class_mask_to_rgb_mask(class_mask)
    ax.imshow(rgb_mask, alpha=alpha)


def plot_multiclass_prediction(image: np.array, pred_mask: np.array, gt_mask: np.array, 
                               filename: str, save_dir: str = "./plots"):
    """
    Plot multiclass predictions.
    
    Arguments:
        image: Original image array.
        pred_mask: Predicted class-index mask of shape [H, W].
        gt_mask: Ground-truth class-index mask of shape [H, W].
        filename: Output file name.
        save_dir: Output directory.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image.
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    
    # Ground-truth mask.
    axes[1].imshow(image)
    show_multiclass_mask(gt_mask, axes[1])
    axes[1].set_title("Ground-truth mask")
    axes[1].axis("off")
    
    # Predicted mask.
    axes[2].imshow(image)
    show_multiclass_mask(pred_mask, axes[2])
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def compute_multiclass_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4) -> float:
    """
    Compute the multiclass Dice score.
    
    Arguments:
        pred_mask: Predicted class-index mask of shape [B, H, W] or [H, W].
        gt_mask: Ground-truth class-index mask of shape [B, H, W] or [H, W].
        num_classes: Number of classes.
        
    Return:
        dice_score: Mean Dice score.
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)
    
    B = pred_mask.shape[0]
    dice_scores = []
    
    for b in range(B):
        pred = pred_mask[b]
        gt = gt_mask[b]
        
        class_dice = []
        for c in range(num_classes):
            pred_c = (pred == c).float()
            gt_c = (gt == c).float()
            
            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            
            # Convert to a Python scalar.
            intersection_val = float(intersection.item() if isinstance(intersection, torch.Tensor) else intersection)
            union_val = float(union.item() if isinstance(union, torch.Tensor) else union)
            
            if union_val > 0:
                dice = 2.0 * intersection_val / union_val
            else:
                dice = 1.0 if intersection_val == 0 else 0.0
            
            class_dice.append(dice)
        
        dice_scores.append(float(np.mean(class_dice)))
    
    return float(np.mean(dice_scores))


def compute_confusion_matrix(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4) -> np.ndarray:
    """
    Compute a confusion matrix.
    
    Arguments:
        pred_mask: Predicted class-index mask of shape [B, H, W] or [H, W].
        gt_mask: Ground-truth class-index mask of shape [B, H, W] or [H, W].
        num_classes: Number of classes.
        
    Return:
        confusion_matrix: Array of shape [num_classes, num_classes].
                          confusion_matrix[i, j] counts pixels with true class i and predicted class j.
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)
    
    # Flatten to 1D.
    pred_flat = pred_mask.flatten().cpu().numpy().astype(np.int64)
    gt_flat = gt_mask.flatten().cpu().numpy().astype(np.int64)
    
    # Map labels outside [0, num_classes-1] to background (0) to avoid invalid indices.
    invalid_gt = (gt_flat < 0) | (gt_flat >= num_classes)
    invalid_pred = (pred_flat < 0) | (pred_flat >= num_classes)
    gt_flat[invalid_gt] = 0
    pred_flat[invalid_pred] = 0
    gt_flat = np.clip(gt_flat, 0, num_classes - 1)
    pred_flat = np.clip(pred_flat, 0, num_classes - 1)
    
    # Compute the confusion matrix.
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(pred_flat)):
        confusion_matrix[int(gt_flat[i]), int(pred_flat[i])] += 1
    
    return confusion_matrix


def compute_classification_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4):
    """
    Compute classification metrics: accuracy, precision, recall, and F1 score.
    
    Arguments:
        pred_mask: Predicted class-index mask of shape [B, H, W] or [H, W].
        gt_mask: Ground-truth class-index mask of shape [B, H, W] or [H, W].
        num_classes: Number of classes.
        
    Return:
        metrics: Dictionary of per-class precision, recall, F1, and overall accuracy.
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)
    
    # Flatten to 1D.
    pred_flat = pred_mask.flatten().cpu().numpy()
    gt_flat = gt_mask.flatten().cpu().numpy()
    
    # Compute the confusion matrix.
    cm = compute_confusion_matrix(pred_mask, gt_mask, num_classes)
    
    # Compute per-class metrics.
    metrics = {
        'overall_accuracy': float(np.sum(pred_flat == gt_flat) / len(pred_flat)),
        'per_class': {}
    }
    
    for c in range(num_classes):
        # True positive: pixels correctly predicted as this class.
        tp = cm[c, c]
        # False positive: pixels incorrectly predicted as this class.
        fp = np.sum(cm[:, c]) - tp
        # False negative: pixels of this class predicted as another class.
        fn = np.sum(cm[c, :]) - tp
        # True negative: pixels correctly predicted as outside this class.
        tn = np.sum(cm) - tp - fp - fn
        
        # Precision: fraction of predicted positives that are true positives.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: fraction of actual positives predicted correctly.
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score: harmonic mean of precision and recall.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Class accuracy: fraction of pixels in this class predicted correctly.
        class_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics['per_class'][c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(class_acc),
            'support': int(tp + fn)  # Number of ground-truth pixels in this class.
        }
    
    return metrics


# Display names for confusion matrix labels.
CM_CLASS_DISPLAY_NAMES = {
    "Background": "Background",
    "copy_move": "Copy-Move",
    "splicing": "Splicing",
    "remove": "Remove",
}


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list, save_path: str = None):
    """
    Plot a confusion matrix.
    
    Arguments:
        confusion_matrix: Array of shape [num_classes, num_classes].
        class_names: List of class names.
        save_path: Output path; display the figure if None.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Use English class labels in the figure.
    display_names = [CM_CLASS_DISPLAY_NAMES.get(name, name) for name in class_names]
    
    # Compute percentages.
    cm_percent = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    
    # Add raw counts below the percentages.
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            text = ax.texts[i * len(display_names) + j]
            text.set_text(f'{text.get_text()}\n({int(confusion_matrix[i, j])})')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
