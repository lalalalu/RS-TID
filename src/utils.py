import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad

# 篡改类型颜色定义
TAMPER_COLORS = {
    0: [0, 0, 0],      # 未篡改 - 黑色
    1: [0, 255, 0],    # copy_move - 绿色
    2: [255, 0, 0],    # splicing - 红色
    3: [0, 0, 255],    # remove - 蓝色
}

TAMPER_NAMES = {
    0: "未篡改",
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
    将RGB掩码转换为类别掩码
    
    Arguments:
        rgb_mask: RGB掩码数组，形状为[H, W, 3]
        
    Return:
        class_mask: 类别掩码数组，形状为[H, W]，值为0-3
    """
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    
    # 颜色匹配（允许小范围误差）
    tolerance = 10
    
    # 先显式处理白色背景：部分数据集中背景为白色 (255,255,255)，与黑色(0,0,0)均编码为类别 0（未篡改）
    white_bg = (rgb_mask[:, :, 0] > (255 - tolerance)) & \
               (rgb_mask[:, :, 1] > (255 - tolerance)) & \
               (rgb_mask[:, :, 2] > (255 - tolerance))
    class_mask[white_bg] = 0
    
    # 再匹配篡改类别（1, 2, 3），最后匹配黑色未篡改（0），避免黑色被覆盖
    # 按优先级：绿色(copy_move) -> 红色(splicing) -> 蓝色(remove) -> 黑色(未篡改)
    for class_id in [1, 2, 3, 0]:  # 先匹配篡改类别，最后匹配未篡改
        color = TAMPER_COLORS[class_id]
        if class_id == 0:  # 黑色（未篡改）
            # 黑色区域：所有通道都接近0
            # 只匹配还没有被分类的像素（仍然是0的像素，即未匹配到白/绿/红/蓝的）
            mask = (rgb_mask[:, :, 0] < tolerance) & \
                   (rgb_mask[:, :, 1] < tolerance) & \
                   (rgb_mask[:, :, 2] < tolerance) & \
                   (class_mask == 0)  # 只匹配还未分类的像素
        else:
            # 其他颜色：严格匹配对应的RGB值
            # 对于非零通道，必须接近目标值；对于零通道，必须接近0
            # 这样可以避免黑色像素被误匹配
            if class_id == 1:  # 绿色 [0, 255, 0]
                mask = (rgb_mask[:, :, 0] < tolerance) & \
                       (rgb_mask[:, :, 1] > (255 - tolerance)) & \
                       (rgb_mask[:, :, 2] < tolerance) & \
                       (class_mask == 0)
            elif class_id == 2:  # 红色 [255, 0, 0]
                mask = (rgb_mask[:, :, 0] > (255 - tolerance)) & \
                       (rgb_mask[:, :, 1] < tolerance) & \
                       (rgb_mask[:, :, 2] < tolerance) & \
                       (class_mask == 0)
            elif class_id == 3:  # 蓝色 [0, 0, 255]
                mask = (rgb_mask[:, :, 0] < tolerance) & \
                       (rgb_mask[:, :, 1] < tolerance) & \
                       (rgb_mask[:, :, 2] > (255 - tolerance)) & \
                       (class_mask == 0)
        class_mask[mask] = class_id
    
    return class_mask


def class_mask_to_rgb_mask(class_mask: np.array) -> np.array:
    """
    将类别掩码转换为RGB掩码
    
    Arguments:
        class_mask: 类别掩码数组，形状为[H, W]，值为0-3
        
    Return:
        rgb_mask: RGB掩码数组，形状为[H, W, 3]
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
  
  # 如果没有非零像素，返回整个图像的边界框
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
    # 移除多余的维度: [B, 1, 1, H, W] -> [B, 1, H, W]
    stk_out = stk_out.squeeze(1)
        
    return stk_gt, stk_out


def show_multiclass_mask(class_mask: np.array, ax, alpha=0.6):
    """
    显示多类别掩码
    
    Arguments:
        class_mask: 类别掩码数组，形状为[H, W]，值为0-3
        ax: matplotlib axes
        alpha: 透明度
    """
    rgb_mask = class_mask_to_rgb_mask(class_mask)
    ax.imshow(rgb_mask, alpha=alpha)


def plot_multiclass_prediction(image: np.array, pred_mask: np.array, gt_mask: np.array, 
                               filename: str, save_dir: str = "./plots"):
    """
    绘制多类别预测结果
    
    Arguments:
        image: 原始图像数组
        pred_mask: 预测的类别掩码，形状为[H, W]
        gt_mask: 真实类别掩码，形状为[H, W]
        filename: 保存的文件名
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("原始图像")
    axes[0].axis("off")
    
    # 真实掩码
    axes[1].imshow(image)
    show_multiclass_mask(gt_mask, axes[1])
    axes[1].set_title("真实掩码")
    axes[1].axis("off")
    
    # 预测掩码
    axes[2].imshow(image)
    show_multiclass_mask(pred_mask, axes[2])
    axes[2].set_title("预测掩码")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def compute_multiclass_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4) -> float:
    """
    计算多类别Dice分数
    
    Arguments:
        pred_mask: 预测的类别掩码，形状为[B, H, W]或[H, W]
        gt_mask: 真实类别掩码，形状为[B, H, W]或[H, W]
        num_classes: 类别数量
        
    Return:
        dice_score: 平均Dice分数
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
            
            # 确保转换为Python数值类型
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
    计算混淆矩阵
    
    Arguments:
        pred_mask: 预测的类别掩码，形状为[B, H, W]或[H, W]
        gt_mask: 真实类别掩码，形状为[B, H, W]或[H, W]
        num_classes: 类别数量
        
    Return:
        confusion_matrix: 混淆矩阵，形状为[num_classes, num_classes]
                          confusion_matrix[i, j] 表示真实类别为i，预测为j的像素数量
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)
    
    # 展平为1D
    pred_flat = pred_mask.flatten().cpu().numpy().astype(np.int64)
    gt_flat = gt_mask.flatten().cpu().numpy().astype(np.int64)
    
    # 将超出 [0, num_classes-1] 的标签统一视为背景(0)，避免索引越界与编码不一致
    invalid_gt = (gt_flat < 0) | (gt_flat >= num_classes)
    invalid_pred = (pred_flat < 0) | (pred_flat >= num_classes)
    gt_flat[invalid_gt] = 0
    pred_flat[invalid_pred] = 0
    gt_flat = np.clip(gt_flat, 0, num_classes - 1)
    pred_flat = np.clip(pred_flat, 0, num_classes - 1)
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(pred_flat)):
        confusion_matrix[int(gt_flat[i]), int(pred_flat[i])] += 1
    
    return confusion_matrix


def compute_classification_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int = 4):
    """
    计算分类准确性指标：准确率、精确率、召回率、F1分数
    
    Arguments:
        pred_mask: 预测的类别掩码，形状为[B, H, W]或[H, W]
        gt_mask: 真实类别掩码，形状为[B, H, W]或[H, W]
        num_classes: 类别数量
        
    Return:
        metrics: 字典，包含每个类别的精确率、召回率、F1分数，以及总体准确率
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)
    
    # 展平为1D
    pred_flat = pred_mask.flatten().cpu().numpy()
    gt_flat = gt_mask.flatten().cpu().numpy()
    
    # 计算混淆矩阵
    cm = compute_confusion_matrix(pred_mask, gt_mask, num_classes)
    
    # 计算每个类别的指标
    metrics = {
        'overall_accuracy': float(np.sum(pred_flat == gt_flat) / len(pred_flat)),
        'per_class': {}
    }
    
    for c in range(num_classes):
        # True Positive: 正确预测为该类别的像素数
        tp = cm[c, c]
        # False Positive: 错误预测为该类别的像素数
        fp = np.sum(cm[:, c]) - tp
        # False Negative: 真实为该类别但预测错误的像素数
        fn = np.sum(cm[c, :]) - tp
        # True Negative: 正确预测为非该类别的像素数
        tn = np.sum(cm) - tp - fp - fn
        
        # 精确率 (Precision): 预测为正例中真正为正例的比例
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 召回率 (Recall): 真正例中被正确预测的比例
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1分数: 精确率和召回率的调和平均
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 类别准确率 (Class Accuracy): 该类别的正确预测比例
        class_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics['per_class'][c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(class_acc),
            'support': int(tp + fn)  # 真实样本数
        }
    
    return metrics


# 混淆矩阵图中类别名对应的英文显示（避免中文乱码）
CM_CLASS_DISPLAY_NAMES = {
    "未篡改": "Background",
    "copy_move": "Copy-Move",
    "splicing": "Splicing",
    "remove": "Remove",
}


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list, save_path: str = None):
    """
    绘制混淆矩阵
    
    Arguments:
        confusion_matrix: 混淆矩阵，形状为[num_classes, num_classes]
        class_names: 类别名称列表
        save_path: 保存路径，如果为None则显示
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 图中使用英文标签，避免无中文字体时出现乱码
    display_names = [CM_CLASS_DISPLAY_NAMES.get(name, name) for name in class_names]
    
    # 计算百分比
    cm_percent = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    
    # 添加原始数值（在百分比下方）
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