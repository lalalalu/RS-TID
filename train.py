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
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
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

# 设置PyTorch内存分配策略，避免内存碎片（使用新的环境变量名）
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# 固定随机种子，尽量保证训练过程可复现
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

# ==================== 配置参数 ====================
# 这些参数需要根据实际情况修改
CONFIG_FILE = "./config.yaml"  # 配置文件路径
OUTPUT_DIR = "./training_results"  # 训练结果保存目录
SAVE_CHECKPOINT_EVERY = 5  # 每N个epoch保存一次checkpoint
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值（验证损失不下降的epoch数）
RESUME_CHECKPOINT = None  # 恢复训练的checkpoint路径，None表示从头开始训练
# RESUME_CHECKPOINT = "./training_results/train_20260117_164947/checkpoints/emergency_checkpoint.pth"  # 如果需要恢复训练，取消注释并设置路径

# 损失函数配置（用于控制轮廓精度）
USE_CE_LOSS = True   # 必须使用CE Loss来提供像素级分类梯度，否则模型无法学习
DICE_WEIGHT = 1.0    # Dice Loss权重（主要损失，关注轮廓精度）
CE_WEIGHT = 1.0      # CE Loss权重（重要：提高权重以帮助模型学习像素级分类）
# ==================================================

# 加载配置文件
with open(CONFIG_FILE, "r", encoding='utf-8') as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# 设置随机种子，优先读取配置，否则使用默认固定值
random_seed = int(config_file.get("TRAIN", {}).get("SEED", DEFAULT_SEED))
set_seed(random_seed)
print(f"固定随机种子: {random_seed}")

# 创建输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(OUTPUT_DIR, f"train_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val_images"), exist_ok=True)

# 保存配置
with open(os.path.join(output_dir, "config.yaml"), "w", encoding='utf-8') as f:
    yaml.dump(config_file, f, allow_unicode=True)

# 数据集路径
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
val_dataset_path = config_file["DATASET"].get("VAL_PATH", None)
if val_dataset_path is None:
    # 尝试自动推断验证集路径
    if "train" in train_dataset_path:
        val_dataset_path = train_dataset_path.replace("train", "val")
    else:
        val_dataset_path = None

# 加载SAM模型
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
# 创建SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam

# 添加多类别分类头
multiclass_head = MulticlassHead(num_classes=4, in_channels=1)
# 指定使用 GPU 0
multiclass_head = multiclass_head.to("cuda:1" if torch.cuda.is_available() else "cpu")

# 处理数据集
processor = Samprocessor(model)
try:
    train_ds = DatasetSegmentation(config_file, processor, mode="train")
    if len(train_ds) == 0:
        raise ValueError("训练集为空！请检查数据集路径和文件格式。")
except Exception as e:
    print(f"错误: 无法加载训练集")
    print(f"  训练集路径: {train_dataset_path}")
    print(f"  错误信息: {str(e)}")
    raise

# 检查验证集路径是否存在
val_ds = None
if val_dataset_path and os.path.exists(val_dataset_path):
    try:
        val_ds = DatasetSegmentation(config_file, processor, mode="val")
        if len(val_ds) == 0:
            print(f"警告: 验证集为空，将只使用训练集")
            val_ds = None
    except Exception as e:
        print(f"警告: 无法加载验证集 {val_dataset_path}，将只使用训练集")
        print(f"  错误信息: {str(e)}")
        val_ds = None

# 创建数据加载器
# 确保batch_size是整数类型
batch_size = config_file["TRAIN"]["BATCH_SIZE"]
if isinstance(batch_size, str):
    batch_size = int(batch_size)
else:
    batch_size = int(batch_size)

# 梯度累积步数（用于减少显存占用）
gradient_accumulation_steps = config_file["TRAIN"].get("GRADIENT_ACCUMULATION_STEPS", 1)
if isinstance(gradient_accumulation_steps, str):
    gradient_accumulation_steps = int(gradient_accumulation_steps)
else:
    gradient_accumulation_steps = int(gradient_accumulation_steps)
print(f"梯度累积步数: {gradient_accumulation_steps} (有效batch_size = {batch_size * gradient_accumulation_steps})")

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

# 初始化优化器和损失函数
# 同时优化SAM的LoRA参数和多类别分类头
# 确保学习率和权重衰减是数值类型
learning_rate = config_file["TRAIN"].get("LEARNING_RATE", 1e-4)
if isinstance(learning_rate, str):
    # 处理科学计数法字符串，如 "1e-4"
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

# 计算类别权重（解决类别不平衡问题）
def compute_class_weights(dataset, num_classes=4):
    """
    计算每个类别的权重，用于处理类别不平衡问题
    
    Arguments:
        dataset: 训练数据集
        num_classes: 类别数量
    
    Return:
        class_weights: 每个类别的权重列表
    """
    print("正在计算类别权重...")
    class_counts = {i: 0 for i in range(num_classes)}
    total_pixels = 0
    
    # 采样计算（如果数据集太大，只计算前1000个样本）
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="计算类别分布"):
        try:
            sample = dataset[idx]
            mask = sample["ground_truth_mask"].numpy()
            for class_id in range(num_classes):
                count = np.sum(mask == class_id)
                class_counts[class_id] += count
                total_pixels += count
        except Exception as e:
            print(f"警告: 处理样本 {idx} 时出错: {e}")
            continue
    
    # 计算权重：使用逆频率加权，但更温和以保持轮廓精度
    # weight = total_pixels / (num_classes * class_count)
    class_weights = []
    print("\n类别分布统计:")
    for class_id in range(num_classes):
        count = class_counts[class_id]
        if count > 0:
            # 使用逆频率加权，但避免权重过大
            weight = total_pixels / (num_classes * count)
            # 限制权重范围在[0.5, 5.0]之间，更温和的权重以保持轮廓精度
            # 之前是[0.1, 10.0]，现在降低上限以更关注轮廓而非过度惩罚
            weight = max(0.5, min(5.0, weight))
        else:
            weight = 1.0
        class_weights.append(weight)
        class_name = utils.TAMPER_NAMES.get(class_id, f"类别{class_id}")
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {class_name} (类别{class_id}): {count} 像素 ({percentage:.2f}%), 权重={weight:.4f}")
    
    return class_weights

# 确保epoch数是整数类型
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
if isinstance(num_epochs, str):
    num_epochs = int(num_epochs)
else:
    num_epochs = int(num_epochs)
# 指定使用 GPU 0
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 计算类别权重（需要在device定义之后）
class_weights = compute_class_weights(train_ds, num_classes=4)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\n类别权重张量: {weights_tensor}")

# 多类别分割损失：主要使用DiceLoss以关注轮廓精度，辅助使用轻量级CE Loss
# 降低CE Loss的权重，让模型更关注形状和轮廓匹配
ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
# DiceLoss: to_onehot_y=True 表示自动将类别索引转换为one-hot，softmax=True 表示对预测应用softmax
# include_background=True 表示包含背景类别
# 原因：虽然背景占90%+，但CE Loss已经通过类别权重处理了不平衡，Dice Loss包含背景可以：
# 1. 提供更稳定的梯度信号
# 2. 帮助模型学习区分背景和篡改区域
# 3. 与CE Loss配合，CE关注分类，Dice关注轮廓
dice_loss = monai.losses.DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
# 损失函数权重：根据配置设置
dice_weight = DICE_WEIGHT
ce_weight = CE_WEIGHT if USE_CE_LOSS else 0.0
print(f"\n损失函数配置:")
if USE_CE_LOSS:
    print(f"  Dice Loss权重: {dice_weight} (主要损失，关注轮廓精度)")
    print(f"  CE Loss权重: {ce_weight} (辅助损失，帮助像素级分类)")
    print(f"  总损失 = {dice_weight} * Dice + {ce_weight} * CE")
else:
    print(f"  仅使用 Dice Loss (权重: {dice_weight})，更关注轮廓精度（类似原始代码）")
    print(f"  总损失 = {dice_weight} * Dice")

# 初始化混合精度训练的scaler（用于内存优化）
use_amp = True  # 启用混合精度训练以节省内存（约节省50%）
scaler = GradScaler() if use_amp else None
if use_amp:
    print("✓ 启用混合精度训练（AMP）以节省GPU内存")

# 设置模型到训练模式和设备
model.train()
model.to(device)
multiclass_head.train()

# ==================== 辅助函数 ====================
def visualize_predictions(model, multiclass_head, dataset, processor, device, save_dir, prefix, num_samples=4):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    multiclass_head.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        image_path = dataset.img_files[idx]
        mask_path = dataset.mask_files[idx]
        
        # 加载原始图像和真实掩码
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask_array = np.array(mask)
        gt_class_mask = utils.rgb_mask_to_class_mask(mask_array)
        
        # 预测
        with torch.no_grad():
            batch = [sample]
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            class_logits = multiclass_head(stk_out)
            pred_class_mask = torch.argmax(class_logits, dim=1)[0].cpu().numpy()
            
            # 调整到原始图像大小
            original_size = sample["original_size"]
            pred_class_mask = F.interpolate(
                torch.from_numpy(pred_class_mask).unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode='nearest'
            )[0, 0].numpy().astype(np.int64)
        
        # 可视化
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
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history["epoch"], history["train_loss"], label="训练损失", marker='o')
    if any(v > 0 for v in history["val_loss"]):
        axes[0].plot(history["epoch"], history["val_loss"], label="验证损失", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("损失")
    axes[0].set_title("训练和验证损失")
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice分数曲线
    axes[1].plot(history["epoch"], history["train_dice"], label="训练Dice", marker='o')
    if any(v > 0 for v in history["val_dice"]):
        axes[1].plot(history["epoch"], history["val_dice"], label="验证Dice", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice分数")
    axes[1].set_title("训练和验证Dice分数")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==================== 训练历史记录 ====================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_dice": [],
    "val_dice": [],
    "epoch": []
}

# 早停机制
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# 恢复训练（如果指定了checkpoint）
start_epoch = 0
if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
    print(f"\n正在从 checkpoint 恢复训练: {RESUME_CHECKPOINT}")
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
    print(f"✓ 已恢复训练状态: 从 epoch {start_epoch} 开始")
    print(f"  训练历史: {len(history['epoch'])} 个epoch记录")
    if checkpoint.get('is_emergency', False):
        print(f"  注意: 这是紧急checkpoint，建议检查训练状态")

# 紧急保存函数（用于 OOM 等情况）
def save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir):
    """紧急保存 checkpoint，防止丢失训练进度"""
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
        print(f"✓ 紧急 checkpoint 已保存: {emergency_path}")
        return True
    except Exception as e:
        print(f"✗ 紧急保存失败: {e}")
        return False

print(f"开始训练，共 {num_epochs} 个epoch")
if start_epoch > 0:
    print(f"从 epoch {start_epoch} 恢复训练")
print(f"训练集大小: {len(train_ds)}")
if val_ds is not None:
    print(f"验证集大小: {len(val_ds)}")
print(f"设备: {device}")
print(f"结果保存目录: {output_dir}")

for epoch in range(start_epoch, num_epochs):
    # ========== 训练阶段 ==========
    model.train()
    multiclass_head.train()
    epoch_train_losses = []
    epoch_train_dice_scores = []

    # 在epoch开始时清零梯度
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
        try:
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 使用混合精度训练进行前向传播
            with autocast(enabled=use_amp):
                outputs = model(batched_input=batch, multimask_output=False)
                
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                # 确保张量在正确的设备上
                stk_out = stk_out.to(device)
                stk_gt = stk_gt.to(device)
                # stk_out: [B, 1, H, W]
                # stk_gt: [B, H, W] (类别掩码，值为0-3)
                
                # 删除不需要的outputs以释放内存
                del outputs
                torch.cuda.empty_cache()
                
                # 通过多类别分类头
                class_logits = multiclass_head(stk_out)  # [B, 4, H, W]
                
                # 调整ground truth到低分辨率
                B, C, H, W = class_logits.shape
                stk_gt_resized = F.interpolate(
                    stk_gt.unsqueeze(1).float(), 
                    size=(H, W), 
                    mode='nearest'
                ).squeeze(1).long()
                
                # 计算损失
                # Dice损失：主要损失，更关注轮廓和形状匹配
                dice = dice_loss(class_logits, stk_gt_resized.unsqueeze(1))  # [B, 1, H, W]
                
                # 组合损失
                if USE_CE_LOSS:
                    # CE损失：辅助损失，帮助像素级分类，但权重较低以更关注轮廓
                    ce = ce_loss(class_logits, stk_gt_resized)
                    loss = dice_weight * dice + ce_weight * ce
                else:
                    # 仅使用Dice Loss，更关注轮廓精度（类似原始代码）
                    loss = dice_weight * dice
                    ce = torch.tensor(0.0, device=device)  # 用于记录，不影响训练
                
                # 计算Dice分数（用于监控）- 在autocast内部但使用no_grad
                with torch.no_grad():
                    pred_classes = torch.argmax(class_logits, dim=1)
                    dice_score = utils.compute_multiclass_dice(pred_classes.cpu(), stk_gt_resized.cpu(), num_classes=4)
                    epoch_train_dice_scores.append(float(dice_score))
                    del pred_classes
                    torch.cuda.empty_cache()  # 立即释放计算Dice分数时使用的临时内存
            
            # 在反向传播前，更积极地清理不需要的中间变量
            del stk_out, class_logits  # 提前删除，释放显存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 确保清理完成
            
            # 使用混合精度训练进行反向传播（支持梯度累积）
            # 将损失除以累积步数，以便在累积多个批次后平均梯度
            loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 只在累积步数达到时才更新参数
            if (i + 1) % gradient_accumulation_steps == 0:
                # 在反向传播前清理缓存，释放更多内存
                torch.cuda.empty_cache()
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # 反向传播后立即清理
                torch.cuda.empty_cache()
            # 如果还没达到累积步数，不更新参数，继续累积梯度

            # 保存损失值（与 train_20260408_132355 时期一致：为梯度累积缩放后的标量）
            loss_value = loss.item()
            epoch_train_losses.append(loss_value)
            
            # 清理中间变量（Dice分数已在autocast内部计算并保存）
            # 注意：stk_out 和 class_logits 已在反向传播前删除
            del stk_gt, stk_gt_resized, ce, dice, loss
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ 错误: GPU内存不足 (batch {i+1}, epoch {epoch+1})")
            print(f"建议: 1) 等待其他进程释放GPU内存 2) 增加GRADIENT_ACCUMULATION_STEPS 3) 减小batch_size")
            print(f"当前配置: batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
            torch.cuda.empty_cache()
            
            # 尝试保存紧急 checkpoint
            print("\n正在尝试保存紧急 checkpoint...")
            save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir)
            
            print("\n程序将退出。请解决显存问题后重新运行训练。")
            print("如果有紧急 checkpoint，可以从 training_results/*/checkpoints/emergency_checkpoint.pth 恢复训练。")
            raise
    
    avg_train_loss = mean(epoch_train_losses)
    # 使用numpy.mean或确保所有值都是同一类型
    if epoch_train_dice_scores:
        avg_train_dice = float(np.mean(epoch_train_dice_scores))
    else:
        avg_train_dice = 0.0
    
    # ========== 验证阶段 ==========
    if val_ds is not None:
        model.eval()
        multiclass_head.eval()
        epoch_val_losses = []
        epoch_val_dice_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                try:
                    # 清理缓存
                    torch.cuda.empty_cache()
                    
                    # 使用混合精度训练进行前向传播（验证阶段也使用以节省内存）
                    with autocast(enabled=use_amp):
                        outputs = model(batched_input=batch, multimask_output=False)
                        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                        # 确保张量在正确的设备上
                        stk_out = stk_out.to(device)
                        stk_gt = stk_gt.to(device)
                        
                        # 删除不需要的outputs
                        del outputs
                        torch.cuda.empty_cache()
                        
                        class_logits = multiclass_head(stk_out)
                        
                        B, C, H, W = class_logits.shape
                        stk_gt_resized = F.interpolate(
                            stk_gt.unsqueeze(1).float(), 
                            size=(H, W), 
                            mode='nearest'
                        ).squeeze(1).long()
                        
                        # Dice损失：主要损失，更关注轮廓和形状匹配
                        dice = dice_loss(class_logits, stk_gt_resized.unsqueeze(1))  # [B, 1, H, W]
                        
                        # 组合损失
                        if USE_CE_LOSS:
                            # CE损失：辅助损失
                            ce = ce_loss(class_logits, stk_gt_resized)
                            loss = dice_weight * dice + ce_weight * ce
                        else:
                            # 仅使用Dice Loss
                            loss = dice_weight * dice
                            ce = torch.tensor(0.0, device=device)  # 用于记录
                        
                        # 计算Dice分数
                        pred_classes = torch.argmax(class_logits, dim=1)
                        dice_score = utils.compute_multiclass_dice(pred_classes.cpu(), stk_gt_resized.cpu(), num_classes=4)
                        epoch_val_dice_scores.append(float(dice_score))
                    
                    epoch_val_losses.append(loss.item())
                    
                    # 清理中间变量
                    del stk_out, stk_gt, class_logits, stk_gt_resized, pred_classes
                    del ce, dice, loss
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n❌ 错误: GPU内存不足 (验证 batch {i+1}, epoch {epoch+1})")
                    torch.cuda.empty_cache()
                    
                    # 尝试保存紧急 checkpoint
                    print("\n正在尝试保存紧急 checkpoint...")
                    save_emergency_checkpoint(model, multiclass_head, optimizer, epoch, history, output_dir)
                    
                    print("\n程序将退出。请解决显存问题后重新运行训练。")
                    raise
        
        avg_val_loss = mean(epoch_val_losses)
        # 使用numpy.mean或确保所有值都是同一类型
        if epoch_val_dice_scores:
            avg_val_dice = float(np.mean(epoch_val_dice_scores))
        else:
            avg_val_dice = 0.0
    else:
        avg_val_loss = 0.0
        avg_val_dice = 0.0
    
    # 记录历史
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_dice"].append(avg_train_dice)
    history["val_dice"].append(avg_val_dice)
    history["epoch"].append(epoch + 1)
    
    # 打印信息
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print(f'训练损失: {avg_train_loss:.4f}, 训练Dice: {avg_train_dice:.4f}')
    if val_ds is not None:
        print(f'验证损失: {avg_val_loss:.4f}, 验证Dice: {avg_val_dice:.4f}')
    
    # 保存可视化结果（每5个epoch）
    if (epoch + 1) % 5 == 0 or epoch == 0:
        # 训练集可视化
        visualize_predictions(
            model, multiclass_head, train_ds, processor, device,
            os.path.join(output_dir, "train_images"),
            f"epoch_{epoch+1}_train",
            num_samples=4
        )
        
        # 验证集可视化
        if val_ds is not None:
            visualize_predictions(
                model, multiclass_head, val_ds, processor, device,
                os.path.join(output_dir, "val_images"),
                f"epoch_{epoch+1}_val",
                num_samples=4
            )
    
    # 保存checkpoint
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
    
    # 早停机制
    if val_ds is not None and avg_val_loss > 0:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
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
            print(f"✓ 保存最佳模型 (验证损失: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n早停触发！验证损失在 {EARLY_STOPPING_PATIENCE} 个epoch内未改善")
                if best_model_state:
                    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_model_state['epoch']})")
                break
    
    # 绘制训练曲线
    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

# 保存最终模型
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(os.path.join(output_dir, f"lora_rank{rank}.safetensors"))
torch.save(multiclass_head.state_dict(), os.path.join(output_dir, "multiclass_head.pth"))

# 保存训练历史
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

# 保存训练总结
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

print(f"\n训练完成！结果保存在: {output_dir}")
