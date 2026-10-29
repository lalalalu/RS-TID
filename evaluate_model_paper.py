import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.lora import LoRA_sam
from src.multiclass_head import MulticlassHead
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b

# ==================== 配置参数 ====================
CONFIG_FILE = "./config.yaml"
TRAINING_RESULT_DIR = "./training_results/train_20260411_173112"  # 修改为你的训练结果目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_TEST_SET = True   # True=测试集, False=验证集
BATCH_SIZE = 1        # 当前项目沿用 batch_size=1，避免与 SAM 推理接口冲突
NUM_CLASSES = 4
IGNORE_BACKGROUND_FOR_MAIN_REPORT = True
SAVE_NORMALIZED_CM_JSON = True
# 优先从验证集最优 checkpoint 加载（与 train.py 保存的 best_model.pth 一致）；
# 若不存在则回退到训练结束导出的 lora_rank*.safetensors + multiclass_head.pth
LOAD_FROM_BEST_CHECKPOINT = True
# ==================================================


def load_model(config_file: dict, training_result_dir: str, device: str):
    """加载训练好的模型"""
    print("正在加载模型...")

    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
    model = sam_lora.sam

    best_path = os.path.join(training_result_dir, "checkpoints", "best_model.pth")
    multiclass_head = MulticlassHead(num_classes=NUM_CLASSES, in_channels=1)

    if LOAD_FROM_BEST_CHECKPOINT and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        if "model_state_dict" not in ckpt or "multiclass_head_state_dict" not in ckpt:
            raise KeyError(
                f"{best_path} 缺少 model_state_dict / multiclass_head_state_dict，"
                "请确认该文件由 train.py 保存。"
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        multiclass_head.load_state_dict(ckpt["multiclass_head_state_dict"])
        ep = ckpt.get("epoch", "?")
        vl = ckpt.get("val_loss", None)
        print(f"✓ 已从验证集最优 checkpoint 加载: {best_path}")
        print(f"  (epoch={ep}, val_loss={vl})")
    else:
        if LOAD_FROM_BEST_CHECKPOINT:
            print(f"  未找到 {best_path}，回退到 safetensors + multiclass_head.pth")
        lora_path = os.path.join(training_result_dir, f"lora_rank{sam_lora.rank}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"未找到 LoRA 权重文件: {lora_path}")
        sam_lora.load_lora_parameters(lora_path)
        print(f"✓ 已加载 LoRA 权重: {lora_path}")

        multiclass_head_path = os.path.join(training_result_dir, "multiclass_head.pth")
        if not os.path.exists(multiclass_head_path):
            raise FileNotFoundError(f"未找到多类别分类头权重文件: {multiclass_head_path}")
        multiclass_head.load_state_dict(torch.load(multiclass_head_path, map_location=device))
        print(f"✓ 已加载多类别分类头权重: {multiclass_head_path}")

    model.to(device)
    model.eval()
    multiclass_head.to(device)
    multiclass_head.eval()

    return model, multiclass_head


def get_dataset(config_file: dict, processor: Samprocessor):
    """获取评估数据集"""
    if USE_TEST_SET:
        dataset = DatasetSegmentation(config_file, processor, mode="test")
        dataset_name = "测试集"
        print("使用测试集进行评估（推荐）")
    else:
        dataset = DatasetSegmentation(config_file, processor, mode="val")
        dataset_name = "验证集"
        print("使用验证集进行评估（注意：可能高估性能）")

    print(f"{dataset_name}大小: {len(dataset)}")
    return dataset, dataset_name


def compute_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    从“全数据集累计混淆矩阵”统一计算论文可用指标。

    confusion_matrix[i, j] = 真值为 i，预测为 j 的像素数
    """
    cm = confusion_matrix.astype(np.float64)
    total = cm.sum()
    overall_accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "dice": float(dice),
            "iou": float(iou),
            "specificity": float(specificity),
            "support": int(cm[c, :].sum()),
            "predicted_pixels": int(cm[:, c].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    def macro_average(indices: List[int]) -> Dict[str, float]:
        names = [class_names[i] for i in indices]
        keys = ["precision", "recall", "f1", "dice", "iou", "specificity"]
        return {
            k: float(np.mean([per_class[n][k] for n in names])) if len(names) > 0 else 0.0
            for k in keys
        }

    def weighted_average(indices: List[int]) -> Dict[str, float]:
        names = [class_names[i] for i in indices]
        supports = np.array([per_class[n]["support"] for n in names], dtype=np.float64)
        total_support = supports.sum()
        keys = ["precision", "recall", "f1", "dice", "iou", "specificity"]
        if total_support <= 0:
            return {k: 0.0 for k in keys}
        return {
            k: float(np.sum(np.array([per_class[n][k] for n in names]) * supports) / total_support)
            for k in keys
        }

    all_indices = list(range(len(class_names)))
    tamper_indices = [1, 2, 3]

    metrics = {
        "overall_accuracy": overall_accuracy,
        "macro_avg_all_classes": macro_average(all_indices),
        "weighted_avg_all_classes": weighted_average(all_indices),
        "macro_avg_tamper_only": macro_average(tamper_indices),
        "weighted_avg_tamper_only": weighted_average(tamper_indices),
        "per_class": per_class,
    }
    return metrics


def normalize_confusion_matrix_by_row(confusion_matrix: np.ndarray) -> np.ndarray:
    """行归一化混淆矩阵：每一行除以该真实类别总数。"""
    cm = confusion_matrix.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def evaluate_dataset(model, multiclass_head, dataset, device: str, dataset_name: str):
    """
    评估逻辑：
    1. SAM 输出 low_res_logits
    2. 多类别头输出 class logits
    3. 将 GT 下采样到预测 logits 尺寸（nearest），与训练脚本口径保持一致（GT → pred）
    4. 在“预测分辨率”上累计总混淆矩阵
    5. 最终全部指标从总混淆矩阵计算
    """
    print(f"\n正在评估 {dataset_name}...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    total_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    model.eval()
    multiclass_head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"评估 {dataset_name}"):
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)

            stk_out = stk_out.to(device)
            stk_gt = stk_gt.to(device).long()  # [B, H_gt, W_gt]

            class_logits = multiclass_head(stk_out)  # [B, C, H_pred, W_pred]

            # 与 train.py 对齐：将 GT 下采样到预测尺寸，再计算指标
            H_pred, W_pred = class_logits.shape[-2:]
            stk_gt_resized = F.interpolate(
                stk_gt.unsqueeze(1).float(),
                size=(H_pred, W_pred),
                mode="nearest",
            ).squeeze(1).long()  # [B, H_pred, W_pred]
            pred_classes = torch.argmax(class_logits, dim=1).long()  # [B, H_pred, W_pred]

            cm = utils.compute_confusion_matrix(
                pred_classes.cpu(),
                stk_gt_resized.cpu(),
                num_classes=NUM_CLASSES,
            )
            total_confusion_matrix += cm

    class_names = [utils.TAMPER_NAMES[i] for i in range(NUM_CLASSES)]
    metrics = compute_metrics_from_confusion_matrix(total_confusion_matrix, class_names)
    normalized_cm = normalize_confusion_matrix_by_row(total_confusion_matrix)
    return total_confusion_matrix, normalized_cm, metrics, class_names


def print_metrics(metrics: Dict, class_names: List[str]):
    print("\n" + "=" * 72)
    print("论文报告指标（由全测试集总混淆矩阵统一计算）")
    print("=" * 72)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")

    print("\n[四类宏平均（含背景）]")
    for k, v in metrics["macro_avg_all_classes"].items():
        print(f"  {k:>12}: {v:.4f}")

    print("\n[仅篡改类宏平均（推荐论文主表使用）]")
    for k, v in metrics["macro_avg_tamper_only"].items():
        print(f"  {k:>12}: {v:.4f}")

    print("\n[Per-Class Metrics]")
    for name in class_names:
        m = metrics["per_class"][name]
        print(f"\n{name}:")
        print(f"  Precision   : {m['precision']:.4f}")
        print(f"  Recall      : {m['recall']:.4f}")
        print(f"  F1          : {m['f1']:.4f}")
        print(f"  Dice        : {m['dice']:.4f}")
        print(f"  IoU         : {m['iou']:.4f}")
        print(f"  Specificity : {m['specificity']:.4f}")
        print(f"  Support     : {m['support']}")



def save_results(output_dir: str, confusion_matrix: np.ndarray, normalized_cm: np.ndarray,
                 metrics: Dict, class_names: List[str]):
    os.makedirs(output_dir, exist_ok=True)

    # 混淆矩阵图片
    utils.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # 原始混淆矩阵 JSON
    with open(os.path.join(output_dir, "confusion_matrix_raw.json"), "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "confusion_matrix": confusion_matrix.tolist(),
        }, f, ensure_ascii=False, indent=2)

    # 行归一化混淆矩阵 JSON
    if SAVE_NORMALIZED_CM_JSON:
        with open(os.path.join(output_dir, "confusion_matrix_row_normalized.json"), "w", encoding="utf-8") as f:
            json.dump({
                "class_names": class_names,
                "confusion_matrix_row_normalized": normalized_cm.tolist(),
            }, f, ensure_ascii=False, indent=2)

    # 论文可用指标 JSON
    with open(os.path.join(output_dir, "metrics_paper_ready.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 已保存混淆矩阵图: {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"✓ 已保存原始混淆矩阵: {os.path.join(output_dir, 'confusion_matrix_raw.json')}")
    if SAVE_NORMALIZED_CM_JSON:
        print(f"✓ 已保存行归一化混淆矩阵: {os.path.join(output_dir, 'confusion_matrix_row_normalized.json')}")
    print(f"✓ 已保存论文可用指标: {os.path.join(output_dir, 'metrics_paper_ready.json')}")



def build_paper_summary(metrics: Dict) -> Dict:
    """给正文/表格直接取值用的摘要字段。"""
    summary = {
        "overall_accuracy": metrics["overall_accuracy"],
        "macro_dice_all_classes": metrics["macro_avg_all_classes"]["dice"],
        "macro_iou_all_classes": metrics["macro_avg_all_classes"]["iou"],
        "macro_f1_all_classes": metrics["macro_avg_all_classes"]["f1"],
        "macro_dice_tamper_only": metrics["macro_avg_tamper_only"]["dice"],
        "macro_iou_tamper_only": metrics["macro_avg_tamper_only"]["iou"],
        "macro_f1_tamper_only": metrics["macro_avg_tamper_only"]["f1"],
    }
    return summary



def main():
    print("=" * 72)
    print("论文可用模型评估脚本")
    print("=" * 72)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    model, multiclass_head = load_model(config_file, TRAINING_RESULT_DIR, DEVICE)
    processor = Samprocessor(model)
    dataset, dataset_name = get_dataset(config_file, processor)

    confusion_matrix, normalized_cm, metrics, class_names = evaluate_dataset(
        model=model,
        multiclass_head=multiclass_head,
        dataset=dataset,
        device=DEVICE,
        dataset_name=dataset_name,
    )

    print_metrics(metrics, class_names)

    summary = build_paper_summary(metrics)
    print("\n[建议论文主文中优先报告的摘要指标]")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    output_dir = os.path.join(TRAINING_RESULT_DIR, "evaluation_paper")
    save_results(output_dir, confusion_matrix, normalized_cm, metrics, class_names)

    with open(os.path.join(output_dir, "paper_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存论文摘要指标: {os.path.join(output_dir, 'paper_summary.json')}")


if __name__ == "__main__":
    main()
