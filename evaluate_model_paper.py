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

# ==================== Configuration ====================
CONFIG_FILE = "./config.yaml"
TRAINING_RESULT_DIR = "./training_results/train_20260411_173112"  # Set your training result directory.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_TEST_SET = True   # True: test set; False: validation set.
BATCH_SIZE = 1        # Keep batch_size=1 for compatibility with the SAM inference interface.
NUM_CLASSES = 4
IGNORE_BACKGROUND_FOR_MAIN_REPORT = True
SAVE_NORMALIZED_CM_JSON = True
# Prefer the best validation checkpoint saved by train.py as best_model.pth.
# Otherwise, use the final exported lora_rank*.safetensors and multiclass_head.pth.
LOAD_FROM_BEST_CHECKPOINT = True
# ==================================================


def load_model(config_file: dict, training_result_dir: str, device: str):
    """Load a trained model."""
    print("Loading model...")

    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
    model = sam_lora.sam

    best_path = os.path.join(training_result_dir, "checkpoints", "best_model.pth")
    multiclass_head = MulticlassHead(num_classes=NUM_CLASSES, in_channels=1)

    if LOAD_FROM_BEST_CHECKPOINT and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        if "model_state_dict" not in ckpt or "multiclass_head_state_dict" not in ckpt:
            raise KeyError(
                f"{best_path} is missing model_state_dict / multiclass_head_state_dict; "
                "make sure this file was saved by train.py."
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        multiclass_head.load_state_dict(ckpt["multiclass_head_state_dict"])
        ep = ckpt.get("epoch", "?")
        vl = ckpt.get("val_loss", None)
        print(f"✓ Loaded best validation checkpoint: {best_path}")
        print(f"  (epoch={ep}, val_loss={vl})")
    else:
        if LOAD_FROM_BEST_CHECKPOINT:
            print(f"  {best_path} not found; falling back to safetensors + multiclass_head.pth")
        lora_path = os.path.join(training_result_dir, f"lora_rank{sam_lora.rank}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        sam_lora.load_lora_parameters(lora_path)
        print(f"✓ Loaded LoRA weights: {lora_path}")

        multiclass_head_path = os.path.join(training_result_dir, "multiclass_head.pth")
        if not os.path.exists(multiclass_head_path):
            raise FileNotFoundError(f"Multiclass head weights not found: {multiclass_head_path}")
        multiclass_head.load_state_dict(torch.load(multiclass_head_path, map_location=device))
        print(f"✓ Loaded multiclass head weights: {multiclass_head_path}")

    model.to(device)
    model.eval()
    multiclass_head.to(device)
    multiclass_head.eval()

    return model, multiclass_head


def get_dataset(config_file: dict, processor: Samprocessor):
    """Load the evaluation dataset."""
    if USE_TEST_SET:
        dataset = DatasetSegmentation(config_file, processor, mode="test")
        dataset_name = "Test set"
        print("Evaluating on the test set (recommended)")
    else:
        dataset = DatasetSegmentation(config_file, processor, mode="val")
        dataset_name = "Validation set"
        print("Evaluating on the validation set (performance may be overestimated)")

    print(f"{dataset_name} size: {len(dataset)}")
    return dataset, dataset_name


def compute_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute metrics from the confusion matrix accumulated over the entire dataset.

    confusion_matrix[i, j] counts pixels with ground-truth label i and predicted label j.
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
    """Normalize each row by the total pixel count of its ground-truth class."""
    cm = confusion_matrix.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def evaluate_dataset(model, multiclass_head, dataset, device: str, dataset_name: str):
    """
    Evaluation procedure:
    1. SAM produces low_res_logits.
    2. The multiclass head produces class logits.
    3. Downsample GT to the logit resolution with nearest-neighbor interpolation, as in training.
    4. Accumulate the confusion matrix at the prediction resolution.
    5. Compute all final metrics from the accumulated confusion matrix.
    """
    print(f"\nEvaluating {dataset_name}...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    total_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    model.eval()
    multiclass_head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            outputs = model(batched_input=batch, multimask_output=False)
            stk_gt, stk_out = utils.stacking_batch(batch, outputs)

            stk_out = stk_out.to(device)
            stk_gt = stk_gt.to(device).long()  # [B, H_gt, W_gt]

            class_logits = multiclass_head(stk_out)  # [B, C, H_pred, W_pred]

            # Match train.py: downsample GT to the prediction size before computing metrics.
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
    print("Paper metrics (computed from the confusion matrix accumulated over the entire test set)")
    print("=" * 72)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")

    print("\n[Macro average over all four classes, including background]")
    for k, v in metrics["macro_avg_all_classes"].items():
        print(f"  {k:>12}: {v:.4f}")

    print("\n[Macro average over tampering classes only, recommended for the main results table]")
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

    # Confusion matrix figure.
    utils.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Raw confusion matrix JSON.
    with open(os.path.join(output_dir, "confusion_matrix_raw.json"), "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "confusion_matrix": confusion_matrix.tolist(),
        }, f, ensure_ascii=False, indent=2)

    # Row-normalized confusion matrix JSON.
    if SAVE_NORMALIZED_CM_JSON:
        with open(os.path.join(output_dir, "confusion_matrix_row_normalized.json"), "w", encoding="utf-8") as f:
            json.dump({
                "class_names": class_names,
                "confusion_matrix_row_normalized": normalized_cm.tolist(),
            }, f, ensure_ascii=False, indent=2)

    # Metrics JSON for paper reporting.
    with open(os.path.join(output_dir, "metrics_paper_ready.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved confusion matrix figure: {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"✓ Saved raw confusion matrix: {os.path.join(output_dir, 'confusion_matrix_raw.json')}")
    if SAVE_NORMALIZED_CM_JSON:
        print(f"✓ Saved row-normalized confusion matrix: {os.path.join(output_dir, 'confusion_matrix_row_normalized.json')}")
    print(f"✓ Saved paper metrics: {os.path.join(output_dir, 'metrics_paper_ready.json')}")



def build_paper_summary(metrics: Dict) -> Dict:
    """Build summary fields for the paper text and tables."""
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
    print("Model evaluation for paper reporting")
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
    print("\n[Summary metrics recommended for the paper]")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    output_dir = os.path.join(TRAINING_RESULT_DIR, "evaluation_paper")
    save_results(output_dir, confusion_matrix, normalized_cm, metrics, class_names)

    with open(os.path.join(output_dir, "paper_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved paper summary metrics: {os.path.join(output_dir, 'paper_summary.json')}")


if __name__ == "__main__":
    main()
