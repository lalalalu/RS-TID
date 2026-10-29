"""
Single-image tampering localization (full-image bbox, no shape prompt or mask_inputs, as in training).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

import src.utils as utils
from src.lora import LoRA_sam
from src.multiclass_head import MulticlassHead
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b

# ==================== Configuration ====================
IMAGE_PATH = "./all_dataset/test/images/P1726.png"

CONFIG_FILE = "./config.yaml"
SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
TRAINING_RESULT_DIR = "./training_results/train_20260210_220741"

LORA_WEIGHTS = None
MULTICLASS_HEAD_WEIGHTS = None

OUTPUT_DIR = "./test_results/single_image"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==================================================


def find_latest_training_result(base_dir="./training_results"):
    if not os.path.exists(base_dir):
        return None
    train_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("train_")
    ]
    if not train_dirs:
        return None
    train_dirs.sort(reverse=True)
    return os.path.join(base_dir, train_dirs[0])


def load_model(config_file, sam_checkpoint, training_result_dir, lora_weights, multiclass_head_weights, device):
    print("Loading model...")

    if training_result_dir and os.path.exists(training_result_dir):
        print(f"Loading from training result directory: {training_result_dir}")
        if lora_weights is None:
            rank = config_file["SAM"]["RANK"]
            lora_weights = os.path.join(training_result_dir, f"lora_rank{rank}.safetensors")
            if not os.path.exists(lora_weights):
                lora_files = [
                    f for f in os.listdir(training_result_dir)
                    if f.startswith("lora_rank") and f.endswith(".safetensors")
                ]
                if lora_files:
                    lora_weights = os.path.join(training_result_dir, lora_files[0])
                    print(f"  Found LoRA file: {lora_files[0]}")

        if multiclass_head_weights is None:
            multiclass_head_weights = os.path.join(training_result_dir, "multiclass_head.pth")
            if not os.path.exists(multiclass_head_weights):
                best_model_path = os.path.join(training_result_dir, "checkpoints", "best_model.pth")
                if os.path.exists(best_model_path):
                    print("  Extracting multiclass_head weights from best_model.pth...")
                    checkpoint = torch.load(best_model_path, map_location=device)
                    if "multiclass_head_state_dict" in checkpoint:
                        multiclass_head_weights = os.path.join(
                            training_result_dir, "multiclass_head_from_best.pth"
                        )
                        torch.save(checkpoint["multiclass_head_state_dict"], multiclass_head_weights)
                        print(f"  Saved to: {multiclass_head_weights}")

    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    rank = config_file["SAM"]["RANK"]
    sam_lora = LoRA_sam(sam, rank)

    if lora_weights and os.path.exists(lora_weights):
        sam_lora.load_lora_parameters(lora_weights)
        print(f"✓ Loaded LoRA: {lora_weights}")
    else:
        print(f"Warning: LoRA file not found: {lora_weights}")

    model = sam_lora.sam
    model.eval()
    model.to(device)

    multiclass_head = MulticlassHead(num_classes=4, in_channels=1)
    if multiclass_head_weights and os.path.exists(multiclass_head_weights):
        multiclass_head.load_state_dict(torch.load(multiclass_head_weights, map_location=device))
        print(f"✓ Loaded classification head: {multiclass_head_weights}")
    else:
        raise FileNotFoundError(f"Classification head weights not found: {multiclass_head_weights}")

    multiclass_head.eval()
    multiclass_head.to(device)
    return model, multiclass_head


def predict_image(model, multiclass_head, processor, image_path, device):
    """Use a full-image bbox, matching DatasetSegmentation and training."""
    print(f"Processing: {image_path}")
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image)
    original_size = tuple(image.size)[::-1]
    H, W = original_size
    box = [0, 0, W, H]

    model.eval()
    multiclass_head.eval()
    with torch.no_grad():
        inputs = processor(image, original_size, box)
        batch = [inputs]
        outputs = model(batched_input=batch, multimask_output=False)
        stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0).squeeze(1)
        class_logits = multiclass_head(stk_out)
        pred_class_mask = torch.argmax(class_logits, dim=1)[0].cpu().numpy()

        pred_class_mask_resized = F.interpolate(
            torch.from_numpy(pred_class_mask).unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        )[0, 0].numpy().astype(np.int64)

        class_probs = F.softmax(class_logits, dim=1)[0].cpu().numpy()
        class_probs_resized = F.interpolate(
            torch.from_numpy(class_probs).unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )[0].numpy()

    return pred_class_mask_resized, class_probs_resized, image_array


def save_results(pred_mask, class_probs, image_array, output_dir, mask_name, vis_name):
    os.makedirs(output_dir, exist_ok=True)

    rgb_mask = utils.class_mask_to_rgb_mask(pred_mask)
    mask_path = os.path.join(output_dir, mask_name)
    Image.fromarray(rgb_mask).save(mask_path)
    print(f"✓ Mask: {mask_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_array)
    utils.show_multiclass_mask(pred_mask, axes[0, 1])
    axes[0, 1].set_title("Predicted overlay")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb_mask)
    axes[1, 0].set_title("Predicted mask")
    axes[1, 0].axis("off")

    prob_vis = np.max(class_probs, axis=0)
    im = axes[1, 1].imshow(prob_vis, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Confidence")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    vis_path = os.path.join(output_dir, vis_name)
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Visualization: {vis_path}")

    print("\nPrediction statistics:")
    for class_id, class_name in utils.TAMPER_NAMES.items():
        n = np.sum(pred_mask == class_id)
        print(f"  {class_name}: {n} pixels ({100.0 * n / pred_mask.size:.2f}%)")


def main():
    print("=" * 50)
    print("Single-image inference (no shape prompt)")
    print("=" * 50)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found: {IMAGE_PATH}")
        return
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_file = yaml.load(f, Loader=yaml.Loader)

    training_result_dir = TRAINING_RESULT_DIR
    if "xxx" in TRAINING_RESULT_DIR or not os.path.exists(TRAINING_RESULT_DIR):
        latest = find_latest_training_result()
        if latest:
            training_result_dir = latest
            print(f"Using latest training directory: {training_result_dir}")

    model, multiclass_head = load_model(
        config_file,
        SAM_CHECKPOINT,
        training_result_dir,
        LORA_WEIGHTS,
        MULTICLASS_HEAD_WEIGHTS,
        DEVICE,
    )
    processor = Samprocessor(model)

    pred_mask, class_probs, image_array = predict_image(
        model, multiclass_head, processor, IMAGE_PATH, DEVICE
    )

    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    save_results(
        pred_mask,
        class_probs,
        image_array,
        OUTPUT_DIR,
        f"{base}_predicted_mask.png",
        f"{base}_visualization.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
