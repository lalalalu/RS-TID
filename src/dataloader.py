import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from src.processor import Samprocessor
import src.utils as utils


def clean_path(path_str):
    """清理路径字符串，去除可能的 r"..." 或引号"""
    path_str = str(path_str).strip()
    if path_str.startswith('r"') and path_str.endswith('"'):
        path_str = path_str[2:-1]
    elif path_str.startswith("r'") and path_str.endswith("'"):
        path_str = path_str[2:-1]
    elif path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]
    elif path_str.startswith("'") and path_str.endswith("'"):
        path_str = path_str[1:-1]
    return path_str.strip()


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        config_file (dict): 含 DATASET 路径
        processor (Samprocessor): 图像与 bbox 预处理
        mode (str): "train" | "val" | "test"

    Return:
        dict: image, original_size, boxes, ground_truth_mask（及 processor 生成的键）
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            train_path = clean_path(config_file["DATASET"]["TRAIN_PATH"])

            images_dir = os.path.join(train_path, 'images')
            masks_dir = os.path.join(train_path, 'masks')

            if not os.path.exists(images_dir):
                raise ValueError(f"训练集图像目录不存在: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"训练集掩码目录不存在: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])

            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"在 {images_dir} 中未找到任何图像文件（支持jpg、jpeg、png格式）")

            valid_pairs = []
            for img_path in self.img_files:
                base_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(base_name)[0]
                mask_path = None
                for mask_ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    potential_mask = os.path.join(masks_dir, name_without_ext + mask_ext)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break

                if mask_path:
                    valid_pairs.append((img_path, mask_path))
                else:
                    print(f"警告: 未找到图像 {img_path} 对应的掩码文件")

            if len(valid_pairs) == 0:
                raise ValueError(f"在 {masks_dir} 中未找到任何掩码文件")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ 加载训练集: 找到 {len(self.img_files)} 个图像-掩码对 (共 {len(self.img_files)} 对)")
        elif mode == "val":
            val_path = config_file["DATASET"].get("VAL_PATH", None)
            if val_path is None:
                train_path = clean_path(config_file["DATASET"]["TRAIN_PATH"])
                val_path = train_path.replace("train", "val")
            else:
                val_path = clean_path(val_path)

            images_dir = os.path.join(val_path, 'images')
            masks_dir = os.path.join(val_path, 'masks')

            if not os.path.exists(images_dir):
                raise ValueError(f"验证集图像目录不存在: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"验证集掩码目录不存在: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])
            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"在 {images_dir} 中未找到任何图像文件")

            valid_pairs = []
            for img_path in self.img_files:
                base_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(base_name)[0]
                mask_path = None
                for mask_ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    potential_mask = os.path.join(masks_dir, name_without_ext + mask_ext)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break

                if mask_path:
                    valid_pairs.append((img_path, mask_path))

            if len(valid_pairs) == 0:
                raise ValueError(f"在 {masks_dir} 中未找到任何掩码文件")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ 加载验证集: 找到 {len(self.img_files)} 个图像-掩码对 (共 {len(self.img_files)} 对)")
        else:  # test mode
            test_path = clean_path(config_file["DATASET"]["TEST_PATH"])

            images_dir = os.path.join(test_path, 'images')
            masks_dir = os.path.join(test_path, 'masks')

            if not os.path.exists(images_dir):
                raise ValueError(f"测试集图像目录不存在: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"测试集掩码目录不存在: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])

            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"在 {images_dir} 中未找到任何图像文件")

            valid_pairs = []
            for img_path in self.img_files:
                base_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(base_name)[0]
                mask_path = None
                for mask_ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    potential_mask = os.path.join(masks_dir, name_without_ext + mask_ext)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break

                if mask_path:
                    valid_pairs.append((img_path, mask_path))

            if len(valid_pairs) == 0:
                raise ValueError(f"在 {masks_dir} 中未找到任何掩码文件")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ 加载测试集: 找到 {len(self.img_files)} 个图像-掩码对 (共 {len(self.img_files)} 对)")

        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int) -> dict:
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode != 'RGB':
            mask = mask.convert('RGB')
        mask_array = np.array(mask)
        original_size = tuple(image.size)[::-1]  # PIL (W,H) -> (H,W)
        if mask_array.shape[:2] != (original_size[0], original_size[1]):
            raise ValueError(
                f"图像与掩码空间尺寸不一致: {img_path} 图像 HW={original_size}, "
                f"掩码 {mask_path} shape={mask_array.shape[:2]}"
            )

        ground_truth_mask = utils.rgb_mask_to_class_mask(mask_array)
        ground_truth_mask = np.ascontiguousarray(ground_truth_mask, dtype=np.int64)

        USE_WHOLE_IMAGE_BBOX = True
        if USE_WHOLE_IMAGE_BBOX:
            H, W = original_size
            box = [0, 0, W, H]
        else:
            box = utils.get_bounding_box(ground_truth_mask)

        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask, dtype=torch.long)
        return inputs


def collate_fn(batch: list) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)
