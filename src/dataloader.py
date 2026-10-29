import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from src.processor import Samprocessor
import src.utils as utils


def clean_path(path_str):
    """Strip surrounding quotes and optional raw-string notation from a path."""
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
        config_file (dict): Configuration containing DATASET paths.
        processor (Samprocessor): Image and bounding-box preprocessing.
        mode (str): "train" | "val" | "test"

    Return:
        dict: image, original_size, boxes, ground_truth_mask, and processor outputs.
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            train_path = clean_path(config_file["DATASET"]["TRAIN_PATH"])

            images_dir = os.path.join(train_path, 'images')
            masks_dir = os.path.join(train_path, 'masks')

            if not os.path.exists(images_dir):
                raise ValueError(f"Training image directory does not exist: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"Training mask directory does not exist: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])

            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"No image files found in {images_dir} (supported formats: jpg, jpeg, png)")

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
                    print(f"Warning: No matching mask found for image {img_path}")

            if len(valid_pairs) == 0:
                raise ValueError(f"No mask files found in {masks_dir}")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ Loaded training set: {len(self.img_files)} image-mask pairs ({len(self.img_files)} pairs in total)")
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
                raise ValueError(f"Validation image directory does not exist: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"Validation mask directory does not exist: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])
            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"No image files found in {images_dir}")

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
                raise ValueError(f"No mask files found in {masks_dir}")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ Loaded validation set: {len(self.img_files)} image-mask pairs ({len(self.img_files)} pairs in total)")
        else:  # test mode
            test_path = clean_path(config_file["DATASET"]["TEST_PATH"])

            images_dir = os.path.join(test_path, 'images')
            masks_dir = os.path.join(test_path, 'masks')

            if not os.path.exists(images_dir):
                raise ValueError(f"Test image directory does not exist: {images_dir}")
            if not os.path.exists(masks_dir):
                raise ValueError(f"Test mask directory does not exist: {masks_dir}")

            img_files_set = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                found_files = glob.glob(os.path.join(images_dir, ext))
                img_files_set.update([os.path.normpath(os.path.abspath(f)) for f in found_files])

            self.img_files = sorted(list(img_files_set))

            if len(self.img_files) == 0:
                raise ValueError(f"No image files found in {images_dir}")

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
                raise ValueError(f"No mask files found in {masks_dir}")

            self.img_files, self.mask_files = zip(*valid_pairs)
            self.img_files = list(self.img_files)
            self.mask_files = list(self.mask_files)

            print(f"✓ Loaded test set: {len(self.img_files)} image-mask pairs ({len(self.img_files)} pairs in total)")

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
                f"Image and mask spatial dimensions do not match: {img_path} image HW={original_size}, "
                f"mask {mask_path} shape={mask_array.shape[:2]}"
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
