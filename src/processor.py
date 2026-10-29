from src.segment_anything.utils.transforms import ResizeLongestSide
from src.lora import LoRA_sam
import numpy as np
import torch
import PIL
from typing import Optional, Tuple


class Samprocessor:
    """
    Processor that transform the image and bounding box prompt with ResizeLongestSide and then pre process both data
        Arguments:
            sam_model: Model of SAM with LoRA weights initialised
        
        Return:
            inputs (list(dict)): list of dict in the input format of SAM containing (prompt key is a personal addition)
                image: Image preprocessed
                boxes: bounding box preprocessed
                prompt: bounding box of the original image

    """
    def __init__(self, sam_model: LoRA_sam):
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def __call__(self, image: PIL.Image, original_size: tuple, prompt: list, mask_prompt: Optional[np.ndarray] = None) -> dict:
        # Processing of the image
        image_torch = self.process_image(image, original_size)

        # Transform input prompts
        box_torch = self.process_prompt(prompt, original_size)
        
        # Process the mask prompt, if provided.
        mask_input_torch = None
        if mask_prompt is not None:
            mask_input_torch = self.process_mask_prompt(mask_prompt, original_size)

        inputs = {"image": image_torch, 
                  "original_size": original_size,
                 "boxes": box_torch,
                 "prompt" : prompt}
        
        # Add the mask prompt, if available.
        if mask_input_torch is not None:
            inputs["mask_inputs"] = mask_input_torch
        
        return inputs


    def process_image(self, image: PIL.Image, original_size: tuple) -> torch.tensor:
        """
        Preprocess the image to make it to the input format of SAM

        Arguments:
            image: Image loaded in PIL
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the image preprocessed
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return input_image_torch

    def process_prompt(self, box: list, original_size: tuple) -> torch.tensor:
        """
        Preprocess the prompt (bounding box) to make it to the input format of SAM

        Arguments:
            box: Bounding bounding box coordinates in [XYXY]
            original_size: tuple of the original image size (H,W)

        Return:
            (tensor): Tensor of the prompt preprocessed
        """
        # We only use boxes
        box_torch = None
        nd_box = np.array(box).reshape((1,4))
        box = self.transform.apply_boxes(nd_box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, :]

        return box_torch

    def process_mask_prompt(self, mask: np.ndarray, original_size: tuple) -> torch.tensor:
        """
        Resize a mask prompt to the SAM mask input size (256x256).
        
        Arguments:
            mask: Binary mask array of shape [H, W].
            original_size: Original image size (H, W).
        
        Return:
            (tensor): Processed mask tensor of shape [1, 1, 256, 256].
        """
        import torch.nn.functional as F
        
        # Ensure the mask is a NumPy array.
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure a contiguous array with the correct dtype.
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        
        # Add a channel dimension to an [H, W] mask.
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]  # [1, H, W]
        elif len(mask.shape) == 3:
            # Convert [H, W, 1] or [1, H, W] to [1, H, W].
            if mask.shape[0] != 1:
                mask = mask.transpose(2, 0, 1)[:1, ...]  # Select the first channel.
        
        # Ensure contiguous storage after adding dimensions.
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        
        # Resize to 256x256, the standard SAM mask input size.
        # Use torch.tensor instead of torch.from_numpy for a reliable copy.
        mask_tensor = torch.tensor(mask, dtype=torch.float32)  # [1, H, W]
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W]
        mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Move the tensor to the correct device.
        mask_resized = mask_resized.to(self.device)
        
        return mask_resized  # [1, 1, 256, 256]


    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
