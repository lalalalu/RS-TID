import torch
import torch.nn as nn


def _pick_group_count(num_channels: int) -> int:
    """Choose a GroupNorm group count that divides the channel count."""
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return groups
    return 1


class MulticlassHead(nn.Module):
    """
    Convert SAM mask logits into multiclass segmentation logits.

    Architecture:
    1. Two 3x3 convolutions extract local context.
    2. GroupNorm supports training with batch_size=1.
    3. Dropout2d reduces overfitting.
    4. A residual branch preserves information from the input mask logits.
    """

    def __init__(self, num_classes=4, in_channels=1, hidden_channels=32, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        groups = _pick_group_count(hidden_channels)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.residual_proj = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            mask_logits: Mask logits of shape [B,1,H,W] or [B,C,H,W].

        Returns:
            class_logits: Multiclass logits of shape [B,num_classes,H,W].
        """
        if mask_logits.dim() != 4:
            raise ValueError(f"mask_logits must be a 4D tensor; received shape={tuple(mask_logits.shape)}")

        if mask_logits.size(1) > 1:
            mask_logits = mask_logits[:, 0:1, :, :]

        features = self.feature_extractor(mask_logits)
        features = features + self.residual_proj(mask_logits)
        class_logits = self.classifier(features)
        return class_logits
