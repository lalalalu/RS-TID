import torch
import torch.nn as nn


def _pick_group_count(num_channels: int) -> int:
    """为 GroupNorm 选择一个能整除通道数的组数。"""
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return groups
    return 1


class MulticlassHead(nn.Module):
    """
    将 SAM 的 mask logits 转换为多类别分割 logits。

    结构:
    1. 两层 3x3 卷积提取局部上下文
    2. GroupNorm 适配 batch_size=1 的训练
    3. Dropout2d 抑制过拟合
    4. 残差分支保留原始 mask logits 的可分性
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
        参数:
            mask_logits: [B,1,H,W] 或 [B,C,H,W] 的 mask logits

        返回:
            class_logits: [B,num_classes,H,W] 的多类别 logits
        """
        if mask_logits.dim() != 4:
            raise ValueError(f"mask_logits 应为 4 维张量，实际收到 shape={tuple(mask_logits.shape)}")

        if mask_logits.size(1) > 1:
            mask_logits = mask_logits[:, 0:1, :, :]

        features = self.feature_extractor(mask_logits)
        features = features + self.residual_proj(mask_logits)
        class_logits = self.classifier(features)
        return class_logits
