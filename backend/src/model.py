#!/usr/bin/env python3
"""
Modular 3D U-Net architecture and compound loss functions (Dice + Cross-Entropy)
implemented natively in PyTorch. The model is fully compatible with torch.jit.script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Represent two consecutive Conv3d -> BatchNorm3d -> ReLU operations.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling path: MaxPool3d -> DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling path: ConvTranspose3d -> Concat skip connections -> DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Transposed convolution halves the channel dimension
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # Input to convolution block is upsampled features + skip connection features
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from lower/deeper level of decoder.
            x2: Skip connection feature map from corresponding encoder level.
        """
        x1 = self.up(x1)
        
        # Spatial size alignment (padding x1 if sizes differ slightly)
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]

        if diff_d > 0 or diff_h > 0 or diff_w > 0:
            x1 = F.pad(x1, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Modular 3D U-Net architecture for multi-class semantic segmentation.
    This model compiles cleanly with torch.jit.script.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, init_features: int = 16):
        """
        Args:
            in_channels: Number of input modalities (e.g. 4 for T1, T1c, T2, FLAIR).
            out_channels: Number of target segmentation classes.
            init_features: Number of filter features in the first convolutional level.
        """
        super().__init__()
        self.inc = DoubleConv(in_channels, init_features)
        self.down1 = Down(init_features, init_features * 2)
        self.down2 = Down(init_features * 2, init_features * 4)
        self.down3 = Down(init_features * 4, init_features * 8)

        self.up1 = Up(init_features * 8, init_features * 4)
        self.up2 = Up(init_features * 4, init_features * 2)
        self.up3 = Up(init_features * 2, init_features)
        
        self.outc = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        return logits


def to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts integer target tensor of shape (B, 1, H, W, D) into
    one-hot encoded float tensor of shape (B, C, H, W, D).
    """
    sh = list(tensor.shape)
    sh[1] = num_classes
    one_hot = torch.zeros(sh, dtype=torch.float32, device=tensor.device)
    one_hot.scatter_(1, tensor.long(), 1.0)
    return one_hot


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss function computed channel-wise and averaged.
    Can be configured to exclude background voxels.
    """
    def __init__(self, smooth: float = 1e-5, include_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Output logits of shape (B, C, H, W, D).
            targets: Voxel targets of shape (B, 1, H, W, D) or (B, H, W, D).
        """
        if targets.dim() == 4:
            targets = targets.unsqueeze(1)
            
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = to_one_hot(targets, num_classes)
        
        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]
            
        # Flatten over spatial dimensions (B, C, H*W*D)
        probs = probs.flatten(2)
        targets_one_hot = targets_one_hot.flatten(2)
        
        intersection = torch.sum(probs * targets_one_hot, dim=2)
        cardinality = torch.sum(probs + targets_one_hot, dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    """
    Compound Loss function blending Dice Loss and Cross-Entropy Loss to handle
    heavy 3D volumetric class imbalances.
    """
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, include_background: bool = True):
        super().__init__()
        self.dice_loss = DiceLoss(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss expects target shape to be (B, H, W, D) and long dtype
        ce_target = targets.squeeze(1) if targets.dim() == 5 else targets
        ce_target = ce_target.long()

        loss_ce = self.ce_loss(logits, ce_target)
        loss_dice = self.dice_loss(logits, targets)
        
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


if __name__ == "__main__":
    # Test scriptability and shape outputs
    print("Testing 3D U-Net scriptability and shapes...")
    model = UNet3D(in_channels=4, out_channels=4, init_features=16)
    
    # Check TorchScript compilation
    try:
        scripted_model = torch.jit.script(model)
        print("Success: 3D U-Net is scriptable!")
    except Exception as e:
        print(f"Error: Model scriptability failed: {e}")
        import sys
        sys.exit(1)
        
    # Check output shape
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, 4, 64, 64, 64), "Shape mismatch!"
    print("Shape test passed successfully.")
