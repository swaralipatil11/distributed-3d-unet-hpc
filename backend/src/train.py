#!/usr/bin/env python3
"""
Training and evaluation runner for the 3D U-Net volumetric inference pipeline.
Leverages Automatic Mixed Precision (AMP) for FP16 training to minimize VRAM usage.
"""

import os
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_loader import get_msd_data_dicts, get_dataloader
from src.model import UNet3D, CompoundLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D U-Net on MSD BrainTumour dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw/Task01_BrainTumour",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train (default: 2 for quick verification).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--init-features",
        type=int,
        default=16,
        help="Number of initial filters in U-Net.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./deploy",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="If > 0, limits the number of batches per epoch (useful for quick verification).",
    )
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    limit_batches: int = 0,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    
    for batch_idx, batch in enumerate(loader):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break
            
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with Automatic Mixed Precision (AMP)
        # Note: on GPU this activates FP16; on CPU it runs with default precision.
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Backward and optimizer step
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        steps += 1
        
    return running_loss / steps if steps > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    limit_batches: int = 0,
) -> float:
    model.eval()
    running_loss = 0.0
    steps = 0
    
    for batch_idx, batch in enumerate(loader):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break
            
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item()
        steps += 1
        
    return running_loss / steps if steps > 0 else 0.0


def main():
    args = parse_args()
    
    # Resolve and create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "unet_model.pt"
    
    print("=" * 60)
    print("STARTING 3D U-NET TRAINING LOOP")
    print(f"Dataset root: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Init features: {args.init_features}")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"AMP/FP16 Scaler: ENABLED")
    else:
        print("AMP/FP16 Scaler: DISABLED (GPU unavailable)")
    print("-" * 60)
    
    # Resolve dataset dictionaries
    try:
        data_dicts = get_msd_data_dicts(args.data_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Split into train and validation splits
    val_size = int(len(data_dicts) * args.val_split)
    if val_size == 0 or val_size == len(data_dicts):
        # Fallback for small datasets (e.g. mock test case)
        print("Warning: Dataset too small for splitting. Reusing dataset for validation.")
        train_dicts = data_dicts
        val_dicts = data_dicts
    else:
        train_dicts = data_dicts[:-val_size]
        val_dicts = data_dicts[-val_size:]
        
    print(f"Training samples: {len(train_dicts)}")
    print(f"Validation samples: {len(val_dicts)}")
    
    # Instantiate PyTorch DataLoaders
    # Training uses standard rand spatial crop 128x128x128
    roi_size = (128, 128, 128)
    
    train_loader = get_dataloader(
        data_dicts=train_dicts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        roi_size=roi_size,
    )
    
    val_loader = get_dataloader(
        data_dicts=val_dicts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        roi_size=roi_size,
    )
    
    # Build 3D U-Net model
    model = UNet3D(
        in_channels=4,  # T1, T1c, T2, FLAIR
        out_channels=4,  # background, edema, non-enhancing, enhancing
        init_features=args.init_features,
    ).to(device)
    
    # Loss, Optimizer, and LR Scheduler
    # include_background=True for Dice Loss, or False to exclude background voxels
    criterion = CompoundLoss(dice_weight=1.0, ce_weight=1.0, include_background=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP GradScaler (only active on CUDA device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            limit_batches=args.limit_batches,
        )
        
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_batches,
        )
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, str(checkpoint_path))
            print(f"  --> Saved best checkpoint to {checkpoint_path}")
            
    print("-" * 60)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY.")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
