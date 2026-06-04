#!/usr/bin/env python3
"""
Data ingestion pipeline for multi-modal 3D MRI volumes (T1, T1c, T2, FLAIR).
Uses MONAI dictionary-based transforms and multi-threaded PyTorch DataLoader.
"""

import os
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
)
from monai.data import Dataset, list_data_collate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_loader")


def get_transforms(roi_size: Tuple[int, int, int] = (128, 128, 128)) -> Compose:
    """
    Constructs and returns MONAI dictionary-based transforms for the pipeline.
    
    Args:
        roi_size: Spatial crop size for RandSpatialCropd.
        
    Returns:
        Compose: A composition of MONAI transforms.
    """
    logger.info("Initializing MONAI dictionary-based transforms...")
    return Compose([
        # Load image modalities and label maps from NIfTI files
        LoadImaged(keys=["image", "label"]),
        
        # Ensure that channel dimension is first, e.g., (C, H, W, D)
        # For image, moves modality dimension to first. For label, adds channel dimension.
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Resample spacing of image and label to uniform isotropic resolution (1.0mm)
        # Use bilinear (or trilinear) interpolation for images, nearest for labels
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        
        # Standardize orientation to RAS (Right, Anterior, Superior)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Normalize only image intensities channel-wise based on non-zero voxels
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        # Randomly crop a 3D sub-volume of specified shape (e.g. 128x128x128)
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False
        ),
    ])


def get_msd_data_dicts(data_dir: str) -> List[Dict[str, str]]:
    """
    Parses dataset.json in the specified dataset directory and returns
    a list of dictionaries mapping 'image' and 'label' to absolute paths.
    
    Args:
        data_dir: Root directory of the dataset containing dataset.json.
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing absolute file paths.
    """
    data_dir_path = Path(data_dir).resolve()
    json_path = data_dir_path / "dataset.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset.json at: {json_path}. "
            "Please make sure the dataset is downloaded and extracted."
        )
        
    logger.info(f"Reading dataset configuration from: {json_path}")
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    training_list = metadata.get("training", [])
    if not training_list:
        raise ValueError(f"No training files listed in {json_path}")
        
    data_dicts = []
    for item in training_list:
        # Resolve paths correctly (some might start with './' or have different formats)
        img_rel = item["image"].lstrip("./")
        lbl_rel = item["label"].lstrip("./")
        
        img_path = data_dir_path / img_rel
        lbl_path = data_dir_path / lbl_rel
        
        data_dicts.append({
            "image": str(img_path.resolve()),
            "label": str(lbl_path.resolve())
        })
        
    logger.info(f"Successfully loaded {len(data_dicts)} dataset items from {json_path}.")
    return data_dicts


def get_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader utilizing MONAI transforms.
    
    Args:
        data_dicts: List of data dictionaries.
        batch_size: Batch size.
        num_workers: Number of worker threads for batch fetching.
        pin_memory: If True, copies Tensors to CUDA pinned memory.
        shuffle: If True, shuffles the dataset.
        roi_size: Size of random crop.
        
    Returns:
        DataLoader: Production-ready PyTorch DataLoader.
    """
    logger.info("Creating MONAI Dataset and PyTorch DataLoader...")
    transforms = get_transforms(roi_size=roi_size)
    dataset = Dataset(data=data_dicts, transform=transforms)
    
    # We use MONAI's list_data_collate to properly collate lists of dictionaries
    # containing image and label tensors.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
    )
    return dataloader


def main():
    """
    Integration test main loop.
    Parses metadata, builds dataset, initializes data loader, and fetches 1 batch.
    """
    logger.info("Starting ingestion pipeline integration test...")
    
    # Define dataset root path
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = "./data/raw/Task01_BrainTumour"
        
    try:
        data_dicts = get_msd_data_dicts(dataset_dir)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        logger.info("Will attempt fallback paths or exit.")
        sys.exit(1)
        
    # For testing, we can limit to a small subset to save time / memory
    test_dicts = data_dicts[:4]
    logger.info(f"Running integration test with a subset of {len(test_dicts)} files.")
    
    # We use a crop size of 128x128x128 as requested.
    # Note: If memory-constrained on CPU, (96, 96, 96) can be used.
    roi_size = (128, 128, 128)
    
    start_time = time.time()
    dataloader = get_dataloader(
        data_dicts=test_dicts,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        roi_size=roi_size,
    )
    
    logger.info("DataLoader initialized. Fetching exactly 1 batch...")
    
    try:
        # Fetching 1 batch
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        # Check shapes
        images = batch["image"]
        labels = batch["label"]
        
        fetch_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("INTEGRATION TEST SUCCESSFUL!")
        logger.info(f"Time elapsed to load first batch: {fetch_time:.2f} seconds")
        logger.info(f"Batch image tensor shape (Expected: [1, 4, 128, 128, 128]): {images.shape}")
        logger.info(f"Batch label tensor shape (Expected: [1, 1, 128, 128, 128]): {labels.shape}")
        logger.info(f"Batch image tensor device: {images.device}")
        logger.info(f"Batch label tensor device: {labels.device}")
        logger.info(f"Is pinned memory used: {images.is_pinned()}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during batch loading: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
