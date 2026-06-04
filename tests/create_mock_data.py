#!/usr/bin/env python3
"""
Generates a mock Task01_BrainTumour dataset for verification and unit/integration testing.
"""

import os
import json
import logging
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("create_mock_data")


def create_mock_dataset(output_dir: str = "./data/mock_Task01_BrainTumour"):
    """
    Creates a folder with mock dataset.json and small NIfTI files.
    """
    out_path = Path(output_dir).resolve()
    images_tr = out_path / "imagesTr"
    labels_tr = out_path / "labelsTr"
    
    # Create directories
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating mock dataset at {out_path}...")
    
    # We use a size slightly larger than crop size (128) to avoid cropping out of bounds,
    # e.g., (140, 140, 140)
    spatial_shape = (140, 140, 140)
    
    # 1. Create mock image (4 modalities)
    logger.info("Generating mock image array...")
    # Using random data
    img_data = np.random.randn(*spatial_shape, 4).astype(np.float32)
    # Scale to typical MRI values
    img_data = (img_data * 100) + 500
    img_data = np.clip(img_data, 0, None)
    
    # 2. Create mock label (integers 0 to 3)
    logger.info("Generating mock label array...")
    lbl_data = np.zeros(spatial_shape, dtype=np.int16)
    # Add a sphere in the center representing tumor
    x, y, z = np.ogrid[:spatial_shape[0], :spatial_shape[1], :spatial_shape[2]]
    center = (70, 70, 70)
    distance_sq = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    lbl_data[distance_sq <= 30**2] = 1
    lbl_data[distance_sq <= 15**2] = 2
    lbl_data[distance_sq <= 5**2] = 3
    
    # Affine matrix (identity with 1.0 spacing)
    affine = np.eye(4)
    
    # Save NIfTI files
    img_nii = nib.Nifti1Image(img_data, affine)
    lbl_nii = nib.Nifti1Image(lbl_data, affine)
    
    img_file = images_tr / "BRATS_001.nii.gz"
    lbl_file = labels_tr / "BRATS_001.nii.gz"
    
    logger.info(f"Saving mock image to {img_file}...")
    nib.save(img_nii, str(img_file))
    
    logger.info(f"Saving mock label to {lbl_file}...")
    nib.save(lbl_nii, str(lbl_file))
    
    # 3. Create dataset.json
    dataset_info = {
        "name": "Task01_BrainTumour",
        "description": "Mock Brain Tumour Segmentation Dataset",
        "reference": "Mock",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 15/01/2018",
        "tensorImageSize": "4D",
        "modality": {
            "0": "T1",
            "1": "T1c",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing tumor",
            "3": "enhancing tumor"
        },
        "numTraining": 1,
        "numTest": 0,
        "training": [
            {
                "image": "./imagesTr/BRATS_001.nii.gz",
                "label": "./labelsTr/BRATS_001.nii.gz"
            }
        ],
        "test": []
    }
    
    json_file = out_path / "dataset.json"
    logger.info(f"Saving dataset.json to {json_file}...")
    with open(json_file, "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    logger.info("Mock dataset created successfully.")


if __name__ == "__main__":
    create_mock_dataset()
