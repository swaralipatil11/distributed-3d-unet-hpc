#!/usr/bin/env python3
"""
Automated data acquisition script for the 3D Volumetric Inference Pipeline.
Programmatically fetches and extracts the MSD Task01_BrainTumour (BraTS) dataset.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_acquisition")

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the Task01_BrainTumour (BraTS) MSD dataset."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        help="Public URL for the Task01_BrainTumour tar archive.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for storing data.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="./data/raw",
        help="Target directory to extract raw volumes.",
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Remove downloaded tar file after extraction to save disk space.",
    )
    args = parser.parse_args()

    # Resolve paths
    data_root = Path(args.data_dir).resolve()
    extract_root = Path(args.extract_dir).resolve()
    tar_filepath = data_root / "Task01_BrainTumour.tar"

    # Ensure directories exist
    data_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target archive destination: {tar_filepath}")
    logger.info(f"Extraction destination: {extract_root}")

    # Import MONAI download utility
    try:
        from monai.apps import download_and_extract
        logger.debug("Imported download_and_extract from monai.apps")
    except ImportError:
        try:
            from monai.apps.utils import download_and_extract
            logger.debug("Imported download_and_extract from monai.apps.utils")
        except ImportError:
            logger.error("MONAI is not installed or download_and_extract could not be imported.")
            logger.error("Please run: pip install -r requirements.txt")
            sys.exit(1)

    logger.info("Initializing download and extraction...")
    logger.info(f"Source URL: {args.url}")
    
    try:
        download_and_extract(
            url=args.url,
            filepath=str(tar_filepath),
            output_dir=str(extract_root),
        )
        if args.remove_archive and tar_filepath.exists():
            logger.info(f"Removing downloaded archive: {tar_filepath}")
            tar_filepath.unlink()
        logger.info("Download and extraction completed successfully.")
    except Exception as e:
        logger.error(f"Failed during download/extraction: {e}", exc_info=True)
        sys.exit(1)

    # Perform validation checks
    logger.info("Running validation checks on extracted dataset...")
    validate_dataset(extract_root)

def validate_dataset(extract_root: Path):
    # Expecting Task01_BrainTumour directory inside raw
    dataset_dir = extract_root / "Task01_BrainTumour"
    if not dataset_dir.exists():
        # Fallback: maybe extracted directly into extract_root
        dataset_dir = extract_root

    dataset_json_path = dataset_dir / "dataset.json"
    images_tr_path = dataset_dir / "imagesTr"
    labels_tr_path = dataset_dir / "labelsTr"

    errors = 0
    if not dataset_json_path.exists():
        logger.warning(f"Metadata file 'dataset.json' not found at {dataset_json_path}")
        errors += 1
    else:
        logger.info("Found dataset.json")
        try:
            with open(dataset_json_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Dataset name: {metadata.get('name', 'N/A')}")
            logger.info(f"Dataset description: {metadata.get('description', 'N/A')}")
            logger.info(f"Dataset release: {metadata.get('release', 'N/A')}")
            logger.info(f"Target modality: {metadata.get('modality', 'N/A')}")
            logger.info(f"Target labels: {metadata.get('labels', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to parse dataset.json: {e}")
            errors += 1

    # Count image files
    img_files = []
    if images_tr_path.exists():
        img_files = list(images_tr_path.glob("*.nii.gz"))
        logger.info(f"Counted training volumes (imagesTr): {len(img_files)}")
    else:
        logger.error(f"Training images directory 'imagesTr' not found at {images_tr_path}")
        errors += 1

    # Count label files
    lbl_files = []
    if labels_tr_path.exists():
        lbl_files = list(labels_tr_path.glob("*.nii.gz"))
        logger.info(f"Counted training labels (labelsTr): {len(lbl_files)}")
    else:
        logger.error(f"Training labels directory 'labelsTr' not found at {labels_tr_path}")
        errors += 1

    # Detailed consistency check
    if len(img_files) > 0 and len(lbl_files) > 0:
        if len(img_files) == len(lbl_files):
            logger.info("Validation PASSED: Training volumes count matches labels count.")
        else:
            logger.warning(
                f"Validation WARNING: Mismatch in count! Images: {len(img_files)}, Labels: {len(lbl_files)}"
            )
            errors += 1
    else:
        logger.error("Validation FAILED: No training images or labels found.")
        errors += 1

    if errors == 0:
        logger.info("Dataset validation completed with 0 errors/warnings.")
    else:
        logger.warning(f"Dataset validation completed with {errors} issue(s). Check output above.")

if __name__ == "__main__":
    main()
