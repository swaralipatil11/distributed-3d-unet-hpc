#!/usr/bin/env python3
"""
Test script to verify the DICOM segmentation export workflow.
"""

import sys
import zipfile
from pathlib import Path
import asyncio
import numpy as np

# Ensure backend/src is on the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

import pydicom
from src.main import process_uploaded_file, download_volume_prediction_dicom

async def run_test():
    print("==================================================")
    print("TESTING DICOM EXPORT WORKFLOW")
    print("==================================================")
    
    volume_id = "test_dicom_export_run"
    mock_zip_path = "./data/mock_dicom_volume.zip"
    
    if not Path(mock_zip_path).exists():
        print(f"Error: Mock zip not found at {mock_zip_path}. Run tests/create_mock_dicom_zip.py first.")
        sys.exit(1)
        
    print(f"1. Processing mock ZIP archive as {volume_id} to create backup & results...")
    try:
        # Save a copy to dicom_sources first (mimicking the upload endpoint backup logic)
        backup_dir = Path("./data/dicom_sources")
        backup_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(mock_zip_path, backup_dir / f"{volume_id}.zip")
        
        # Run upload processing to generate the result NPZ
        await process_uploaded_file(volume_id, mock_zip_path)
        print("Slices streamed. Waiting for background inference to complete...")
        await asyncio.sleep(5)
        
        # Check if results NPZ exists
        results_path = Path(f"./data/results/{volume_id}.npz")
        if not results_path.exists():
            raise FileNotFoundError("Results NPZ was not generated in ./data/results/")
            
        print(f"2. Invoking download_volume_prediction_dicom endpoint for {volume_id}...")
        response = await download_volume_prediction_dicom(volume_id)
        
        # Check if response is successful FileResponse
        from fastapi.responses import FileResponse
        if not isinstance(response, FileResponse):
            raise TypeError(f"Endpoint did not return FileResponse. Got: {type(response)}")
            
        export_zip_path = Path(response.path)
        print(f"Generated DICOM segmentation zip: {export_zip_path} (Size: {export_zip_path.stat().st_size} bytes)")
        
        # Verify ZIP contains valid DICOM files and modified tags
        print("3. Inspecting exported DICOM series...")
        extract_dir = Path(f"./data/uploads/test_export_verify_{volume_id}")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(export_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        dcm_files = sorted(list(extract_dir.glob("*.dcm")))
        print(f"Found {len(dcm_files)} DICOM files in the exported ZIP.")
        if len(dcm_files) == 0:
            raise ValueError("No DICOM files found in the exported ZIP archive.")
            
        # Read the first file and check modified tags
        first_dcm = dcm_files[0]
        ds = pydicom.dcmread(str(first_dcm))
        
        print("--------------------------------------------------")
        print("EXPORTED DICOM METADATA VERIFICATION:")
        print(f"Series Description: {getattr(ds, 'SeriesDescription', 'NOT FOUND')}")
        print(f"Series Instance UID: {getattr(ds, 'SeriesInstanceUID', 'NOT FOUND')}")
        print(f"SOP Instance UID: {getattr(ds, 'SOPInstanceUID', 'NOT FOUND')}")
        print(f"Rows x Columns: {getattr(ds, 'Rows', 0)} x {getattr(ds, 'Columns', 0)}")
        
        # Check pixel values map to [0, 80, 160, 240]
        pixels = ds.pixel_array
        unique_vals = np.unique(pixels)
        print(f"Unique pixel values in slice: {unique_vals}")
        print("--------------------------------------------------")
        
        # Verify SOPInstanceUID contains .99 overlay
        if ".99" not in str(getattr(ds, 'SOPInstanceUID', '')):
            raise ValueError("SOPInstanceUID does not contain the .99 overlay suffix.")
            
        # Verify SeriesDescription matches expected
        if getattr(ds, 'SeriesDescription', '') != "3D U-Net Brain Tumor Segmentation":
            raise ValueError(f"SeriesDescription did not match. Got: {ds.SeriesDescription}")
            
        print("SUCCESS: DICOM Export workflow verified successfully!")
        print("==================================================")
        
        # Clean up test files
        shutil.rmtree(extract_dir)
        if export_zip_path.exists():
            export_zip_path.unlink()
        if (backup_dir / f"{volume_id}.zip").exists():
            (backup_dir / f"{volume_id}.zip").unlink()
        if results_path.exists():
            results_path.unlink()
            
    except Exception as e:
        print(f"ERROR during DICOM export testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_test())
