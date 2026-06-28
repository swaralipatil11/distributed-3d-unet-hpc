#!/usr/bin/env python3
"""
Test script to run process_uploaded_file on generated mock ZIP volumes.
"""

import sys
from pathlib import Path
import asyncio

# Ensure backend/src is on the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

from src.main import process_uploaded_file

async def run_test():
    print("==================================================")
    print("TESTING DICOM UPLOAD AND PARSING WORKFLOW")
    print("==================================================")
    
    mock_zip_path = "./data/mock_dicom_volume.zip"
    if not Path(mock_zip_path).exists():
        print(f"Error: Mock zip not found at {mock_zip_path}. Run create_mock_dicom_zip.py first.")
        sys.exit(1)
        
    print(f"Loading and processing {mock_zip_path}...")
    try:
        # Run processing logic directly
        await process_uploaded_file("test_run_01", mock_zip_path)
        print("Slices streamed. Waiting 5 seconds for background inference to complete...")
        await asyncio.sleep(5)
        
        # Check if results npz exists
        results_path = Path("./data/results/test_run_01.npz")
        if not results_path.exists():
            raise FileNotFoundError("Results NPZ was not generated in ./data/results/")
            
        # Verify content
        import numpy as np
        import json
        archive = np.load(results_path)
        meta = json.loads(str(archive["metadata"]))
        
        print("--------------------------------------------------")
        print("VERIFIED PROCESSED RESULT NPZ:")
        print(f"Volume ID: {meta.get('volume_id')}")
        print(f"Device: {meta.get('device')}")
        print(f"Total Slices: {meta.get('total_packets')}")
        
        pat_meta = meta.get("patient_metadata", {})
        print(f"Patient Name: {pat_meta.get('patient_name')}")
        print(f"Patient ID: {pat_meta.get('patient_id')}")
        print(f"Study Date: {pat_meta.get('study_date')}")
        print(f"Description: {pat_meta.get('study_description')}")
        print("--------------------------------------------------")
        
        print("SUCCESS: Finished processing and verifying uploaded mock DICOM volume!")
        print("==================================================")
    except Exception as e:
        print(f"ERROR during DICOM processing/verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_test())
