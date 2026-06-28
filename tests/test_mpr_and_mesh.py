#!/usr/bin/env python3
"""
Integration test for MPR slicing (planes parameter) and 3D mesh points endpoints.
"""

import sys
import json
from pathlib import Path
import asyncio

# Ensure backend/src is on the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

from src.main import get_slice_modality, get_slice_label, get_volume_mesh

async def run_test():
    print("==================================================")
    print("TESTING MPR SLICING & 3D MESH ENDPOINTS")
    print("==================================================")
    
    volume_id = "test_run_01"
    results_path = Path(f"./data/results/{volume_id}.npz")
    
    if not results_path.exists():
        print(f"Error: Processed volume not found at {results_path}. Run tests/test_upload_api.py first.")
        sys.exit(1)
        
    print("1. Testing MPR Modality slices for Axial, Sagittal, and Coronal...")
    for plane in ["axial", "sagittal", "coronal"]:
        try:
            response = await get_slice_modality(volume_id=volume_id, slice_idx=5, modality_idx=0, plane=plane)
            from fastapi.responses import StreamingResponse
            if not isinstance(response, StreamingResponse):
                raise TypeError(f"Expected StreamingResponse, got: {type(response)}")
            print(f"   [SUCCESS] Modality slice for plane '{plane}' generated successfully.")
        except Exception as e:
            print(f"   [FAILED] plane '{plane}' modality slice failed: {e}")
            sys.exit(1)
            
    print("2. Testing MPR Label overlays for Axial, Sagittal, and Coronal...")
    for plane in ["axial", "sagittal", "coronal"]:
        try:
            response = await get_slice_label(volume_id=volume_id, slice_idx=5, classes="1,2,3", plane=plane)
            from fastapi.responses import StreamingResponse
            if not isinstance(response, StreamingResponse):
                raise TypeError(f"Expected StreamingResponse, got: {type(response)}")
            print(f"   [SUCCESS] Label overlay slice for plane '{plane}' generated successfully.")
        except Exception as e:
            print(f"   [FAILED] plane '{plane}' label overlay slice failed: {e}")
            sys.exit(1)
            
    print("3. Testing 3D mesh boundary points endpoint...")
    try:
        response = await get_volume_mesh(volume_id=volume_id)
        if not isinstance(response, dict) or "points" not in response:
            raise TypeError(f"Expected dict response with 'points' key, got: {type(response)}")
            
        points = response["points"]
        print(f"   [SUCCESS] 3D mesh points generated successfully. Total points: {len(points)}")
        
        if len(points) > 0:
            first_pt = points[0]
            print(f"   Sample point: coord=({first_pt[0]:.3f}, {first_pt[1]:.3f}, {first_pt[2]:.3f}), label={first_pt[3]}")
            # Verify coordinates are normalized between -1 and 1
            for coord in first_pt[:3]:
                if not (-1.0 <= coord <= 1.0):
                    raise ValueError(f"Voxel coordinates are not normalized: {coord}")
            print("   [SUCCESS] Coordinates normalization checked successfully.")
            
    except Exception as e:
        print(f"   [FAILED] 3D mesh points failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("==================================================")
    print("ALL MPR & 3D ENDPOINTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_test())
