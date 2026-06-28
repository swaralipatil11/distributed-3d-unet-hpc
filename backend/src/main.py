#!/usr/bin/env python3
"""
Distributed Worker Engine and Web API implemented using FastAPI and aiokafka.
Consumes slice packets from Kafka, reassembles 3D volumes in memory,
executes inference using JIT compiled TorchScript UNet3D, logs latency,
and exposes REST endpoints and WebSockets for the frontend visualizer.
"""

import os
import sys
import json
import time
import base64
import asyncio
import io
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, BackgroundTasks, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import shutil
import pydicom
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("worker_engine")

from fastapi.middleware.cors import CORSMiddleware
import monai.transforms as mt

# FastAPI App
app = FastAPI(title="3D MRI Volumetric Inference Worker", version="1.0.0")

# Add CORS Middleware to support decoupled frontends (e.g. Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.jit.ScriptModule = None
volume_cache: Dict[str, List[Any]] = {} # Maps volume_id -> list of slices
volume_metadata: Dict[str, Dict[str, Any]] = {}
kafka_connected = False

# Simple cache for processed volumes to avoid loading/decompressing files on every slice request
# Stores volume_id -> {"image": np.ndarray, "label": np.ndarray, "timestamp": float}
processed_volume_cache: Dict[str, Dict[str, Any]] = {}
PROCESSED_CACHE_LIMIT = 5 # Cache up to 5 volumes in memory

def get_processed_volume(volume_id: str) -> Dict[str, Any]:
    global processed_volume_cache
    
    # If already cached, update timestamp and return
    if volume_id in processed_volume_cache:
        processed_volume_cache[volume_id]["timestamp"] = time.time()
        return processed_volume_cache[volume_id]
        
    results_path = Path(f"./data/results/{volume_id}.npz")
    if not results_path.exists():
        raise FileNotFoundError(f"Volume results not found for {volume_id}")
        
    archive = np.load(results_path)
    image = archive["image"]
    label = archive["label"]
    
    # Evict oldest if limit reached
    if len(processed_volume_cache) >= PROCESSED_CACHE_LIMIT:
        oldest_key = min(processed_volume_cache.keys(), key=lambda k: processed_volume_cache[k]["timestamp"])
        del processed_volume_cache[oldest_key]
        logger.info(f"Evicted volume {oldest_key} from processed cache.")
        
    processed_volume_cache[volume_id] = {
        "image": image,
        "label": label,
        "timestamp": time.time()
    }
    return processed_volume_cache[volume_id]


async def cleanup_cache_loop():
    """
    Background task that runs periodically to remove stale volumes
    from the in-memory cache to prevent memory leaks.
    Also cleans up old exported NIfTI prediction files.
    """
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale_threshold = 300 # 5 minutes
        stale_volumes = []
        for volume_id, meta in list(volume_metadata.items()):
            if now - meta.get("start_time", now) > stale_threshold:
                stale_volumes.append(volume_id)
                
        for volume_id in stale_volumes:
            logger.info(f"Cleaning up stale cache for volume {volume_id} to prevent memory leak.")
            if volume_id in volume_cache:
                del volume_cache[volume_id]
            if volume_id in volume_metadata:
                del volume_metadata[volume_id]

        # Clean up old exported NIfTI and DICOM zip files (older than 24 hours)
        try:
            export_dir = Path("./data/exports")
            if export_dir.exists():
                for file in list(export_dir.glob("*.nii.gz")) + list(export_dir.glob("*.zip")):
                    if now - file.stat().st_mtime > 86400: # 24 hours
                        logger.info(f"Removing old exported file: {file.name}")
                        file.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up old exported files: {e}")

        # Clean up old backup source DICOM zip files (older than 24 hours)
        try:
            dicom_sources_dir = Path("./data/dicom_sources")
            if dicom_sources_dir.exists():
                for file in dicom_sources_dir.glob("*.zip"):
                    if now - file.stat().st_mtime > 86400: # 24 hours
                        logger.info(f"Removing old source DICOM backup zip: {file.name}")
                        file.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up old backup source DICOM files: {e}")


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # Silently ignore broken connections
                pass

manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """
    On startup, loads the compiled TorchScript model, ensures directories exist,
    and starts the Kafka consumer loop in the background.
    """
    global model
    
    # Ensure folders exist
    Path("./data/results").mkdir(parents=True, exist_ok=True)
    Path("./data/uploads").mkdir(parents=True, exist_ok=True)
    
    # Start the Kafka consumer background task
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "mri-inference-requests")
    asyncio.create_task(consume_kafka_loop(bootstrap_servers, topic))
    
    # Start cache cleanup task
    asyncio.create_task(cleanup_cache_loop())
    
    model_path = "./deploy/model_trace.pt"
    logger.info(f"Loading compiled TorchScript model from {model_path} onto {device}...")
    
    if not Path(model_path).exists():
        logger.error(f"TorchScript model not found at {model_path}. Please run src/export.py first.")
        # Do not crash startup so simulation/dev environment is still testable
        return
        
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        logger.info("TorchScript model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load TorchScript model: {e}")
        return


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            await websocket.send_text(f"ACK: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Health check endpoint
@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    global kafka_connected
    is_healthy = model is not None
    return {
        "status": "healthy" if is_healthy else "degraded",
        "device": str(device),
        "active_streams": len(volume_cache),
        "kafka_connected": kafka_connected,
        "model_loaded": model is not None
    }


# API: List all processed runs
@app.get("/api/volumes")
async def list_volumes():
    results_dir = Path("./data/results")
    if not results_dir.exists():
        return []
        
    volumes = []
    for file in results_dir.glob("*.npz"):
        volume_id = file.stem
        try:
            # Quick read of metadata
            archive = np.load(file, allow_pickle=True)
            meta = json.loads(str(archive["metadata"]))
            meta["id"] = volume_id
            volumes.append(meta)
        except Exception as e:
            logger.error(f"Failed to read archive {file.name}: {e}")
            
    return volumes


# API: Download prediction as NIfTI (.nii.gz) file
@app.get("/api/volume/{volume_id}/download")
async def download_volume_prediction(volume_id: str):
    try:
        results_path = Path(f"./data/results/{volume_id}.npz")
        if not results_path.exists():
            return Response(
                content=json.dumps({"status": "error", "message": "Volume results not found."}),
                status_code=status.HTTP_404_NOT_FOUND,
                media_type="application/json"
            )
            
        archive = np.load(results_path)
        label = archive["label"] # Shape (128, 128, 128)
        meta = json.loads(str(archive["metadata"]))
        
        affine_list = meta.get("affine")
        if affine_list is not None:
            affine = np.array(affine_list)
        else:
            affine = np.eye(4)
            
        import nibabel as nib
        # Create NIfTI image
        nii_img = nib.Nifti1Image(label.astype(np.int16), affine)
        
        temp_exports_dir = Path("./data/exports")
        temp_exports_dir.mkdir(parents=True, exist_ok=True)
        
        export_file = temp_exports_dir / f"{volume_id}_segmentation.nii.gz"
        nib.save(nii_img, str(export_file))
        
        return FileResponse(
            path=str(export_file),
            filename=f"{volume_id}_segmentation.nii.gz",
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Failed to generate download for volume {volume_id}: {e}", exc_info=True)
        return Response(
            content=json.dumps({"status": "error", "message": f"Export failed: {e}"}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )


# API: Download prediction as DICOM (.zip) series
@app.get("/api/volume/{volume_id}/download/dicom")
async def download_volume_prediction_dicom(volume_id: str):
    try:
        # Load the processed segmentation results
        results_path = Path(f"./data/results/{volume_id}.npz")
        if not results_path.exists():
            return Response(
                content=json.dumps({"status": "error", "message": "Volume results not found."}),
                status_code=status.HTTP_404_NOT_FOUND,
                media_type="application/json"
            )
            
        archive = np.load(results_path, allow_pickle=True)
        label = archive["label"] # Shape (128, 128, 128)
        meta = json.loads(str(archive["metadata"]))
        
        # Prepare export directory
        export_dir = Path("./data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_zip = export_dir / f"{volume_id}_dicom_segmentation.zip"
        
        # Create temp folder for storing dcm slices before zipping
        dcm_temp = Path(f"./data/uploads/export_dcm_{volume_id}")
        if dcm_temp.exists():
            shutil.rmtree(dcm_temp)
        dcm_temp.mkdir(parents=True, exist_ok=True)
        
        backup_zip_path = Path(f"./data/dicom_sources/{volume_id}.zip")
        
        if backup_zip_path.exists():
            logger.info(f"DICOM source zip found. Generating segmentation overlay on original DICOM series...")
            # Extract original DICOM files
            extract_temp = dcm_temp / "original_extracted"
            extract_temp.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(backup_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_temp)
                
            dcm_files = sorted(list(extract_temp.rglob("*.dcm")) + list(extract_temp.rglob("*.DCM")))
            
            # Group by directories or simply sort original files by slice z-position
            slices_data = []
            for d_file in dcm_files:
                try:
                    ds = pydicom.dcmread(str(d_file), force=True)
                    pos = getattr(ds, "ImagePositionPatient", [0, 0, 0])
                    z_coord = pos[2] if len(pos) > 2 else 0.0
                    slice_loc = getattr(ds, "SliceLocation", z_coord)
                    slices_data.append({
                        "path": d_file,
                        "z": slice_loc,
                        "ds": ds
                    })
                except Exception as ex:
                    logger.warning(f"Error reading slice during export: {ex}")
                    
            if not slices_data:
                raise ValueError("Could not parse original DICOM files from backup.")
                
            slices_data = sorted(slices_data, key=lambda x: x["z"])
            num_original_slices = len(slices_data)
            
            # Generate valid UIDs under 64 characters for the segmentation series
            import random
            from pydicom.uid import ImplicitVRLittleEndian
            seg_series_uid = f"1.2.826.0.1.3680043.8.498.99.{random.randint(10**9, 10**10 - 1)}"
            
            # Map segmentation slices to original slices
            for i, item in enumerate(slices_data):
                ds = item["ds"]
                
                # Rescale slice index from original count to 128
                z_label_idx = int(i * 128 / num_original_slices)
                z_label_idx = min(max(z_label_idx, 0), 127)
                
                # Get the 2D label slice (128x128)
                label_slice = label[:, :, z_label_idx]
                
                # Resize 2D label slice to match original columns and rows
                rows = getattr(ds, "Rows", 128)
                cols = getattr(ds, "Columns", 128)
                
                img = Image.fromarray(label_slice.astype(np.uint8), mode="L")
                img_resized = img.resize((cols, rows), resample=Image.NEAREST)
                label_resized = np.array(img_resized)
                
                # Map classes to high-contrast grey levels [0, 80, 160, 240]
                m_label = np.zeros_like(label_resized, dtype=np.uint16)
                m_label[label_resized == 1] = 80
                m_label[label_resized == 2] = 160
                m_label[label_resized == 3] = 240
                
                # Modify DICOM tags with valid UIDs
                ds.SeriesDescription = "3D U-Net Brain Tumor Segmentation"
                ds.SeriesInstanceUID = seg_series_uid
                ds.SOPInstanceUID = f"{seg_series_uid}.99.{i}"
                ds.PixelData = m_label.tobytes()
                ds.Rows, ds.Columns = rows, cols
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                
                if hasattr(ds, "RescaleSlope"):
                    ds.RescaleSlope = 1.0
                if hasattr(ds, "RescaleIntercept"):
                    ds.RescaleIntercept = 0.0
                
                # Set File Meta Header to ensure standard Part 10 format
                from pydicom.dataset import Dataset
                ds.file_meta = Dataset()
                ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                ds.file_meta.MediaStorageSOPClassUID = getattr(ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2")
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                
                ds.is_little_endian = True
                ds.is_implicit_VR = True
                    
                # Save to export folder as standard DICOM Part 10
                export_path = dcm_temp / f"segmentation_slice_{i:03d}.dcm"
                ds.save_as(str(export_path), write_like_original=False)
                
        else:
            logger.info("No DICOM source zip backup found. Creating synthetic DICOM series...")
            # Fallback: create basic synthetic DICOM series
            import datetime
            for i in range(128):
                from pydicom.dataset import FileDataset, Dataset
                from pydicom.uid import ExplicitVRLittleEndian
                
                file_meta = Dataset()
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                file_meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.8.498.{volume_id}.{i}"
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                
                filename_dcm = dcm_temp / f"synthetic_slice_{i:03d}.dcm"
                ds = FileDataset(str(filename_dcm), {}, file_meta=file_meta, local_media_storage_sop_class_uid='1.2.840.10008.5.1.4.1.1.2')
                
                pat_meta = meta.get("patient_metadata", {})
                ds.PatientName = pat_meta.get("patient_name", "Anonymous")
                ds.PatientID = pat_meta.get("patient_id", "N/A")
                ds.StudyDate = pat_meta.get("study_date", datetime.date.today().strftime('%Y%m%d'))
                ds.StudyDescription = pat_meta.get("study_description", "Synthetic DICOM Export")
                ds.SeriesDescription = "3D U-Net Brain Tumor Segmentation (Synthetic)"
                
                ds.StudyInstanceUID = f"1.2.826.0.1.3680043.8.498.1.{volume_id}"
                ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.8.498.2.{volume_id}"
                ds.SOPInstanceUID = f"1.2.826.0.1.3680043.8.498.3.{volume_id}.{i}"
                ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                
                ds.Modality = "MR"
                ds.ImagePositionPatient = [0.0, 0.0, float(i)]
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                ds.SliceThickness = 1.0
                ds.PixelSpacing = [1.0, 1.0]
                
                ds.Rows = 128
                ds.Columns = 128
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                
                label_slice = label[:, :, i]
                m_label = np.zeros_like(label_slice, dtype=np.uint16)
                m_label[label_slice == 1] = 80
                m_label[label_slice == 2] = 160
                m_label[label_slice == 3] = 240
                
                ds.PixelData = m_label.tobytes()
                ds.save_as(str(filename_dcm))
                
        # Zip files
        with zipfile.ZipFile(export_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in dcm_temp.glob("*.dcm"):
                zipf.write(file, arcname=file.name)
                
        # Clean up temp folder
        shutil.rmtree(dcm_temp)
        
        return FileResponse(
            path=str(export_zip),
            filename=f"{volume_id}_dicom_segmentation.zip",
            media_type="application/zip"
        )
        
    except Exception as e:
        logger.error(f"Failed to generate DICOM download for volume {volume_id}: {e}", exc_info=True)
        return Response(
            content=json.dumps({"status": "error", "message": f"DICOM Export failed: {e}"}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )


# API: Get MRI modality slice as PNG image
@app.get("/api/volume/{volume_id}/slice/{slice_idx}/modality/{modality_idx}")
async def get_slice_modality(volume_id: str, slice_idx: int, modality_idx: int, plane: str = "axial"):
    try:
        volume = get_processed_volume(volume_id)
        image = volume["image"] # Shape (4, 128, 128, 128)
        
        # Extract 2D slice based on anatomical plane
        if plane == "sagittal":
            # Slice along 1st dimension (x-axis)
            slice_data = image[modality_idx, slice_idx, :, :]
            # Transpose and flip so Z-axis is vertical, Y-axis is horizontal
            slice_data = np.flip(slice_data.T, axis=0)
        elif plane == "coronal":
            # Slice along 2nd dimension (y-axis)
            slice_data = image[modality_idx, :, slice_idx, :]
            # Transpose and flip so Z-axis is vertical, X-axis is horizontal
            slice_data = np.flip(slice_data.T, axis=0)
        else:
            # Axial (default: slice along 3rd dimension, z-axis)
            slice_data = image[modality_idx, :, :, slice_idx]
        
        # Normalize to 0 - 255
        min_val, max_val = slice_data.min(), slice_data.max()
        if max_val > min_val:
            slice_data = (slice_data - min_val) / (max_val - min_val) * 255.0
        else:
            slice_data = np.zeros_like(slice_data)
            
        img_array = slice_data.astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_array, mode="L")
        
        # Save to memory stream
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except FileNotFoundError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error serving slice modality: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# API: Get color-coded transparency PNG for segmentation label
@app.get("/api/volume/{volume_id}/slice/{slice_idx}/label")
async def get_slice_label(volume_id: str, slice_idx: int, classes: str = "1,2,3", plane: str = "axial"):
    try:
        volume = get_processed_volume(volume_id)
        label = volume["label"] # Shape (128, 128, 128)
        
        # Extract 2D slice based on anatomical plane
        if plane == "sagittal":
            slice_label = label[slice_idx, :, :]
            slice_label = np.flip(slice_label.T, axis=0)
        elif plane == "coronal":
            slice_label = label[:, slice_idx, :]
            slice_label = np.flip(slice_label.T, axis=0)
        else:
            # Axial (default: slice along 3rd dimension, z-axis)
            slice_label = label[:, :, slice_idx]
        
        # Parse classes parameter (e.g. "1,3" -> [1, 3])
        try:
            enabled_classes = [int(c) for c in classes.split(",") if c.strip().isdigit()]
        except Exception:
            enabled_classes = [1, 2, 3]
            
        # Map values to RGBA color mapping
        # 0: background -> transparent (0, 0, 0, 0)
        # 1: edema -> green (0, 255, 0, 150)
        # 2: non-enhancing -> blue (0, 0, 255, 150)
        # 3: enhancing -> red (255, 0, 0, 180)
        rgba = np.zeros((128, 128, 4), dtype=np.uint8)
        if 1 in enabled_classes:
            rgba[slice_label == 1] = [0, 255, 0, 150]
        if 2 in enabled_classes:
            rgba[slice_label == 2] = [0, 0, 255, 150]
        if 3 in enabled_classes:
            rgba[slice_label == 3] = [255, 0, 0, 180]
        
        # Create PIL image
        img = Image.fromarray(rgba, mode="RGBA")
        
        # Save to memory stream
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except FileNotFoundError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error serving slice label: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# API: Get 3D tumor boundary points for Three.js rendering
@app.get("/api/volume/{volume_id}/mesh")
async def get_volume_mesh(volume_id: str):
    try:
        volume = get_processed_volume(volume_id)
        label = volume["label"] # Shape (128, 128, 128)
        
        # Find points where label > 0
        indices = np.argwhere(label > 0)
        if indices.size == 0:
            return {"points": []}
            
        # Generate boundary voxel shell to reduce payload size
        points = []
        padded_label = np.pad(label, 1, mode='constant', constant_values=0)
        
        for x, y, z in indices:
            px, py, pz = x + 1, y + 1, z + 1
            # Check 6-neighborhood for background (0)
            if (padded_label[px-1, py, pz] == 0 or padded_label[px+1, py, pz] == 0 or
                padded_label[px, py-1, pz] == 0 or padded_label[px, py+1, pz] == 0 or
                padded_label[px, py, pz-1] == 0 or padded_label[px, py, pz+1] == 0):
                
                # Normalize coords to [-1, 1] relative to center
                nx = (x - 64.0) / 64.0
                ny = (y - 64.0) / 64.0
                nz = (z - 64.0) / 64.0
                val = int(label[x, y, z])
                points.append([nx, ny, nz, val])
                
        # Limit points to 8000 max by downsampling
        if len(points) > 8000:
            indices_sample = np.random.choice(len(points), 8000, replace=False)
            points = [points[idx] for idx in indices_sample]
            
        return {"points": points}
        
    except FileNotFoundError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error serving 3D mesh points: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def simulate_streaming_task(volume_path: str, delay: float):
    """
    Simulates a hospital scan stream: loads, slices, and processes slice payloads,
    broadcasting progress via WebSockets in real-time.
    """
    from src.producer import load_and_slice_volume
    try:
        packets = load_and_slice_volume(volume_path)
    except Exception as e:
        logger.error(f"Slicing failed for simulation: {e}")
        await manager.broadcast({"type": "error", "message": f"Slicing failed: {e}"})
        return
        
    logger.info(f"Simulating scan stream of {len(packets)} packets...")
    for idx, packet in enumerate(packets):
        await process_slice_payload(packet)
        # Simulate network transfer speed
        await asyncio.sleep(delay)


# API: Trigger a local simulation scan run
@app.post("/api/simulate")
async def trigger_simulation(background_tasks: BackgroundTasks):
    search_dirs = [
        "./data/mock_Task01_BrainTumour/imagesTr",
        "./data/raw/Task01_BrainTumour/imagesTr"
    ]
    volume_path = ""
    for s_dir in search_dirs:
        p = Path(s_dir)
        if p.exists():
            files = list(p.glob("*.nii.gz"))
            if files:
                volume_path = str(files[0])
                break
                
    if not volume_path:
        return {"status": "error", "message": "No NIfTI scans found to simulate."}
        
    # Launch simulation task in background
    background_tasks.add_task(simulate_streaming_task, volume_path, 0.03)
    return {"status": "started", "volume_path": volume_path}


# API: Upload a raw NIfTI scan or a zipped DICOM folder
@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint that accepts NIfTI (.nii.gz / .nii) or DICOM (.zip) uploads.
    Extracts the volumetric files, creates slice packets, and processes them.
    """
    logger.info(f"Received file upload request: {file.filename}")
    
    # Check extension
    filename = file.filename
    if not (filename.endswith(".nii.gz") or filename.endswith(".nii") or filename.endswith(".zip")):
        return Response(
            content=json.dumps({"status": "error", "message": "Unsupported file format. Please upload .nii.gz, .nii, or .zip."}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )
        
    # Enforce file size limit (100MB)
    try:
        await file.seek(0, 2)
        file_size = await file.tell()
        await file.seek(0)
    except Exception as e:
        logger.warning(f"Could not determine upload file size: {e}")
        file_size = 0
        
    if file_size > 100 * 1024 * 1024:
        return Response(
            content=json.dumps({"status": "error", "message": "File exceeds maximum upload size of 100MB."}),
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            media_type="application/json"
        )
        
    # Save the file to ./data/uploads/
    upload_id = str(int(time.time()))
    temp_dir = Path(f"./data/uploads/{upload_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # If the file is a .zip archive, copy it to dicom_sources for future DICOM export
        if filename.endswith(".zip"):
            try:
                backup_dir = Path("./data/dicom_sources")
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(file_path), str(backup_dir / f"{upload_id}.zip"))
                logger.info(f"Backed up source DICOM zip to ./data/dicom_sources/{upload_id}.zip")
            except Exception as e:
                logger.warning(f"Could not back up source DICOM zip: {e}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return Response(
            content=json.dumps({"status": "error", "message": f"Failed to save file: {e}"}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )
        
    # Schedule the parsing and processing in a background task
    background_tasks.add_task(process_uploaded_file, upload_id, str(file_path))
    return {"status": "started", "upload_id": upload_id, "filename": filename}


async def process_uploaded_file(upload_id: str, file_path: str):
    """
    Background worker task to load NIfTI/DICOM zip volumes, slice them, and push internally.
    """
    logger.info(f"Processing uploaded file: {file_path} (Upload ID: {upload_id})")
    temp_dir = Path(file_path).parent
    
    try:
        p = Path(file_path)
        patient_name = "Anonymous"
        patient_id = "N/A"
        study_date = "N/A"
        study_desc = "NIfTI Volumetric Scan"
        
        # Check if zip (DICOM)
        if p.suffix == ".zip":
            # Extract zip
            extract_dir = p.parent / "extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(p, 'r') as zip_ref:
                # Sum up uncompressed sizes to prevent zip bomb
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > 250 * 1024 * 1024:
                    raise ValueError("ZIP archive uncompressed size exceeds safe limit of 250MB.")
                
                for member in zip_ref.infolist():
                    # Safe extraction path validation (prevent Path Traversal)
                    target_path = Path(extract_dir / member.filename).resolve()
                    abs_extract_dir = extract_dir.resolve()
                    if not target_path.is_relative_to(abs_extract_dir):
                        raise ValueError(f"Path traversal attempt detected in zip archive: {member.filename}")
                        
                    # Only extract files, make directories dynamically
                    if not member.is_dir():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                
            # Recursively find all DCM files
            dcm_files = sorted(list(extract_dir.rglob("*.dcm")) + list(extract_dir.rglob("*.DCM")))
            if not dcm_files:
                raise ValueError("No .dcm files found inside the uploaded ZIP folder.")
                
            logger.info(f"Found {len(dcm_files)} DICOM files in uploaded zip. Verifying consistency...")
            
            # Verify clinical integrity: Patient ID consistency check & metadata extraction
            consist_patient_id = None
            for d_file in dcm_files:
                try:
                    ds_h = pydicom.dcmread(str(d_file), stop_before_pixels=True, force=True)
                    curr_pid = getattr(ds_h, "PatientID", "")
                    curr_pname = str(getattr(ds_h, "PatientName", "Anonymous"))
                    curr_date = str(getattr(ds_h, "StudyDate", "N/A"))
                    curr_desc = str(getattr(ds_h, "StudyDescription", "N/A"))
                    
                    if curr_pid:
                        if consist_patient_id is None:
                            consist_patient_id = curr_pid
                            patient_id = curr_pid
                            patient_name = curr_pname
                            study_date = curr_date
                            study_desc = curr_desc
                        elif consist_patient_id != curr_pid:
                            raise ValueError(f"Clinical integrity validation failed: ZIP contains scans from multiple patients ({consist_patient_id} vs {curr_pid}).")
                except ValueError as ve:
                    raise ve
                except Exception as ex:
                    logger.warning(f"Failed to read DICOM header for integrity check on {d_file.name}: {ex}")
            
            # Find directories that contain at least one DICOM file
            dcm_folders = sorted(list(set(f.parent for f in dcm_files)))
            
            modality_volumes = {} # maps channel index -> slices_data
            
            # Helper to map folder name to MRI channels
            def get_channel_idx(folder_path: Path) -> int:
                name = folder_path.name.lower()
                if "t1" in name:
                    if "c" in name or "ce" in name or "contrast" in name:
                        return 1 # T1c / T1ce
                    return 0 # T1
                if "t2" in name:
                    return 2 # T2
                if "flair" in name or "flr" in name:
                    return 3 # FLAIR
                return -1

            if len(dcm_folders) <= 1:
                # Single modality fallback: replicate single channel 4 times
                logger.info("Single DICOM directory detected. Processing single-channel volume fallback...")
                slices_data = []
                for d_file in dcm_files:
                    try:
                        ds = pydicom.dcmread(str(d_file), force=True)
                        pos = getattr(ds, "ImagePositionPatient", [0, 0, 0])
                        z_coord = pos[2] if len(pos) > 2 else 0.0
                        slice_loc = getattr(ds, "SliceLocation", z_coord)
                        pixel_array = ds.pixel_array.astype(np.float32)
                        
                        rescale_slope = getattr(ds, "RescaleSlope", 1.0)
                        rescale_intercept = getattr(ds, "RescaleIntercept", 0.0)
                        pixel_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
                        
                        slices_data.append({
                            "z": slice_loc,
                            "data": pixel_array,
                            "ds": ds
                        })
                    except Exception as ex:
                        logger.warning(f"Failed to read DICOM file {d_file.name}: {ex}")
                        
                if not slices_data:
                    raise ValueError("Could not parse any valid DICOM files from the zip.")
                
                slices_data = sorted(slices_data, key=lambda x: x["z"])
                h, w = slices_data[0]["data"].shape
                volume_3d = np.stack([s["data"] for s in slices_data], axis=2)
                full_volume = np.stack([volume_3d] * 4, axis=3) # Shape (H, W, Z, 4)
                
                first_ds = slices_data[0]["ds"]
                pixel_spacing = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
                slice_thickness = getattr(first_ds, "SliceThickness", 1.0)
                
                dx = float(pixel_spacing[0])
                dy = float(pixel_spacing[1])
                dz = float(slice_thickness)
                affine = np.diag([dx, dy, dz, 1.0])
                
            else:
                # Multi-modality processing
                logger.info(f"Multi-modality directories detected: {[f.name for f in dcm_folders]}. Aligning channels...")
                
                channel_to_folder = {}
                unmapped_folders = []
                for folder in dcm_folders:
                    ch_idx = get_channel_idx(folder)
                    if ch_idx != -1 and ch_idx not in channel_to_folder:
                        channel_to_folder[ch_idx] = folder
                    else:
                        unmapped_folders.append(folder)
                
                # Assign unmapped folders alphabetically to empty slots
                unmapped_folders.sort(key=lambda f: f.name.lower())
                for ch_idx in range(4):
                    if ch_idx not in channel_to_folder and unmapped_folders:
                        channel_to_folder[ch_idx] = unmapped_folders.pop(0)
                        
                # Fill remaining empty slots with the first available folder to prevent failure
                available_folders = [channel_to_folder[c] for c in sorted(channel_to_folder.keys())]
                if not available_folders:
                    available_folders = [dcm_folders[0]]
                    
                for ch_idx in range(4):
                    if ch_idx not in channel_to_folder:
                        channel_to_folder[ch_idx] = available_folders[ch_idx % len(available_folders)]
                        
                # Log warning if missing unique channels
                unique_folders_mapped = len(set(channel_to_folder.values()))
                if unique_folders_mapped < 4:
                    logger.warning(f"Fewer than 4 unique modalities mapped (found {unique_folders_mapped}). Synthesizing missing channels by duplicating available data.")
                
                # Parse slices for each channel
                for ch_idx in range(4):
                    folder_path = channel_to_folder[ch_idx]
                    ch_files = sorted(list(folder_path.glob("*.dcm")) + list(folder_path.glob("*.DCM")))
                    ch_slices = []
                    
                    for f in ch_files:
                        try:
                            ds = pydicom.dcmread(str(f), force=True)
                            pos = getattr(ds, "ImagePositionPatient", [0, 0, 0])
                            z_coord = pos[2] if len(pos) > 2 else 0.0
                            slice_loc = getattr(ds, "SliceLocation", z_coord)
                            pixel_array = ds.pixel_array.astype(np.float32)
                            
                            rescale_slope = getattr(ds, "RescaleSlope", 1.0)
                            rescale_intercept = getattr(ds, "RescaleIntercept", 0.0)
                            pixel_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
                            
                            ch_slices.append({
                                "z": slice_loc,
                                "data": pixel_array,
                                "ds": ds
                            })
                        except Exception as ex:
                            logger.warning(f"Failed to read file {f.name} in channel {ch_idx}: {ex}")
                            
                    if not ch_slices:
                        raise ValueError(f"Could not parse any valid DICOM files from channel subdirectory: {folder_path.name}")
                    
                    ch_slices = sorted(ch_slices, key=lambda x: x["z"])
                    modality_volumes[ch_idx] = ch_slices
                
                # Validate multi-modality slice and grid alignment
                num_slices_list = [len(modality_volumes[c]) for c in range(4)]
                if len(set(num_slices_list)) > 1:
                    raise ValueError(f"Multi-modality alignment failed: mismatching slice counts between channels: {num_slices_list}")
                
                target_h, target_w = modality_volumes[0][0]["data"].shape
                for c in range(4):
                    h, w = modality_volumes[c][0]["data"].shape
                    if h != target_h or w != target_w:
                        raise ValueError(f"Multi-modality alignment failed: mismatching grid size (channel {c} is {h}x{w} vs {target_h}x{target_w})")
                
                # Stack to form (H, W, Z, 4) volume
                stacked_channels = []
                for c in range(4):
                    vol_3d = np.stack([s["data"] for s in modality_volumes[c]], axis=2)
                    stacked_channels.append(vol_3d)
                full_volume = np.stack(stacked_channels, axis=3)
                
                first_ds = modality_volumes[0][0]["ds"]
                pixel_spacing = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
                slice_thickness = getattr(first_ds, "SliceThickness", 1.0)
                
                dx = float(pixel_spacing[0])
                dy = float(pixel_spacing[1])
                dz = float(slice_thickness)
                affine = np.diag([dx, dy, dz, 1.0])
                
        else:
            # NIfTI file (load via nibabel)
            import nibabel as nib
            img = nib.load(str(p))
            img_data = img.get_fdata()
            affine = img.affine
            
            patient_name = "Anonymous (NIfTI)"
            patient_id = "N/A"
            study_date = "N/A"
            study_desc = f"NIfTI Scan: {p.name}"
            
            # If img_data is 3D, stack to 4 channels
            if img_data.ndim == 3:
                full_volume = np.stack([img_data] * 4, axis=3)
            elif img_data.ndim == 4:
                full_volume = img_data
            else:
                raise ValueError(f"Unsupported image dimension {img_data.ndim}")
                
        # Generate packets and stream internally to process_slice_payload
        # Slice full_volume along the third dimension (Z)
        num_slices = full_volume.shape[2]
        logger.info(f"Successfully loaded volume for upload {upload_id}. Stack size: {full_volume.shape}. Slicing...")
        
        # Build patient metadata dictionary
        pat_meta_dict = {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "study_date": study_date,
            "study_description": study_desc
        }
        
        # We process slices sequentially and feed them locally
        for z_idx in range(num_slices):
            slice_data = full_volume[:, :, z_idx, :]
            slice_data_f32 = slice_data.astype(np.float32)
            slice_bytes = slice_data_f32.tobytes()
            b64_data = base64.b64encode(slice_bytes).decode("utf-8")
            
            packet = {
                "volume_id": upload_id,
                "packet_index": z_idx,
                "total_packets": num_slices,
                "slice_shape": list(slice_data_f32.shape),
                "dtype": "float32",
                "data": b64_data,
                "timestamp": time.time(),
                "affine": affine.tolist(),
                "patient_metadata": pat_meta_dict
            }
            await process_slice_payload(packet)
            # Yield control briefly to keep API responsive
            await asyncio.sleep(0.001)
            
        logger.info(f"Completed streaming slices for upload {upload_id}.")
        
    except Exception as e:
        logger.error(f"Error processing uploaded volume {upload_id}: {e}", exc_info=True)
        await manager.broadcast({"type": "error", "volume_id": upload_id, "message": str(e)})
    finally:
        # Cleanup temporary files safely (only if in data/uploads)
        try:
            if "data/uploads" in str(temp_dir).replace("\\", "/"):
                shutil.rmtree(str(temp_dir))
                logger.info(f"Cleaned up temporary directories for upload {upload_id}.")
            else:
                logger.info(f"Bypassed cleanup of non-temporary directory {temp_dir}.")
        except Exception as e:
            logger.warning(f"Could not clean up temporary upload directory {temp_dir}: {e}")


async def process_slice_payload(payload: Dict[str, Any]):
    """
    Receives slice payloads, stores them in memory, and triggers 3D inference when complete.
    Also broadcasts real-time updates via WebSockets.
    """
    global volume_cache, volume_metadata
    
    volume_id = payload["volume_id"]
    packet_idx = payload["packet_index"]
    total_packets = payload["total_packets"]
    slice_shape = payload["slice_shape"]
    data_b64 = payload["data"]
    
    # Broadcast progress
    await manager.broadcast({
        "type": "slice_received",
        "volume_id": volume_id,
        "packet_index": packet_idx,
        "total_packets": total_packets,
        "slice_shape": slice_shape,
        "payload_size_kb": len(data_b64) / 1024.0,
    })
    
    # Decode slice array
    raw_bytes = base64.b64decode(data_b64)
    slice_array = np.frombuffer(raw_bytes, dtype=np.float32).reshape(slice_shape)
    
    if volume_id not in volume_cache:
        volume_cache[volume_id] = [None] * total_packets
        volume_metadata[volume_id] = {
            "start_time": time.time(),
            "slice_shape": slice_shape,
            "received_count": 0,
            "affine": payload.get("affine"),
            "patient_metadata": payload.get("patient_metadata"),
        }
        
    # Guard against out-of-bounds packet_idx and duplicate packets
    if 0 <= packet_idx < total_packets:
        if volume_cache[volume_id][packet_idx] is None:
            volume_cache[volume_id][packet_idx] = slice_array
            volume_metadata[volume_id]["received_count"] += 1
            
    # If complete, run inference
    if volume_metadata[volume_id]["received_count"] == total_packets:
        await manager.broadcast({"type": "reassembly_started", "volume_id": volume_id})
        # Execute reassembly and inference asynchronously to avoid blocking the network stream
        asyncio.create_task(reassemble_and_infer(volume_id, total_packets))


async def reassemble_and_infer(volume_id: str, total_packets: int):
    """
    Reassembles slice stack into a single 3D/4D PyTorch tensor, resizes it,
    runs JIT inference, saves compressed output array to disk, and broadcasts details.
    """
    global volume_cache, volume_metadata, model
    
    metadata = volume_metadata.get(volume_id, {"start_time": time.time()})
    start_time = metadata["start_time"]
    
    await manager.broadcast({"type": "inference_started", "volume_id": volume_id})
    
    # Retrieve slices in sequential order
    slices = volume_cache[volume_id]
    
    # Stack slices along z-axis (axial plane dimension 2)
    # Stacked shape (H, W, Z, C) -> (240, 240, Z, 4)
    full_array = np.stack(slices, axis=2)
    
    # Get affine matrix from metadata
    affine_list = metadata.get("affine")
    if affine_list is not None:
        affine = np.array(affine_list)
    else:
        affine = np.eye(4)
        
    # Transpose to shape (C, H, W, Z) for MONAI channel-first orientation/spacing transforms
    data_array = np.transpose(full_array, (3, 0, 1, 2))
    
    # Wrap in MetaTensor to carry the spatial affine matrix consistently
    from monai.data import MetaTensor
    meta_tensor = MetaTensor(data_array, affine=affine)
    
    # Standardize orientation to RAS and spacing to 1.0mm isotropic resolution
    try:
        orient = mt.Orientation(axcodes="RAS")
        spacing = mt.Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")
        meta_tensor = orient(meta_tensor)
        meta_tensor = spacing(meta_tensor)
    except Exception as e:
        logger.warning(f"Could not apply orientation/spacing standardisation: {e}")
        
    # Apply intensity normalization (channel-wise, non-zero voxels)
    try:
        normalizer = mt.NormalizeIntensity(nonzero=True, channel_wise=True)
        meta_tensor = normalizer(meta_tensor)
    except Exception as e:
        logger.warning(f"Could not apply intensity normalization via MONAI: {e}")
        try:
            logger.info("Applying fallback z-score normalization on non-zero voxels...")
            for c in range(meta_tensor.shape[0]):
                ch_data = meta_tensor[c]
                nonzero_mask = ch_data != 0
                if nonzero_mask.any():
                    mean = ch_data[nonzero_mask].mean()
                    std = ch_data[nonzero_mask].std()
                    if std > 1e-5:
                        meta_tensor[c] = torch.where(nonzero_mask, (ch_data - mean) / std, ch_data)
                    else:
                        meta_tensor[c] = torch.where(nonzero_mask, ch_data - mean, ch_data)
                else:
                    ch_min, ch_max = ch_data.min(), ch_data.max()
                    if ch_max > ch_min:
                        meta_tensor[c] = (ch_data - ch_min) / (ch_max - ch_min)
        except Exception as ex:
            logger.warning(f"Fallback intensity normalization failed: {ex}")
        
    # Convert to pure torch tensor shape (1, C, H, W, Z)
    if hasattr(meta_tensor, "as_tensor"):
        pure_tensor = meta_tensor.as_tensor()
    else:
        pure_tensor = torch.as_tensor(meta_tensor)
        
    tensor = pure_tensor.float().unsqueeze(0)
    tensor = tensor.to(device)
    
    # Perform resizing/interpolation using MONAI Resize to standard input shape (1, 4, 128, 128, 128)
    resize_start = time.time()
    try:
        resize = mt.Resize(spatial_size=(128, 128, 128), mode="trilinear")
        with torch.no_grad():
            input_tensor = resize(tensor.squeeze(0)).unsqueeze(0)
    except Exception as e:
        logger.error(f"Resize failed: {e}. Falling back to default F.interpolate.")
        with torch.no_grad():
            input_tensor = F.interpolate(
                tensor,
                size=(128, 128, 128),
                mode="trilinear",
                align_corners=False
            )
    resize_time = time.time() - resize_start
    
    # Run TorchScript engine inference
    infer_start = time.time()
    if model is not None:
        with torch.no_grad():
            outputs = model(input_tensor)
        inference_time = time.time() - infer_start
        device_used = str(outputs.device)
        
        # Post-process: extract class predictions (argmax along channel dimension)
        # Shape: (128, 128, 128)
        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.int8)
    else:
        # Dry-run fallback if model is not loaded (create mock outputs)
        await asyncio.sleep(0.5)
        inference_time = 0.500
        device_used = "cpu (simulated)"
        pred = np.zeros((128, 128, 128), dtype=np.int8)
        # Add a mock tumor sphere for visualization
        x, y, z = np.ogrid[:128, :128, :128]
        dist_sq = (x - 64)**2 + (y - 64)**2 + (z - 64)**2
        pred[dist_sq <= 25**2] = 1
        pred[dist_sq <= 15**2] = 2
        pred[dist_sq <= 6**2] = 3
        
    total_latency = time.time() - start_time
    
    # Calculate volumetric statistics
    edema_voxels = int((pred == 1).sum())
    non_enhancing_voxels = int((pred == 2).sum())
    enhancing_voxels = int((pred == 3).sum())
    total_tumor_voxels = edema_voxels + non_enhancing_voxels + enhancing_voxels
    
    # Obtain original pixel dimensions & spacing
    dx, dy, dz = np.abs(np.diag(affine)[:3])
    orig_h, orig_w, orig_z = full_array.shape[:3]
    orig_volume_mm3 = (orig_h * orig_w * orig_z) * (dx * dy * dz)
    pred_voxel_volume_mm3 = orig_volume_mm3 / (128.0 ** 3)
    
    edema_vol_cc = (edema_voxels * pred_voxel_volume_mm3) / 1000.0
    non_enhancing_vol_cc = (non_enhancing_voxels * pred_voxel_volume_mm3) / 1000.0
    enhancing_vol_cc = (enhancing_voxels * pred_voxel_volume_mm3) / 1000.0
    total_vol_cc = edema_vol_cc + non_enhancing_vol_cc + enhancing_vol_cc
    
    volumetric_stats = {
        "edema_voxels": edema_voxels,
        "edema_volume_cc": edema_vol_cc,
        "non_enhancing_voxels": non_enhancing_voxels,
        "non_enhancing_volume_cc": non_enhancing_vol_cc,
        "enhancing_voxels": enhancing_voxels,
        "enhancing_volume_cc": enhancing_vol_cc,
        "total_tumor_voxels": total_tumor_voxels,
        "total_tumor_volume_cc": total_vol_cc
    }
    
    # Extract resized image values for caching and slice visualization
    image_np = input_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    
    patient_metadata = metadata.get("patient_metadata")
    if patient_metadata is None:
        patient_metadata = {
            "patient_name": "Smith^John (Simulated)",
            "patient_id": "MR-55421",
            "study_date": "2026-06-25",
            "study_description": "Simulated Hospital Scan Stream"
        }
 
    # Save results to disk as compressed numpy archive
    results_path = f"./data/results/{volume_id}.npz"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_path,
        image=image_np,
        label=pred,
        metadata=json.dumps({
            "volume_id": volume_id,
            "total_packets": total_packets,
            "total_latency": total_latency,
            "inference_time": inference_time,
            "resize_time": resize_time,
            "timestamp": time.time(),
            "device": device_used,
            "patient_metadata": patient_metadata,
            "volumetric_stats": volumetric_stats
        })
    )
    
    logger.info(f"3D VOLUMETRIC INFERENCE SUCCESSFUL. Output saved to {results_path}")
    
    # Broadcast results
    await manager.broadcast({
        "type": "inference_completed",
        "volume_id": volume_id,
        "total_latency": total_latency,
        "inference_time": inference_time,
        "resize_time": resize_time,
        "device": device_used,
    })
    
    # Cleanup memory cache
    if volume_id in volume_cache:
        del volume_cache[volume_id]
    if volume_id in volume_metadata:
        del volume_metadata[volume_id]


async def consume_kafka_loop(bootstrap_servers: str, topic: str):
    """
    Main loop consuming slice payloads from the Kafka topic. Retries connection dynamically if offline.
    """
    global kafka_connected
    while True:
        logger.info(f"Initializing Kafka Consumer: consuming from topic '{topic}'...")
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id="mri-inference-workers",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_partition_fetch_bytes=5242880,
        )
        
        try:
            await consumer.start()
            logger.info("Kafka Consumer started successfully!")
            kafka_connected = True
            
            async for message in consumer:
                payload = message.value
                await process_slice_payload(payload)
                
        except KafkaConnectionError as e:
            logger.warning(f"Could not connect to Kafka at {bootstrap_servers}: {e}")
            kafka_connected = False
        except Exception as e:
            logger.error(f"Error in Kafka consumer loop: {e}")
            kafka_connected = False
        finally:
            logger.info("Stopping/Cleaning Kafka Consumer...")
            try:
                await consumer.stop()
            except Exception:
                pass
                
        logger.info("Kafka consumer offline. Retrying connection in 10 seconds...")
        await asyncio.sleep(10)


# Mount static files folder pointing to the frontend directory
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    # Route root endpoint to serve UI index.html
    @app.get("/")
    async def read_index():
        return FileResponse(str(frontend_dir / "index.html"))
else:
    logger.warning("Frontend directory not found at any fallback location. Static files and root UI endpoint are disabled.")


async def run_simulation_harness():
    """
    Command line simulation runner.
    """
    global model
    logger.info("Running offline simulation harness...")
    # Load model locally
    model_path = "./deploy/model_trace.pt"
    if Path(model_path).exists():
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        
    from src.producer import load_and_slice_volume
    search_dirs = ["./data/mock_Task01_BrainTumour/imagesTr", "./data/raw/Task01_BrainTumour/imagesTr"]
    volume_path = ""
    for s_dir in search_dirs:
        p = Path(s_dir)
        if p.exists():
            files = list(p.glob("*.nii.gz"))
            if files:
                volume_path = str(files[0])
                break
    if volume_path:
        packets = load_and_slice_volume(volume_path)
        for packet in packets:
            await process_slice_payload(packet)
            await asyncio.sleep(0.005)


if __name__ == "__main__":
    if "--simulate" in sys.argv:
        asyncio.run(run_simulation_harness())
    else:
        import uvicorn
        uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
