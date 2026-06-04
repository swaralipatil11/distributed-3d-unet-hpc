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
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("worker_engine")

# FastAPI App
app = FastAPI(title="3D MRI Volumetric Inference Worker", version="1.0.0")

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.jit.ScriptModule = None
volume_cache: Dict[str, Dict[int, np.ndarray]] = {}
volume_metadata: Dict[str, Dict[str, Any]] = {}


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
        
    # Start the Kafka consumer background task
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "mri-inference-requests")
    asyncio.create_task(consume_kafka_loop(bootstrap_servers, topic))


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
    return {
        "status": "healthy" if model is not None else "degraded",
        "device": str(device),
        "active_streams": len(volume_cache)
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


# API: Get MRI modality slice as PNG image
@app.get("/api/volume/{volume_id}/slice/{slice_idx}/modality/{modality_idx}")
async def get_slice_modality(volume_id: str, slice_idx: int, modality_idx: int):
    results_path = Path(f"./data/results/{volume_id}.npz")
    if not results_path.exists():
        return Response(status_code=status.HTTP_404_NOT_FOUND)
        
    try:
        archive = np.load(results_path)
        image = archive["image"] # Shape (4, 128, 128, 128)
        
        # Extract 2D slice
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
    except Exception as e:
        logger.error(f"Error serving slice modality: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# API: Get color-coded transparency PNG for segmentation label
@app.get("/api/volume/{volume_id}/slice/{slice_idx}/label")
async def get_slice_label(volume_id: str, slice_idx: int):
    results_path = Path(f"./data/results/{volume_id}.npz")
    if not results_path.exists():
        return Response(status_code=status.HTTP_404_NOT_FOUND)
        
    try:
        archive = np.load(results_path)
        label = archive["label"] # Shape (128, 128, 128)
        
        # Extract 2D slice
        slice_label = label[:, :, slice_idx]
        
        # Map values to RGBA color mapping
        # 0: background -> transparent (0, 0, 0, 0)
        # 1: edema -> green (0, 255, 0, 150)
        # 2: non-enhancing -> blue (0, 0, 255, 150)
        # 3: enhancing -> red (255, 0, 0, 180)
        rgba = np.zeros((128, 128, 4), dtype=np.uint8)
        rgba[slice_label == 1] = [0, 255, 0, 150]
        rgba[slice_label == 2] = [0, 0, 255, 150]
        rgba[slice_label == 3] = [255, 0, 0, 180]
        
        # Create PIL image
        img = Image.fromarray(rgba, mode="RGBA")
        
        # Save to memory stream
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error serving slice label: {e}")
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
        volume_cache[volume_id] = {}
        volume_metadata[volume_id] = {
            "start_time": time.time(),
            "slice_shape": slice_shape,
        }
        
    volume_cache[volume_id][packet_idx] = slice_array
    
    # If complete, run inference
    if len(volume_cache[volume_id]) == total_packets:
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
    slices = [volume_cache[volume_id][i] for i in range(total_packets)]
    
    # Stack slices along z-axis (axial plane dimension 2)
    # Stacked shape (H, W, Z, C) -> (240, 240, Z, 4)
    full_array = np.stack(slices, axis=2)
    
    # Permute to PyTorch shape (C, H, W, Z) -> (4, 240, 240, Z)
    # Add batch dimension -> (1, 4, 240, 240, Z)
    tensor = torch.from_numpy(full_array).float()
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    
    # Move to execution device
    tensor = tensor.to(device)
    
    # Perform trilinear interpolation to resize to target input shape (1, 4, 128, 128, 128)
    resize_start = time.time()
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
    
    # Extract resized image values for caching and slice visualization
    image_np = input_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Save results to disk as compressed numpy archive
    results_path = f"./data/results/{volume_id}.npz"
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
            "device": device_used
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
    Main loop consuming slice payloads from the Kafka topic.
    """
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
    except KafkaConnectionError as e:
        logger.warning(f"Could not connect to Kafka at {bootstrap_servers}: {e}")
        logger.info("FastAPI backend is offline from Kafka. Use REST simulation API trigger.")
        return
    except Exception as e:
        logger.error(f"Failed to start Kafka Consumer: {e}")
        return
        
    try:
        async for message in consumer:
            payload = message.value
            await process_slice_payload(payload)
    except Exception as e:
        logger.error(f"Error in Kafka consumer: {e}")
    finally:
        logger.info("Stopping Kafka Consumer...")
        await consumer.stop()


# Mount static files folder pointing to the frontend directory
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Route root endpoint to serve UI index.html
@app.get("/")
async def read_index():
    return FileResponse(str(frontend_dir / "index.html"))


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
