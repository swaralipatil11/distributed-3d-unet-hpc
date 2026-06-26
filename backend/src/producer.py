#!/usr/bin/env python3
"""
Asynchronous Python publisher simulating a hospital scanning server.
Slices 3D volumetric MRI arrays along the z-axis and streams packetized payloads
asynchronously to Apache Kafka using the aiokafka library.
"""

import os
import sys
import json
import time
import base64
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import nibabel as nib
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kafka_producer")


def load_and_slice_volume(volume_path: str) -> List[Dict[str, Any]]:
    """
    Loads a 3D/4D NIfTI volume and slices it along the z-axis (depth dimension).
    Serializes each slice to base64 encoded bytes and appends sequencing metadata.
    
    Args:
        volume_path: Path to the .nii.gz file.
        
    Returns:
        List[Dict[str, Any]]: List of packet dictionaries ready to stream.
    """
    logger.info(f"Loading volumetric volume from {volume_path}...")
    try:
        img = nib.load(volume_path)
    except Exception as e:
        logger.error(f"Failed to read NIfTI file: {e}")
        raise
        
    img_data = img.get_fdata()
    logger.info(f"Loaded volume shape: {img_data.shape}, datatype: {img_data.dtype}")
    
    # BraTS shape is usually (240, 240, 155, 4) where 155 is the depth (z) dimension
    # Let's extract metadata
    volume_id = Path(volume_path).name.split(".")[0]
    
    # If 4D volume, third dimension is depth (z)
    if img_data.ndim == 4:
        num_slices = img_data.shape[2]
    elif img_data.ndim == 3:
        num_slices = img_data.shape[2]
    else:
        raise ValueError(f"Unsupported image dimensions: {img_data.ndim}")
        
    packets = []
    logger.info(f"Slicing volume into {num_slices} sub-payloads along z-axis...")
    
    for z in range(num_slices):
        # Extract 2D slice with all modality channels
        if img_data.ndim == 4:
            slice_data = img_data[:, :, z, :]
        else:
            slice_data = img_data[:, :, z]
            
        # Standardize datatype to float32
        slice_data_f32 = slice_data.astype(np.float32)
        
        # Serialize raw numpy array data to bytes
        slice_bytes = slice_data_f32.tobytes()
        
        # Base64 encode the byte array to embed inside JSON payload
        b64_data = base64.b64encode(slice_bytes).decode("utf-8")
        
        packet = {
            "volume_id": volume_id,
            "packet_index": z,
            "total_packets": num_slices,
            "slice_shape": list(slice_data_f32.shape),
            "dtype": "float32",
            "data": b64_data,
            "timestamp": time.time(),
            "affine": img.affine.tolist(),
        }
        packets.append(packet)
        
    logger.info("Slicing and serialization completed successfully.")
    return packets


async def run_simulation(packets: List[Dict[str, Any]], topic: str, delay: float):
    """
    Offline fallback simulation when Kafka is unreachable.
    Prints simulated packet transmissions to standard output.
    """
    logger.info("=" * 60)
    logger.info("RUNNING OFFLINE FALLBACK SIMULATION (KAFKA UNREACHABLE)")
    logger.info(f"Target streaming topic: {topic}")
    logger.info("=" * 60)
    
    start_time = time.time()
    for idx, packet in enumerate(packets):
        # Simulate send delay
        await asyncio.sleep(delay)
        
        # Extract small log information (do not print full base64 data to keep logs clean)
        payload_size_kb = len(packet["data"]) / 1024.0
        logger.info(
            f"SIMULATED: Sent Packet [{packet['packet_index'] + 1}/{packet['total_packets']}] | "
            f"Volume: {packet['volume_id']} | "
            f"Shape: {packet['slice_shape']} | "
            f"Payload: {payload_size_kb:.1f} KB"
        )
        
    elapsed = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"SIMULATION SUCCESSFUL: Streamed {len(packets)} slices in {elapsed:.2f} seconds.")
    logger.info("=" * 60)


async def main_async():
    parser = argparse.ArgumentParser(description="Async MRI Kafka Publisher.")
    parser.add_argument(
        "--volume-path",
        type=str,
        default="",
        help="Path to NIfTI volume. Defaults to first volume found in data directory.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap server address.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="mri-inference-requests",
        help="Kafka topic to publish messages to.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="Interval delay (seconds) between sequential packets.",
    )
    parser.add_argument(
        "--simulate-fallback",
        action="store_true",
        help="Force run offline fallback simulation.",
    )
    args = parser.parse_args()
    
    # Resolve volume path if empty
    volume_path = args.volume_path
    if not volume_path:
        # Search mock directory first, then raw directory
        search_dirs = [
            "./data/mock_Task01_BrainTumour/imagesTr",
            "./data/raw/Task01_BrainTumour/imagesTr"
        ]
        found = False
        for s_dir in search_dirs:
            p = Path(s_dir)
            if p.exists():
                files = list(p.glob("*.nii.gz"))
                if files:
                    volume_path = str(files[0])
                    found = True
                    break
        if not found:
            logger.error("No .nii.gz volume files found in mock or raw directories.")
            sys.exit(1)
            
    # Load and slice volume
    try:
        packets = load_and_slice_volume(volume_path)
    except Exception as e:
        logger.error(f"Slicing pipeline failed: {e}")
        sys.exit(1)
        
    if args.simulate_fallback:
        await run_simulation(packets, args.topic, args.delay)
        return
        
    logger.info(f"Connecting to Kafka broker at: {args.bootstrap_servers}")
    producer = AIOKafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # Increase maximum request size to handle large slice payloads safely (e.g. 5MB limit)
        max_request_size=5242880,
    )
    
    try:
        await producer.start()
        logger.info("Kafka Connection established successfully!")
    except KafkaConnectionError as e:
        logger.warning(f"Could not connect to Kafka broker at {args.bootstrap_servers}: {e}")
        logger.info("Falling back to local offline simulation...")
        await run_simulation(packets, args.topic, args.delay)
        return
    except Exception as e:
        logger.error(f"Unexpected connection error: {e}")
        logger.info("Falling back to local offline simulation...")
        await run_simulation(packets, args.topic, args.delay)
        return
        
    # Stream packets asynchronously
    logger.info("=" * 60)
    logger.info(f"STARTING ASYNCHRONOUS STREAM TO TOPIC: {args.topic}")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        for idx, packet in enumerate(packets):
            # Send message asynchronously and wait for acknowledgment
            await producer.send_and_wait(args.topic, packet)
            
            # Send log info
            payload_size_kb = len(packet["data"]) / 1024.0
            logger.info(
                f"PUBLISHED: Packet [{packet['packet_index'] + 1}/{packet['total_packets']}] | "
                f"Offset ACK'd | Payload size: {payload_size_kb:.1f} KB"
            )
            
            # Rate-limiting delay
            await asyncio.sleep(args.delay)
            
        elapsed = time.time() - start_time
        logger.info("-" * 60)
        logger.info(f"STREAM SUCCESSFUL: Published {len(packets)} packets in {elapsed:.2f} seconds.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        logger.info("Stopping Kafka producer...")
        await producer.stop()
        logger.info("Producer stopped.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
