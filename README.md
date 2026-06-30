# Distributed High-Performance Inference Pipeline for Volumetric 3D Medical Segmentation

This repository implements a production-grade, distributed volumetric inference pipeline designed to ingest, stream, process, and visualize multi-modality 3D brain MRI scans (FLAIR, T1w, t1gd, T2w) for brain tumor segmentation.

The architecture combines **MONAI** medical image processing, **PyTorch 3D Convolutional Neural Networks**, **Apache Kafka** distributed message streams, **FastAPI** web handlers, **WebSockets** for real-time telemetry, and **Three.js WebGL** graphics for volumetric visual analysis.

---

## 🏗️ Architecture Overview

The system operates as a distributed microservices pipeline structured for high data velocities, low-latency execution, and clinical PACS/DICOM interoperability.

```mermaid
graph TD
    subgraph Hospital Scanning Server (Ingestion)
        NIfTI[Raw .nii.gz / DICOM Zip] -->|nibabel / zip load| Slicer[z-Axis Axial Slicer]
        Slicer -->|numpy tobytes| B64[base64 Encoder]
        B64 -->|packet metadata| Prod[Async Kafka Publisher]
    end

    subgraph Messaging Broker (Kafka Stack)
        Prod -->|stream slices| Topic[(Kafka: mri-inference-requests)]
    end

    subgraph Distributed Inference Worker (FastAPI & DL)
        Topic -->|consume| Cons[Async Kafka Consumer]
        Cons -->|decode & cache| Cache[(In-Memory Slice Cache)]
        Cache -->|reassemble| Stack[3D Array Reassembler]
        Stack -->|manual fallback norm| Interp[PyTorch Trilinear Interpolation]
        Interp -->|resize 1,4,128,128,128| JIT[TorchScript UNet3D Engine]
        JIT -->|forward pass| Post[Argmax & Volumetric Volume Math]
        Post -->|save results .npz| Disk[(data/results/)]
        Post -->|real-time telemetry| WS[FastAPI WebSocket Server]
    end

    subgraph Visualization & Analytics (Presentation Layer)
        WS -->|telemetry progress| UI[Glassmorphic Web UI Dashboard]
        UI -->|GET slice + plane query| Disk
        UI -->|GET 3D mesh points| Disk
        Disk -->|MPR Slicing Engine| UI
        Disk -->|Three.js Particle Mesh| UI
        Disk -->|DICOM Tag Overwriter| DICOMZip[Clinical DICOM Zip Export]
    end
```

### Data Pipeline Stages:
1. **Volumetric Slicing**: The hospital scanning server (Producer) reads 3D/4D NIfTI or DICOM volumes, slices them along the z-axis (axial plane), base64 encodes the binary byte arrays, wraps them in sequential packet metadata, and streams them asynchronously to Kafka.
2. **Concurrently Streamed Messaging**: Apache Kafka serves as the high-throughput message ingestion broker, isolating the scanners from the inference workers and guaranteeing packet delivery.
3. **Volumetric Reassembly & Fallback Normalization**: FastAPI workers consume slices asynchronously, caching them in memory per `volume_id`. When all slices are collected, they are stacked back to reconstruct the raw 3D modality grid. If MONAI's default `NormalizeIntensity` fails, manual channel-wise z-score and min-max normalization over non-zero voxels are applied as a fallback.
4. **Isotropic Voxel Resizing**: The reassembled tensor is permuted to channel-first format `(1, 4, H, W, Z)` and resized using PyTorch's native `F.interpolate` trilinear interpolation to the exact shapes expected by the neural network: `(1, 4, 128, 128, 128)`.
5. **JIT Compilation Inference**: The input tensor is fed to the compiled **TorchScript 3D U-Net** engine. The logits output is processed via `argmax` along the channel dimension to generate the final segmentation label map.
6. **Multi-Planar Reconstruction (MPR)**: The backend transposes and flips the 3D results array to serve 2D slices along standard clinical planes (**Axial**, **Sagittal**, and **Coronal** views) maintaining correct anatomical orientations.
7. **Interactive 3D WebGL Visualization**: Evaluates a 3D neighborhood comparison mask to extract a hollow boundary voxel shell of the tumor, normalizes coordinates to `[-1, 1]` for center-orbital scaling, and renders it in the web dashboard using a **Three.js particle system** that updates reactively to toggle checkboxes.
8. **PACS-Compatible DICOM Export**: If original DICOM source backups exist, nearest-neighbor resizing scales segmentation masks to match original dimensions (avoiding class label interpolation), maps intensities to grey values (`[0, 80, 160, 240]`), assigns valid UIDs ($<64$ characters), writes standard Part 10 formats, and zips them for PACS download.

---

## 🛠️ System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS.
- **Python**: version `3.10` or `3.11` (Python `3.13` is supported, but `3.11` is recommended for optimized CUDA wheel compatibility).
- **Docker**: Docker Desktop (or standalone docker-compose) to spin up the local Kafka broker.
- **Hardware (Optional)**: NVIDIA GPU with CUDA support for high-throughput training/inference acceleration.

---

## 🚀 Setup & Execution Guide

> [!TIP]
> **Windows Quick-Start:** You can automatically verify dependencies, generate mock files, trace the JIT model, and launch the server by running a single command in the root folder: `.\start.ps1`

### 1. Installation
Clone the repository and install the dependencies from the `backend/` directory:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Download and Validate the Dataset
Fetch and extract the Medical Segmentation Decathlon (MSD) Task01_BrainTumour dataset (7.09 GB):
```bash
python src/download_dataset.py --remove-archive
```

### 3. Launch the Local Kafka Broker Stack
Use Docker Compose to launch a single-node Apache Kafka broker and Zookeeper mapping ports `9092` and `2181`:
```bash
docker-compose up -d
```

### 4. Compile the 3D U-Net to TorchScript
Compile the PyTorch neural network to a language-agnostic production binary:
```bash
python -m src.export
```
This reads the local checkpoint `backend/deploy/unet_model.pt` (created during training) and JIT-compiles it, saving the serialized trace to `backend/deploy/model_trace.pt`.

### 5. Start the FastAPI Worker Engine
Run the worker server from the `backend/` directory:
```bash
python -m src.main
```
This loads the TorchScript engine and opens the web application at `http://localhost:8000`.

### 6. Stream Scan Simulation
To trigger scan streaming over Kafka:
- Run the command-line publisher:
  ```bash
  python -m src.producer
  ```
- Or click the **⚡ Simulate Hospital Scan Stream** button directly in the web UI dashboard at `http://localhost:8000`.

---

## ⚡ Production Optimizations

- **Automatic Mixed Precision (AMP)**: The training runner (`backend/src/train.py`) and inference worker (`backend/src/main.py`) utilize PyTorch's native `autocast` to execute training and model forward passes in FP16 on GPU. This reduces the GPU memory footprint by up to 50% and optimizes throughput with negligible loss in accuracy.
- **C++ JIT Compilation**: The model architecture is written to be 100% scriptable. By compiling the model via `torch.jit.script` to `model_trace.pt`, we eliminate all Python interpreter runtime overhead. This JIT binary can be loaded directly inside a C++ daemon (`torch::jit::load`) to serve high-throughput, low-latency concurrent requests.
- **Volumetric Metric Mathematics**: Uses original spatial affine diagonals to calculate precise physical voxel volumes, computing exact sub-region tumor volume in cubic centimeters ($cc$) for quantitative diagnostics.
- **Vectorized Boundary Shell Mesh Generation**: Prevents UI lag in WebGL canvases by identifying boundary voxels (active voxels containing background neighbors in a 6-neighborhood) using a high-performance vectorized NumPy slicing algorithm, shrinking scene geometry from 100,000+ voxels to under 8,000 points dynamically in milliseconds.
- **Dynamic Reconnection Loops**: Implemented a retry loop for Kafka client managers that attempts connections every 10 seconds rather than failing on network glitches, reporting connectivity telemetry to `/healthz`.
- **Kafka-Lag-Driven Auto-Scaling**: The system supports Horizontal Pod Autoscaling via standard Kubernetes external metrics (`backend/deploy/hpa.yaml`) as well as a native event-driven KEDA configuration (`backend/deploy/keda-scaledobject.yaml`). This monitors consumer group lag on the `mri-inference-requests` topic to dynamically scale inference worker replicas from 2 up to 10 pods.

