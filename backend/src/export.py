#!/usr/bin/env python3
"""
TorchScript compilation script for the 3D U-Net model.
Compiles the model to a language-agnostic production binary at deploy/model_trace.pt.
"""

import os
import argparse
import sys
from pathlib import Path

import torch
from src.model import UNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 3D U-Net to TorchScript binary.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./deploy/unet_model.pt",
        help="Path to the training checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./deploy/model_trace.pt",
        help="Target output path for TorchScript model.",
    )
    parser.add_argument(
        "--init-features",
        type=int,
        default=16,
        help="Number of initial filters in U-Net (must match training).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    
    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TORCHSCRIPT EXPORT PIPELINE")
    print(f"Loading model checkpoint from: {checkpoint_path}")
    print(f"Target production output path: {output_path}")
    print("=" * 60)
    
    # 1. Instantiate the model
    print("Instantiating UNet3D model...")
    model = UNet3D(in_channels=4, out_channels=4, init_features=args.init_features)
    model.eval()
    
    # 2. Load model weights if checkpoint exists
    if checkpoint_path.exists():
        print(f"Found checkpoint file. Loading state dictionary...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Successfully loaded weights from epoch {checkpoint.get('epoch', 'N/A')}.")
            else:
                model.load_state_dict(checkpoint)
                print("Successfully loaded weights directly from state dictionary.")
        except Exception as e:
            print(f"Error loading checkpoint weights: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Warning: Checkpoint file not found. Exporting model with randomly initialized weights.")
        print("This is normal if compiling prior to running a full training loop.")
    print("-" * 60)
    
    # 3. Compile the model using torch.jit.script
    print("Compiling model using TorchScript (torch.jit.script)...")
    try:
        scripted_model = torch.jit.script(model)
        print("Model compiled successfully to TorchScript format!")
    except Exception as e:
        print(f"Failed to script the model: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Serialize and save the production binary
    print(f"Saving compiled production binary to {output_path}...")
    try:
        scripted_model.save(str(output_path))
        print("Serialization completed successfully.")
    except Exception as e:
        print(f"Failed to save the serialized model: {e}", file=sys.stderr)
        sys.exit(1)
    print("-" * 60)
    
    # 5. Load and verify the serialized model
    print("Verifying the serialized TorchScript binary...")
    try:
        loaded_model = torch.jit.load(str(output_path))
        print("  - Successfully loaded binary using torch.jit.load!")
        
        # Test forward pass with a dummy tensor
        dummy_input = torch.randn(1, 4, 128, 128, 128)
        with torch.no_grad():
            outputs = loaded_model(dummy_input)
            
        print(f"  - Dummy input shape: {dummy_input.shape}")
        print(f"  - Inference outputs shape: {outputs.shape}")
        
        assert outputs.shape == (1, 4, 128, 128, 128), "Output shape mismatch!"
        print("  - Verification outputs shape test PASSED!")
        print("=" * 60)
        print("TORCHSCRIPT COMPILATION AND EXPORT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Verification check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
