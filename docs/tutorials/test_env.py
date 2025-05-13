#!/usr/bin/env python
# Simple test script to verify the environment

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Try importing essential packages
packages_to_test = [
    "numpy", 
    "matplotlib", 
    "torch", 
    "geopandas", 
    "lancedb",
    "stacchip",
    "box"
]

for package in packages_to_test:
    try:
        __import__(package)
        print(f"✓ Successfully imported {package}")
    except ImportError as e:
        print(f"✗ Failed to import {package}: {e}")

# Check for model file
import os
model_paths = [
    "../../checkpoints/clay-v1-base.ckpt",
    "/Users/brunosan/code/model/checkpoints/clay-v1-base.ckpt"
]

for path in model_paths:
    if os.path.exists(path):
        print(f"✓ Model file exists at: {path}")
    else:
        print(f"✗ Model file NOT found at: {path}")