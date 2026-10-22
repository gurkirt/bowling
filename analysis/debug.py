#!/usr/bin/env python3
"""
test.py - Test a trained bowling classifier on a single image
"""

import os
import argparse
import torch.nn.functional as F
import pathlib
import sys
from pathlib import Path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
from modellib import Device, Model  # noqa: E402
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Test bowling action classifier on single image')
    parser.add_argument('image_path', help='Path to input image', type=Path)
    parser.add_argument('model_path', help='Path to model checkpoint (.pth file) or model directory', type=Path)
    parser.add_argument('--device', type=Device.from_string, default=Device.CPU, help='Device to use (cpu/cuda)')
    args = parser.parse_args()
    is_action = Model(args.model_path, args.device)
    image = Image.open(str(args.image_path)).convert('RGB')
    is_bowling, score = is_action(image)
    if is_bowling:
        print("🎳 Strong bowling action detected!")
    else:
        print("❌ No bowling action detected")
    print(f"Confidence score: {score:.4f}")


if __name__ == "__main__":
    main()