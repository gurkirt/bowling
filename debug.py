#!/usr/bin/env python3
"""
test.py - Test a trained bowling classifier on a single image
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm
from enum import Enum
from pathlib import Path
from typing import Any, Tuple


class Device(str, Enum):
    CPU = 'cpu'
    CUDA = 'cuda'

    @classmethod
    def from_string(cls, value: str) -> "Device" :
        match value:
            case ""|"cpu":
                return cls.CPU
            case "cuda" | "gpu":
                return cls.CUDA
            case _:
                raise ValueError(f"Invalid device '{value}'. Choose from: {', '.join(d.value for d in cls)}")

def path(str) -> Path:
    p = Path(str)
    if not p.exists():
        raise ValueError(f"file not found: {p}")
    return p


def load_model(model_path: Path, device: Device) -> Tuple[Any, int , int]:
    """Load trained model from checkpoint"""    
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    model = timm.create_model(checkpoint['args'].model_name, pretrained=False, num_classes=checkpoint['args'].num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.device(device.value))
    model.eval()
    return model, checkpoint['args'].input_height, checkpoint['args'].input_width 


def get_transforms(input_height=320, input_width=128) -> transforms.Compose:
    """Get test transforms matching training"""
    return transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def predict_is_in_action(model, image: Image.Image, transform, device) -> bool:
    """Return True if the model predicts the action class for a given PIL RGB image."""
    device_t = torch.device(device.value) if isinstance(device, Device) else device
    input_tensor = transform(image).unsqueeze(0).to(device_t)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class == 1


def main():
    parser = argparse.ArgumentParser(description='Test bowling action classifier on single image')
    parser.add_argument('image_path', help='Path to input image', type=path)
    parser.add_argument('model_path', help='Path to model checkpoint (.pth file) or model directory', type=path)
    parser.add_argument('--device', type=Device.from_string, default=Device.CPU, choices=list(Device), help='Device to use (cpu/cuda)')
    args = parser.parse_args()
    
    model, height, width = load_model(args.model_path, args.device)
    print(model)
    transform = get_transforms(height, width)
    
    image = Image.open(str(args.image_path)).convert('RGB')
    is_bowling = predict_is_in_action(model, image, transform, args.device)

    if is_bowling:
        print("🎳 Strong bowling action detected!")
    else:
        print("❌ No bowling action detected")
        

if __name__ == "__main__":
    main()