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
import json
import numpy as np


def get_transforms(input_height=320, input_width=128):
    """Get test transforms matching training"""
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_model(checkpoint_path, model_name='efficientnet_b0', num_classes=2, device='auto'):
    """Load trained model from checkpoint"""
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Loading model on device: {device}")
    
    # Create model architecture
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            if 'args' in checkpoint:
                saved_args = checkpoint['args']
                print(f"Model details: {saved_args.model_name if hasattr(saved_args, 'model_name') else 'unknown'}")
                print(f"Best val acc: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_image(model, image_path, transform, device, class_names=None):
    """Predict single image"""
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        print(f"Original image size: {image.size}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get class probabilities
    class_probs = probabilities[0].cpu().numpy()
    
    # Prepare results
    if class_names is None:
        class_names = ['No Action', 'Action']  # Default for bowling classifier
    
    results = {
        'predicted_class': predicted_class,
        'predicted_label': class_names[predicted_class],
        'confidence': confidence,
        'class_probabilities': {
            class_names[i]: float(prob) for i, prob in enumerate(class_probs)
        },
        'raw_logits': outputs[0].cpu().numpy().tolist()
    }
    
    return results


def load_model_config(model_dir):
    """Try to load model configuration from training results"""
    config_files = ['results.json', 'validation_results.json']
    
    for config_file in config_files:
        config_path = os.path.join(model_dir, config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    args = config.get('args', {})
                    return {
                        'model_name': args.get('model_name', 'efficientnet_b0'),
                        'input_height': args.get('input_height', 320),
                        'input_width': args.get('input_width', 128),
                        'num_classes': args.get('num_classes', 2),
                        'best_val_acc': config.get('best_val_acc'),
                        'exp_name': config.get('exp_name')
                    }
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                continue
    
    print("Warning: No model config found, using defaults")
    return {
        'model_name': 'efficientnet_b0',
        'input_height': 320,
        'input_width': 128,
        'num_classes': 2,
        'best_val_acc': None,
        'exp_name': None
    }


def main():
    parser = argparse.ArgumentParser(description='Test bowling action classifier on single image')
    
    # Required arguments
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('model_path', help='Path to model checkpoint (.pth file) or model directory')
    
    # Optional arguments
    parser.add_argument('--model_name', default=None, 
                       help='Model architecture (auto-detected from config if available)')
    parser.add_argument('--input_height', type=int, default=None,
                       help='Input image height (auto-detected from config if available)')
    parser.add_argument('--input_width', type=int, default=None,
                       help='Input image width (auto-detected from config if available)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected from config if available)')
    parser.add_argument('--device', default='auto', 
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--class_names', nargs='+', default=None,
                       help='Class names (default: "No Action" "Action")')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed prediction information')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Determine model checkpoint path
    if os.path.isdir(args.model_path):
        # Model directory provided, look for best_model.pth
        checkpoint_path = os.path.join(args.model_path, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Error: best_model.pth not found in {args.model_path}")
            return
        model_dir = args.model_path
    else:
        # Direct checkpoint path provided
        checkpoint_path = args.model_path
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
        model_dir = os.path.dirname(checkpoint_path)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model configuration
    config = load_model_config(model_dir)
    
    # Override config with command line arguments if provided
    model_name = args.model_name or config['model_name']
    input_height = args.input_height or config['input_height']
    input_width = args.input_width or config['input_width']
    num_classes = args.num_classes or config['num_classes']
    
    print(f"Model config:")
    print(f"  Architecture: {model_name}")
    print(f"  Input size: {input_width}x{input_height}")
    print(f"  Classes: {num_classes}")
    if config['exp_name']:
        print(f"  Experiment: {config['exp_name']}")
    if config['best_val_acc']:
        print(f"  Best val acc: {config['best_val_acc']:.2f}%")
    
    # Load model
    model, device = load_model(checkpoint_path, model_name, num_classes, args.device)
    print(model)
    torch.onnx.export(model, torch.randn(1, 3, input_height, input_width).to(device), "model.onnx", opset_version=18)
    torch.jit.save(torch.jit.script(model), "model.pt")
    # Get transforms
    transform = get_transforms(input_height, input_width)
    
    # Set class names
    class_names = args.class_names
    if class_names is None:
        if num_classes == 2:
            class_names = ['No Action', 'Action']
        else:
            class_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(f"Class names: {class_names}")
    print("-" * 50)
    
    # Predict
    results = predict_image(model, args.image_path, transform, device, class_names)
    
    if results is None:
        print("Prediction failed!")
        return
    
    # Display results
    print("PREDICTION RESULTS:")
    print("=" * 50)
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"Predicted class: {results['predicted_class']} ({results['predicted_label']})")
    print(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    
    print("\nClass probabilities:")
    for class_name, prob in results['class_probabilities'].items():
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    if args.verbose:
        print(f"\nRaw logits: {results['raw_logits']}")
        print(f"Device used: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple interpretation
    print("\nInterpretation:")
    if results['predicted_class'] == 1:  # Assuming 1 is "Action"
        if results['confidence'] > 0.8:
            print("🎳 Strong bowling action detected!")
        elif results['confidence'] > 0.6:
            print("🎳 Bowling action detected (moderate confidence)")
        else:
            print("🎳 Possible bowling action (low confidence)")
    else:
        if results['confidence'] > 0.8:
            print("❌ No bowling action detected")
        elif results['confidence'] > 0.6:
            print("❌ Likely no bowling action")
        else:
            print("❓ Uncertain prediction")


if __name__ == "__main__":
    main()