#!/usr/bin/env python3
"""
export_coreml.py - Export trained PyTorch model to CoreML format (.mlmodel) with float16 precision
"""

import argparse
import pathlib
import torch
import timm
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import sys
from typing import Dict, Any, Tuple


def load_checkpoint(model_path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model checkpoint and extract configuration"""
    print(f"Loading checkpoint from: {model_path}")
    
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    if 'args' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'args'. Cannot determine model configuration.")
    
    args = checkpoint['args']
    
    config = {
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'input_height': args.input_height,
        'input_width': args.input_width,
    }
    
    print(f"Model configuration:")
    print(f"  Architecture: {config['model_name']}")
    print(f"  Input size: {config['input_width']}x{config['input_height']}")
    print(f"  Number of classes: {config['num_classes']}")
    
    return checkpoint, config


def create_model(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> torch.nn.Module:
    """Create and load the PyTorch model"""
    print(f"\nCreating model: {config['model_name']}")
    
    model = timm.create_model(
        config['model_name'], 
        pretrained=False, 
        num_classes=config['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def export_to_coreml(
    model: torch.nn.Module, 
    config: Dict[str, Any], 
    output_path: pathlib.Path, 
    use_float16: bool = True
) -> ct.models.MLModel:
    """Export PyTorch model to CoreML format"""
    
    input_height = config['input_height']
    input_width = config['input_width']
    
    print(f"\nPreparing for CoreML conversion...")
    print(f"Input shape: (1, 3, {input_height}, {input_width})")
    
    # Create example input for tracing
    example_input = torch.rand(1, 3, input_height, input_width)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Define input shape with flexible batch size
    input_shape = ct.Shape(shape=(1, 3, input_height, input_width))
    
    # Convert to CoreML
    print("Converting to CoreML...")
    
    # Define class labels
    class_labels = ['no_action', 'action'] if config['num_classes'] == 2 else [f'class_{i}' for i in range(config['num_classes'])]
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        classifier_config=ct.ClassifierConfig(class_labels),
        compute_precision=ct.precision.FLOAT16 if use_float16 else ct.precision.FLOAT32,
    )
    
    # Add metadata
    mlmodel.author = "Bowling Action Classifier"
    mlmodel.short_description = f"Bowling action detection model ({config['model_name']})"
    mlmodel.version = "1.0"
    
    # Add input/output descriptions
    mlmodel.input_description["input"] = f"Input image tensor of shape (1, 3, {input_height}, {input_width})"
    mlmodel.output_description["output"] = "Class probabilities"
    
    # Save the model
    print(f"\nSaving CoreML model to: {output_path}")
    mlmodel.save(str(output_path))
    
    # Get model size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model saved successfully!")
    print(f"File size: {size_mb:.2f} MB")
    print(f"Precision: {'float16' if use_float16 else 'float32'}")
    
    return mlmodel


def validate_coreml_model(mlmodel: ct.models.MLModel, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the CoreML model with a test input"""
    print("\nValidating CoreML model...")
    
    import numpy as np
    
    input_height = config['input_height']
    input_width = config['input_width']
    
    # Create a random test input
    test_input = np.random.rand(1, 3, input_height, input_width).astype(np.float32)
    
    # Run prediction
    predictions = mlmodel.predict({"input": test_input})
    
    print("Validation successful!")
    print(f"Output keys: {list(predictions.keys())}")
    print(f"Output shape: {predictions['output'].shape if 'output' in predictions else 'N/A'}")
    
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch bowling classifier to CoreML format (.mlmodel) with float16 precision"
    )
    
    parser.add_argument(
        'model_path',
        type=pathlib.Path,
        help='Path to the PyTorch model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--output',
        type=pathlib.Path,
        default=None,
        help='Output path for CoreML model (.mlmodel). Default: same name as input with .mlmodel extension'
    )
    
    parser.add_argument(
        '--float32',
        action='store_true',
        help='Use float32 precision instead of float16 (larger model size)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of the exported model'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        output_path = args.model_path.with_suffix('.mlmodel')
    else:
        output_path = args.output
        if output_path.suffix != '.mlmodel':
            output_path = output_path.with_suffix('.mlmodel')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load checkpoint
        checkpoint, config = load_checkpoint(args.model_path)
        
        # Create model
        model = create_model(checkpoint, config)
        
        # Export to CoreML
        use_float16 = not args.float32
        mlmodel = export_to_coreml(model, config, output_path, use_float16)
        
        # Validate
        if not args.no_validate:
            validate_coreml_model(mlmodel, config)
        
        print("\n" + "="*50)
        print("Export completed successfully!")
        print(f"CoreML model saved to: {output_path}")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during export: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
