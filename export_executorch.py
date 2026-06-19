#!/usr/bin/env python3
"""
export_executorch.py - Export trained PyTorch model to ExecuTorch (.pte) using the XNNPACK backend.

This mirrors export_coreml.py but produces a portable ExecuTorch program that runs on the
ExecuTorch runtime (iOS / Android / desktop) with no coremltools dependency.

Backend: XNNPACK (CPU). Precision: fp32 (default) or fp16 (--fp16).
"""

import argparse
import pathlib
import sys
from typing import Any, Dict, Tuple

import torch
import timm


def load_checkpoint(model_path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model checkpoint and extract configuration."""
    print(f"Loading checkpoint from: {model_path}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    if "args" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'args'. Cannot determine model configuration.")

    args = checkpoint["args"]
    config = {
        "model_name": args.model_name,
        "num_classes": args.num_classes,
        "input_height": args.input_height,
        "input_width": args.input_width,
    }

    print("Model configuration:")
    print(f"  Architecture: {config['model_name']}")
    print(f"  Input size: {config['input_width']}x{config['input_height']}")
    print(f"  Number of classes: {config['num_classes']}")

    return checkpoint, config


def create_model(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> torch.nn.Module:
    """Create and load the PyTorch model, wrapped with softmax to match the CoreML export."""
    print(f"\nCreating model: {config['model_name']}")

    base_model = timm.create_model(
        config["model_name"],
        pretrained=False,
        num_classes=config["num_classes"],
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    class ModelWithSoftmax(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.base_model(x)
            return torch.nn.functional.softmax(logits, dim=1)

    model = ModelWithSoftmax(base_model)
    model.eval()

    print("Model loaded successfully with softmax normalization")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def export_to_executorch(
    model: torch.nn.Module,
    config: Dict[str, Any],
    output_path: pathlib.Path,
    use_fp16: bool = False,
) -> None:
    """Export a PyTorch model to an ExecuTorch .pte file using the XNNPACK backend."""
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    input_height = config["input_height"]
    input_width = config["input_width"]

    print("\nPreparing for ExecuTorch (XNNPACK) conversion...")
    print(f"Input shape: (1, 3, {input_height}, {input_width})")
    print(f"Precision: {'float16' if use_fp16 else 'float32'}")

    example_input = torch.rand(1, 3, input_height, input_width)
    if use_fp16:
        model = model.half()
        example_input = example_input.half()

    # 1) Capture an ExportedProgram via torch.export (PT2 path).
    print("Exporting with torch.export...")
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,))

    # 2) Lower to Edge dialect and delegate supported ops to XNNPACK.
    print("Lowering to Edge + XNNPACK delegate...")
    et_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    # 3) Serialize the .pte.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel saved successfully to: {output_path}")
    print(f"Total size: {size_mb:.2f} MB")


def validate_executorch_model(output_path: pathlib.Path, config: Dict[str, Any], use_fp16: bool) -> None:
    """Run the exported .pte through the ExecuTorch runtime with a random input."""
    print("\nValidating ExecuTorch model...")

    input_height = config["input_height"]
    input_width = config["input_width"]
    dtype = torch.float16 if use_fp16 else torch.float32
    test_input = torch.rand(1, 3, input_height, input_width, dtype=dtype)

    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(str(output_path))
        method = program.load_method("forward")
        outputs = method.execute([test_input])
    except Exception:
        # Fallback to the lower-level pybindings API on older ExecuTorch versions.
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        module = _load_for_executorch(str(output_path))
        outputs = module.forward([test_input])

    out = outputs[0]
    print("Validation successful!")
    print(f"  output: shape {tuple(out.shape)}, dtype {out.dtype}")
    print(f"  probabilities: {out.flatten().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch bowling classifier to ExecuTorch (.pte) using the XNNPACK backend."
    )
    parser.add_argument("model_path", type=pathlib.Path, help="Path to the PyTorch checkpoint (.pth).")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path for the ExecuTorch model (.pte). Default: same name as input with .pte extension.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in float16 precision (smaller, often faster on mobile). Default is float32.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of the exported model.",
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_path = args.model_path.with_suffix(".pte")
    else:
        output_path = args.output
        if output_path.suffix != ".pte":
            output_path = output_path.with_suffix(".pte")

    try:
        checkpoint, config = load_checkpoint(args.model_path)
        model = create_model(checkpoint, config)
        export_to_executorch(model, config, output_path, use_fp16=args.fp16)

        if not args.no_validate:
            validate_executorch_model(output_path, config, use_fp16=args.fp16)

        print("\n" + "=" * 50)
        print("Export completed successfully!")
        print(f"ExecuTorch model saved to: {output_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during export: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
