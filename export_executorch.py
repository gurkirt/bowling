#!/usr/bin/env python3
"""
export_executorch.py - Export a trained PyTorch model for on-device inference.

Three export targets are supported:
  * ExecuTorch .pte (XNNPACK)  - CPU delegate, portable (iOS / Android / desktop).
  * ExecuTorch .pte (CoreML)   - Apple Neural Engine (ANE/NPU) via the ExecuTorch CoreML delegate.
  * Core ML .mlpackage         - native Core ML model (MLProgram), loaded by Apple's Core ML runtime.

Reparameterizable backbones (FastViT, MobileOne) are fused for inference by default.
Use --compare to export all three and check numerical parity against the fp32 PyTorch model.

Precision: fp32 (default) or fp16 (--fp16). CoreML / mlpackage targets require macOS + coremltools.
"""

import argparse
import pathlib
import sys
from typing import Any, Dict, Tuple

import torch
import timm


def load_checkpoint(model_path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model checkpoint and extract configuration.

    Accepts two formats:
      * a portable artifact (from export_pytorch.py) with a 'config' dict, or
      * a raw training checkpoint with an argparse 'args' Namespace.
    """
    print(f"Loading checkpoint from: {model_path}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    if "config" in checkpoint:
        # Portable artifact produced by export_pytorch.py.
        cfg = checkpoint["config"]
        config = {
            "model_name": cfg["model_name"],
            "num_classes": cfg["num_classes"],
            "input_height": cfg["input_height"],
            "input_width": cfg["input_width"],
        }
    elif "args" in checkpoint:
        args = checkpoint["args"]
        config = {
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "input_height": args.input_height,
            "input_width": args.input_width,
        }
    else:
        raise ValueError(
            "Checkpoint contains neither 'config' nor 'args'. Cannot determine model configuration."
        )

    print("Model configuration:")
    print(f"  Architecture: {config['model_name']}")
    print(f"  Input size: {config['input_width']}x{config['input_height']}")
    print(f"  Number of classes: {config['num_classes']}")

    return checkpoint, config


def create_model(checkpoint: Dict[str, Any], config: Dict[str, Any], reparameterize: bool = True) -> torch.nn.Module:
    """Create and load the PyTorch model, wrapped with softmax to match the CoreML export."""
    print(f"\nCreating model: {config['model_name']}")

    base_model = timm.create_model(
        config["model_name"],
        pretrained=False,
        num_classes=config["num_classes"],
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    # Reparameterize structurally-reparameterizable backbones (FastViT, MobileOne, etc.)
    # for inference: multi-branch train-time blocks are fused into a single conv path,
    # which is smaller and faster on device. Apple's FastViT exporter always does this.
    # The fused weights are mathematically equivalent, so accuracy is unchanged.
    if reparameterize:
        from timm.utils import reparameterize_model

        before = sum(p.numel() for p in base_model.parameters())
        try:
            base_model = reparameterize_model(base_model)
            base_model.eval()
            after = sum(p.numel() for p in base_model.parameters())
            print(f"Reparameterized for inference: params {before:,} -> {after:,}")
        except Exception as exc:
            print(f"Reparameterization skipped ({type(exc).__name__}: {exc}); exporting train-time graph.")

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
    backend: str = "xnnpack",
    compute_unit: str = "all",
    min_ios: int = 17,
) -> None:
    """Export a PyTorch model to an ExecuTorch .pte file.

    backend="xnnpack": CPU delegate. fp16 is applied by casting model/inputs to half.
    backend="coreml":  Apple CoreML delegate. Dispatches to the Neural Engine (ANE/NPU)
                       on iOS/macOS. The model/inputs stay fp32 in Python; precision is
                       controlled by the CoreML 'compute_precision' compile spec (fp16),
                       and the target hardware (ANE) by 'compute_unit'. Requires macOS +
                       coremltools to build, and Apple hardware to run.
    """
    input_height = config["input_height"]
    input_width = config["input_width"]

    print(f"\nPreparing for ExecuTorch ({backend.upper()}) conversion...")
    print(f"Input shape: (1, 3, {input_height}, {input_width})")
    print(f"Precision: {'float16' if use_fp16 else 'float32'}")

    example_input = torch.rand(1, 3, input_height, input_width)

    if backend == "coreml":
        import coremltools as ct
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner

        compute_unit_map = {
            "all": ct.ComputeUnit.ALL,            # CoreML scheduler may use ANE+GPU+CPU
            "ane": ct.ComputeUnit.CPU_AND_NE,     # restrict to ANE (NPU) + CPU fallback
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu": ct.ComputeUnit.CPU_ONLY,
        }
        target_map = {15: ct.target.iOS15, 16: ct.target.iOS16, 17: ct.target.iOS17, 18: ct.target.iOS18}
        ct_compute_unit = compute_unit_map[compute_unit]
        ct_target = target_map.get(min_ios, ct.target.iOS17)

        print(f"CoreML compute_unit: {ct_compute_unit.name} (NPU/ANE: "
              f"{'yes' if compute_unit in ('all', 'ane') else 'no'})")
        print(f"CoreML minimum_deployment_target: iOS{min_ios}")

        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_unit=ct_compute_unit,
            minimum_deployment_target=ct_target,
            compute_precision=ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32,
        )
        partitioner = CoreMLPartitioner(compile_specs=compile_specs)

        # 1) Capture an ExportedProgram via torch.export (model stays fp32 for CoreML).
        print("Exporting with torch.export...")
        with torch.no_grad():
            exported = torch.export.export(model, (example_input,))

        # 2) Lower to Edge dialect and delegate supported ops to the CoreML backend.
        print("Lowering to Edge + CoreML delegate...")
        et_program = to_edge_transform_and_lower(
            exported,
            partitioner=[partitioner],
        ).to_executorch()
    else:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

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


def export_to_mlpackage(
    model: torch.nn.Module,
    config: Dict[str, Any],
    output_path: pathlib.Path,
    use_fp16: bool = False,
    compute_unit: str = "all",
    min_ios: int = 17,
) -> None:
    """Export a PyTorch model to a native Core ML .mlpackage (MLProgram) via coremltools.

    This is the direct Core ML path (no ExecuTorch). It traces the model and converts it
    with coremltools, targeting the Neural Engine (ANE/NPU) when compute_unit allows.
    Requires macOS + coremltools.
    """
    import coremltools as ct

    input_height = config["input_height"]
    input_width = config["input_width"]

    print("\nPreparing for Core ML (.mlpackage / MLProgram) conversion...")
    print(f"Input shape: (1, 3, {input_height}, {input_width})")
    print(f"Precision: {'float16' if use_fp16 else 'float32'}")

    compute_unit_map = {
        "all": ct.ComputeUnit.ALL,
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }
    target_map = {15: ct.target.iOS15, 16: ct.target.iOS16, 17: ct.target.iOS17, 18: ct.target.iOS18}
    ct_compute_unit = compute_unit_map[compute_unit]
    ct_target = target_map.get(min_ios, ct.target.iOS17)

    print(f"CoreML compute_unit: {ct_compute_unit.name} (NPU/ANE: "
          f"{'yes' if compute_unit in ('all', 'ane') else 'no'})")
    print(f"CoreML minimum_deployment_target: iOS{min_ios}")

    example_input = torch.rand(1, 3, input_height, input_width)
    print("Tracing model with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), example_input)

    print("Converting to MLProgram...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example_input.shape)],
        outputs=[ct.TensorType(name="probs")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32,
        compute_units=ct_compute_unit,
        minimum_deployment_target=ct_target,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nModel saved successfully to: {output_path}")
    print(f"Total size: {size_mb:.2f} MB")


def _run_executorch_model(output_path: pathlib.Path, test_input: torch.Tensor):
    """Execute a .pte through the ExecuTorch runtime and return the first output tensor."""
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
    return outputs[0]


def _run_mlpackage_model(output_path: pathlib.Path, test_input_np):
    """Run a Core ML .mlpackage via coremltools.predict and return the output array."""
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(output_path))
    spec = mlmodel.get_spec()
    in_name = spec.description.input[0].name
    out_name = spec.description.output[0].name
    preds = mlmodel.predict({in_name: test_input_np})
    return preds[out_name]


def validate_executorch_model(output_path: pathlib.Path, config: Dict[str, Any], use_fp16: bool) -> None:
    """Run the exported .pte through the ExecuTorch runtime with a random input."""
    print("\nValidating ExecuTorch model...")

    input_height = config["input_height"]
    input_width = config["input_width"]
    dtype = torch.float16 if use_fp16 else torch.float32
    test_input = torch.rand(1, 3, input_height, input_width, dtype=dtype)

    out = _run_executorch_model(output_path, test_input)
    print("Validation successful!")
    print(f"  output: shape {tuple(out.shape)}, dtype {out.dtype}")
    print(f"  probabilities: {out.flatten().tolist()}")


def compare_all_models(
    model: torch.nn.Module,
    config: Dict[str, Any],
    base_output: pathlib.Path,
    use_fp16: bool,
    compute_unit: str,
    min_ios: int,
) -> None:
    """Export all three targets and compare their outputs against the fp32 PyTorch model.

    The fp32 PyTorch (reparameterized + softmax) model is the reference. Each artifact is
    exported at the requested precision and run on the *same* fixed random input; the max
    absolute probability difference vs. the reference is reported. Backends that cannot be
    built/run in the current environment (e.g. CoreML on non-macOS) are skipped gracefully.
    """
    import numpy as np

    input_height = config["input_height"]
    input_width = config["input_width"]

    torch.manual_seed(0)
    input_fp32 = torch.rand(1, 3, input_height, input_width)
    input_np = input_fp32.numpy()

    # Reference: fp32 PyTorch.
    with torch.no_grad():
        ref = model(input_fp32).float().numpy().flatten()
    print("\n" + "=" * 60)
    print("COMPARISON vs fp32 PyTorch reference")
    print("=" * 60)
    prec = "float16" if use_fp16 else "float32"
    print(f"Reference (pytorch_fp32) probabilities: {ref.tolist()}")
    print(f"Exported-artifact precision: {prec}\n")

    rows = []

    # 1) ExecuTorch XNNPACK.
    try:
        pte_x = base_output.with_name(base_output.stem + "_xnnpack.pte")
        export_to_executorch(model, config, pte_x, use_fp16=use_fp16, backend="xnnpack")
        et_in = input_fp32.half() if use_fp16 else input_fp32
        out = _run_executorch_model(pte_x, et_in)
        probs = out.float().numpy().flatten()
        rows.append(("executorch_xnnpack", probs, float(np.max(np.abs(probs - ref)))))
    except Exception as exc:
        rows.append(("executorch_xnnpack", None, f"skipped: {type(exc).__name__}: {exc}"))

    # 2) ExecuTorch CoreML (ANE).
    try:
        if sys.platform != "darwin":
            raise RuntimeError("CoreML delegate requires macOS/Apple hardware")
        pte_c = base_output.with_name(base_output.stem + "_coreml.pte")
        export_to_executorch(
            model, config, pte_c, use_fp16=use_fp16, backend="coreml",
            compute_unit=compute_unit, min_ios=min_ios,
        )
        out = _run_executorch_model(pte_c, input_fp32)
        probs = out.float().numpy().flatten()
        rows.append(("executorch_coreml", probs, float(np.max(np.abs(probs - ref)))))
    except Exception as exc:
        rows.append(("executorch_coreml", None, f"skipped: {type(exc).__name__}: {exc}"))

    # 3) Native Core ML .mlpackage.
    try:
        if sys.platform != "darwin":
            raise RuntimeError("Core ML .mlpackage prediction requires macOS")
        mlpkg = base_output.with_name(base_output.stem + ".mlpackage")
        export_to_mlpackage(
            model, config, mlpkg, use_fp16=use_fp16, compute_unit=compute_unit, min_ios=min_ios,
        )
        out = _run_mlpackage_model(mlpkg, input_np)
        probs = np.asarray(out).flatten()
        rows.append(("coreml_mlpackage", probs, float(np.max(np.abs(probs - ref)))))
    except Exception as exc:
        rows.append(("coreml_mlpackage", None, f"skipped: {type(exc).__name__}: {exc}"))

    print("\n" + "-" * 60)
    print(f"{'artifact':<22}{'max|Δprob| vs fp32':<22}{'probs'}")
    print("-" * 60)
    for name, probs, diff in rows:
        if probs is None:
            print(f"{name:<22}{diff}")
        else:
            print(f"{name:<22}{diff:<22.6e}{probs.tolist()}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch bowling classifier to ExecuTorch (.pte) using the XNNPACK (CPU) or CoreML (Apple Neural Engine / NPU) backend."
    )
    parser.add_argument("model_path", type=pathlib.Path, help="Path to the PyTorch checkpoint (.pth).")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path for the ExecuTorch model (.pte). Default: same name as input with .pte extension.",
    )
    parser.add_argument(
        "--backend",
        choices=["xnnpack", "coreml", "mlpackage"],
        default="xnnpack",
        help="Export target: 'xnnpack' (ExecuTorch CPU .pte, portable), 'coreml' (ExecuTorch CoreML/ANE .pte), or 'mlpackage' (native Core ML MLProgram). CoreML targets require macOS + coremltools. Default: xnnpack.",
    )
    parser.add_argument(
        "--compute-unit",
        choices=["all", "ane", "gpu", "cpu"],
        default="all",
        help="CoreML target hardware. 'all' lets CoreML use the Neural Engine when beneficial; 'ane' restricts to ANE(NPU)+CPU. Only used with --backend coreml. Default: all.",
    )
    parser.add_argument(
        "--min-ios",
        type=int,
        choices=[15, 16, 17, 18],
        default=17,
        help="CoreML minimum deployment target (iOS version). Only used with --backend coreml. Default: 17.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in float16 precision (smaller, often faster on mobile). Default is float32.",
    )
    parser.add_argument(
        "--no-reparameterize",
        action="store_true",
        help="Do not reparameterize/fuse inference branches (FastViT/MobileOne). By default reparameterization is applied for a smaller, faster on-device model.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Export all three targets (xnnpack .pte, coreml .pte, mlpackage) and compare their outputs against the fp32 PyTorch model. Backends unavailable in this environment are skipped.",
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

    target_suffix = ".mlpackage" if args.backend == "mlpackage" else ".pte"
    if args.output is None:
        output_path = args.model_path.with_suffix(target_suffix)
    else:
        output_path = args.output
        if output_path.suffix != target_suffix:
            output_path = output_path.with_suffix(target_suffix)

    try:
        checkpoint, config = load_checkpoint(args.model_path)
        model = create_model(checkpoint, config, reparameterize=not args.no_reparameterize)

        if args.compare:
            compare_all_models(
                model,
                config,
                output_path,
                use_fp16=args.fp16,
                compute_unit=args.compute_unit,
                min_ios=args.min_ios,
            )
            print("\n" + "=" * 50)
            print("Comparison completed!")
            print("=" * 50)
            return

        if args.backend == "mlpackage":
            export_to_mlpackage(
                model,
                config,
                output_path,
                use_fp16=args.fp16,
                compute_unit=args.compute_unit,
                min_ios=args.min_ios,
            )
        else:
            export_to_executorch(
                model,
                config,
                output_path,
                use_fp16=args.fp16,
                backend=args.backend,
                compute_unit=args.compute_unit,
                min_ios=args.min_ios,
            )

        # CoreML / mlpackage targets the Apple Neural Engine and can only be executed
        # on Apple hardware, so a non-macOS host cannot validate them.
        if not args.no_validate:
            if args.backend in ("coreml", "mlpackage") and sys.platform != "darwin":
                print("\nSkipping validation: a CoreML/mlpackage artifact can only run on Apple hardware (macOS/iOS).")
            elif args.backend == "mlpackage":
                print("\nValidating Core ML model...")
                import numpy as np

                test_input = torch.rand(1, 3, config["input_height"], config["input_width"]).numpy()
                out = _run_mlpackage_model(output_path, test_input)
                print("Validation successful!")
                print(f"  probabilities: {np.asarray(out).flatten().tolist()}")
            else:
                validate_executorch_model(output_path, config, use_fp16=args.fp16)

        print("\n" + "=" * 50)
        print("Export completed successfully!")
        print(f"Model saved to: {output_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during export: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
