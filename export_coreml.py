#!/usr/bin/env python3
"""
export_coreml.py - Export trained PyTorch model to CoreML format (.mlpackage) with float16 precision
"""

import argparse
import pathlib
import torch
import timm
import coremltools as ct
import sys
from typing import Dict, Any, Tuple, Optional
import platform
import cv2
import numpy as np
from PIL import Image, ImageOps
from modellib import crop


class PreprocessWrapper(torch.nn.Module):
    """Wraps a base model with ImageNet normalization and optional softmax.

    Expects input as NCHW float32 tensor with pixel values in [0, 255].
    Applies: x = (x/255 - mean) / std per channel, then forwards through base.
    Optionally applies softmax to return probabilities.
    """
    def __init__(self, base: torch.nn.Module, softmax_output: bool = True):
        super().__init__()
        self.base = base
        self.softmax_output = softmax_output
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # Register as buffers to trace/script properly and keep on the right device
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = (x - self._mean) / self._std
        logits = self.base(x)
        if self.softmax_output:
            return torch.nn.functional.softmax(logits, dim=1)
        return logits


def print_env_info() -> None:
    """Print environment versions to help debug binary/ABI issues."""
    try:
        import numpy as np
        numpy_ver = np.__version__
    except Exception as e:
        numpy_ver = f"unavailable ({e})"

    timm_ver = getattr(timm, "__version__", "unknown")
    ct_ver = getattr(ct, "__version__", "unknown")

    print("\nEnvironment info:")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Platform:     {platform.platform()} ({platform.machine()})")
    print(f"  NumPy:        {numpy_ver}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  timm:         {timm_ver}")
    print(f"  coremltools:  {ct_ver}\n")
    try:
        from packaging.version import Version
        if numpy_ver != "unavailable" and Version(numpy_ver) >= Version("2.0.0"):
            print("NOTE: NumPy >= 2 detected. If you see ABI errors, downgrade to 'numpy<2' and reinstall binary deps.")
    except Exception:
        pass


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


def create_model(checkpoint: Dict[str, Any], config: Dict[str, Any], scriptable: bool = False) -> torch.nn.Module:
    """Create and load the PyTorch model"""
    print(f"\nCreating model: {config['model_name']} (scriptable={scriptable})")
    
    def _build(scriptable_flag: bool):
        kwargs = dict(pretrained=False, num_classes=config['num_classes'])
        if scriptable_flag:
            # export-friendly module variants
            kwargs.update(dict(scriptable=True))
        return timm.create_model(config['model_name'], **kwargs)

    try:
        base_model = _build(scriptable)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        if scriptable:
            print(f"  Failed to load with scriptable=True ({e}); falling back to default model.")
            base_model = _build(False)
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise
    base_model.eval()
    # print(base_model)
    print(f"Model loaded successfully (base model without preprocessing wrapper)")
    print(f"Total parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    return base_model


def export_to_coreml(
    model: torch.nn.Module,
    config: Dict[str, Any],
    output_path: pathlib.Path,
    use_float16: bool = True,
    scriptable: bool = False,
    backend: str = "mlprogram",
) -> ct.models.MLModel:
    """Export PyTorch model to CoreML format"""
    
    input_height = config['input_height']
    input_width = config['input_width']
    
    print(f"\nPreparing for CoreML conversion...")
    print(f"Input shape: (1, 3, {input_height}, {input_width})")
    
    # Create example input for tracing - try to use a real sample if available
    sample_image_path = pathlib.Path("debug_frames/debug_frame_000012.png")
    if sample_image_path.exists():
        print(f"Using real sample image for tracing: {sample_image_path}")
        from PIL import Image as PILImage
        sample_pil = PILImage.open(sample_image_path).convert('RGB')
        sample_pil = sample_pil.resize((input_width, input_height), resample=PILImage.BILINEAR)
        sample_array = np.array(sample_pil).astype(np.float32)
        example_input = np.transpose(sample_array, (2, 0, 1))[np.newaxis, :]  # HWC -> NCHW, values in [0,255]
        print(f"  Sample input shape: {example_input.shape}, range: [{example_input.min():.3f}, {example_input.max():.3f}]")
    else:
        print(f"Real sample not found, using random data for tracing")
        # Create example input as unnormalized pixels in [0,255]
        example_input = (torch.rand(1, 3, input_height, input_width) * 255.0).numpy().astype(np.float32)
    
    # Build inference wrapper (normalization + softmax for deployment simplicity)
    base_model = model.eval()
    deploy_model = PreprocessWrapper(base_model, softmax_output=True).eval()

    # Trace the deploy wrapper
    print("Tracing deploy model (with preprocessing and softmax)...")
    example_tensor = torch.from_numpy(example_input)
    with torch.no_grad():
        traced_model = torch.jit.trace(deploy_model, example_tensor, strict=False, check_trace=True)
    
    ## print traced model
    # print("Traced model structure:")
    # print(traced_model)
    print("Converting JIT model to CoreML...")
    convert_kwargs = {
        "convert_to": backend,
        # Use TensorType so Python predict accepts NCHW float32 arrays directly
        "inputs": [ct.TensorType(name="image", shape=tuple(example_tensor.shape))],
    }
    # Set deployment target based on backend
    try:
        if backend == "mlprogram":
            target = getattr(ct.target, "iOS16", None) or getattr(ct.target, "macOS13", None)
        else:
            target = (
                getattr(ct.target, "iOS14", None)
                or getattr(ct.target, "iOS13", None)
                or getattr(ct.target, "macOS11", None)
                or getattr(ct.target, "macOS10_15", None)
            )
        if target is not None:
            convert_kwargs["minimum_deployment_target"] = target
    except Exception:
        pass

    precision = getattr(ct, "precision", None)
    if backend == "mlprogram":
        if precision is not None:
            convert_kwargs["compute_precision"] = precision.FLOAT16 if use_float16 else precision.FLOAT32
            exported_precision = "float16" if use_float16 else "float32"
        else:
            exported_precision = "float32"
            if use_float16:
                print("  Warning: coremltools.precision not available; exporting in float32.")
    else:
        # NN backend ignores compute_precision; we'll optionally compress weights to FP16 after conversion
        exported_precision = "float32" if not use_float16 else "float16"

    mlmodel = ct.convert(traced_model, **convert_kwargs)
    print("CoreML conversion complete")

    # Defer output renaming to after initial save for MLProgram (needs weights_dir)

    # Determine current output name (pre-rename) for parity check
    try:
        spec = mlmodel.get_spec()
        output_names = [output.name for output in spec.description.output]
        final_output_name = output_names[0] if len(output_names) > 0 else None
    except Exception:
        final_output_name = "probs"
    # Quick parity check: compare probabilities on the example input
    try:
        with torch.no_grad():
            torch_probs = traced_model(example_tensor).cpu().numpy().reshape(-1)
        coreml_out = mlmodel.predict({"image": example_input.astype(np.float32)})
        out_key = final_output_name if final_output_name else next(iter(coreml_out.keys()))
        cm_values = np.array(coreml_out[out_key], dtype=np.float32).reshape(-1)
        diff = float(np.max(np.abs(cm_values - torch_probs)))
        print(f"  CoreML parity (probs): max|Δ|={diff:.6f}")
    except Exception as e:
        print(f"  Skipping parity check due to: {e}")

    # For NN backend, optionally compress weights to FP16 explicitly if requested
    if backend == "neuralnetwork" and use_float16 and hasattr(ct, "utils") and hasattr(ct.utils, "convert_neural_network_weights_to_fp16"):
        try:
            mlmodel = ct.utils.convert_neural_network_weights_to_fp16(mlmodel)
            exported_precision = "float16"
            print("  Compressed NN weights to FP16 for smaller size.")
        except Exception as e_fp16:
            print(f"  Skipping NN FP16 weight compression due to: {e_fp16}")
    
    # Add metadata
    mlmodel.author = "Bowling Action Classifier"
    mlmodel.short_description = f"Bowling action detection model ({config['model_name']})"
    mlmodel.version = "1.0"
    
    # Add input/output descriptions
    mlmodel.input_description["image"] = (
        "RGB image input (cropped externally), NCHW float32 with pixel values in [0,255]. "
        "Model applies ImageNet normalization and outputs class probabilities (softmax)."
    )
    
    # Save the model (first pass)
    print(f"\nSaving CoreML model to: {output_path}")
    mlmodel.save(str(output_path))

    # Inspect outputs and set metadata/description without renaming
    try:
        spec = mlmodel.get_spec()
        output_names = [output.name for output in spec.description.output]
        print(f"Output feature names: {output_names}")
        if len(output_names) > 0:
            try:
                mlmodel.output_description[output_names[0]] = "Class probabilities (softmax) as a dense array ordered by class index"
                mlmodel.user_defined_metadata["output_name"] = output_names[0]
            except Exception:
                pass
    except Exception:
        pass
    
    # Get model size (handle both directory and file)
    if output_path.is_dir():
        # .mlpackage is a directory
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
    else:
        # .mlmodel is a single file
        size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"Model saved successfully!")
    print(f"Total size: {size_mb:.2f} MB")
    print(f"Precision: {exported_precision}")
    print(f"Format: {'ML Package (.mlpackage)' if output_path.suffix == '.mlpackage' else 'ML Model (.mlmodel)'}")

    # Also save a copy into ./line&length/line&length/best_model.mlpackage
    try:
        secondary_path = pathlib.Path('line&length/line&length/best_model.mlpackage')
        secondary_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Also saving CoreML model to: {secondary_path}")
        mlmodel.save(str(secondary_path))
        if secondary_path.is_dir():
            sec_size_mb = sum(f.stat().st_size for f in secondary_path.rglob('*') if f.is_file()) / (1024 * 1024)
        else:
            sec_size_mb = secondary_path.stat().st_size / (1024 * 1024)
        print(f"  Saved duplicate successfully (size: {sec_size_mb:.2f} MB)")
    except Exception as e:
        print(f"  Note: failed to save duplicate model to {secondary_path}: {e}")
    
    return mlmodel


def _apply_rotation(frame: np.ndarray, rotate: int) -> np.ndarray:
    """Rotate frame by given degrees (allowed: -90, 0, 90, 180)."""
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == -90 or rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame



def validate_on_video(
    mlmodel: ct.models.MLModel,
    video_path: pathlib.Path,
    config: Dict[str, Any],
    max_frames: int = -1,
    stride: int = 1,
    rotate: int = 0,
    debug_dir: Optional[pathlib.Path] = None,
    crop_from: str = 'top',
) -> Dict[str, Any]:
    """Run the CoreML model on a sample of frames from the provided video.

    Cropping, resize, and ImageNet normalization are applied externally. The CoreML
    model expects an NCHW float32 tensor normalized as (x/255 - mean)/std.
    """
    print(f"\nValidating on video: {video_path}")
    if not video_path.exists():
        print(f"Warning: video not found: {video_path}")
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: failed to open video: {video_path}")
        return {}

    class_labels = ['no_action', 'action'] if config['num_classes'] == 2 else [f'class_{i}' for i in range(config['num_classes'])]
    label_to_idx = {l: i for i, l in enumerate(class_labels)}
    action_idx = label_to_idx.get('action', 1)

    total = 0
    action_count = 0
    avg_prob = 0.0
    printed = 0
    frame_idx = 0

    # Create debug directory if requested
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Determine output name once for efficient access
    try:
        out_name = mlmodel.user_defined_metadata.get("output_name")  # type: ignore[attr-defined]
    except Exception:
        out_name = None
    if not out_name:
        try:
            _spec = mlmodel.get_spec()
            _outs = [o.name for o in _spec.description.output]
            out_name = _outs[0] if len(_outs) > 0 else "probs"
        except Exception:
            out_name = "probs"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        # Fix orientation before cropping if needed
        proc = frame
        if rotate != 0:
            proc = _apply_rotation(proc, rotate)
        # If the current orientation makes crop() remove from the wrong side,
        # allow flipping vertically before/after crop to switch top/bottom semantics.
        flip_before_after = (crop_from.lower() == 'bottom')
        if flip_before_after:
            proc = cv2.flip(proc, 0)
        # External crop
        cropped = crop(proc)
        if flip_before_after:
            cropped = cv2.flip(cropped, 0)
        # BGR (OpenCV) -> RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # Optionally save a few cropped frames before resizing (for visual debug)
        if debug_dir is not None and printed < 50:
            Image.fromarray(rgb).save(debug_dir / f"debug_frame_{frame_idx:06d}.png")

        # Resize externally to model's expected size using cv2 for consistency
        rgb_resized = cv2.resize(
            rgb,
            (config['input_width'], config['input_height']),
            interpolation=cv2.INTER_LINEAR,
        )

        # Prepare CoreML input as raw pixels NCHW float32 in [0,255] (model handles normalization)
        cm_input = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1))[np.newaxis, :]

        preds = mlmodel.predict({"image": cm_input})
        probs_ml = np.array(preds[out_name], dtype=np.float32).reshape(-1)
        prob_action = float(probs_ml[action_idx])
        label = 'action' if prob_action >= 0.5 else 'no_action'

        action_count += int(label == 'action')
        avg_prob += prob_action
        total += 1
        if printed < 500:
            print(f"  frame {frame_idx:6d}: label={label}, p(action)={prob_action:.4f}")
            printed += 1
        frame_idx += 1
        if max_frames > 0 and total >= max_frames:
            break

    cap.release()
    if total > 0:
        avg_prob /= total
    print(f"Video summary: frames_eval={total}, action_frames={action_count}, mean_p_action={avg_prob:.4f}")
    return {"frames_eval": total, "action_frames": action_count, "mean_p_action": avg_prob}


def validate_torch_on_video(
    model: torch.nn.Module,
    video_path: pathlib.Path,
    config: Dict[str, Any],
    max_frames: int = -1,
    stride: int = 1,
    rotate: int = 0,
    debug_dir: Optional[pathlib.Path] = None,
    crop_from: str = 'top',
) -> Dict[str, Any]:
    """Run the PyTorch model (float) on the provided video with the same external crop and internal preprocessing.

    The provided model is expected to output softmax probabilities (as created in create_model()).
    """
    print(f"\nValidating PyTorch model on video: {video_path}")
    if not video_path.exists():
        print(f"Warning: video not found: {video_path}")
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: failed to open video: {video_path}")
        return {}

    class_labels = ['no_action', 'action'] if config['num_classes'] == 2 else [f'class_{i}' for i in range(config['num_classes'])]
    label_to_idx = {l: i for i, l in enumerate(class_labels)}
    action_idx = label_to_idx.get('action', 1)

    total = 0
    action_count = 0
    avg_prob = 0.0
    printed = 0
    frame_idx = 0

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            proc = frame
            if rotate != 0:
                proc = _apply_rotation(proc, rotate)
            flip_before_after = (crop_from.lower() == 'bottom')
            if flip_before_after:
                proc = cv2.flip(proc, 0)
            cropped = crop(proc)
            if flip_before_after:
                cropped = cv2.flip(cropped, 0)

            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            if debug_dir is not None and printed < 50:
                Image.fromarray(rgb).save(debug_dir / f"torch_debug_{frame_idx:06d}.png")

            # Resize externally to model's expected size
            rgb_resized = cv2.resize(rgb, (config['input_width'], config['input_height']), interpolation=cv2.INTER_LINEAR)
            # Prepare tensor for model: NCHW float; apply ImageNet normalization
            x = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float()
            x = x / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            prob_action = float(probs[0, action_idx].item())
            label = 'action' if prob_action >= 0.5 else 'no_action'

            action_count += int(label == 'action')
            avg_prob += prob_action
            total += 1
            if printed < 500:
                print(f"  [torch] frame {frame_idx:6d}: label={label}, p(action)={prob_action:.4f}")
                printed += 1
            frame_idx += 1
            if max_frames > 0 and total >= max_frames:
                break

    cap.release()
    if total > 0:
        avg_prob /= total
    print(f"PyTorch video summary: frames_eval={total}, action_frames={action_count}, mean_p_action={avg_prob:.4f}")
    return {"frames_eval": total, "action_frames": action_count, "mean_p_action": avg_prob}


def compare_on_video(
    mlmodel: ct.models.MLModel,
    model: torch.nn.Module,
    video_path: pathlib.Path,
    config: Dict[str, Any],
    max_frames: int = -1,
    stride: int = 1,
    rotate: int = 0,
    crop_from: str = 'top',
) -> Dict[str, Any]:
    """Run both CoreML and PyTorch models and report agreement and probability differences."""
    print(f"\nComparing CoreML vs PyTorch on: {video_path}")
    if not video_path.exists():
        print(f"Warning: video not found: {video_path}")
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: failed to open video: {video_path}")
        return {}

    class_labels = ['no_action', 'action'] if config['num_classes'] == 2 else [f'class_{i}' for i in range(config['num_classes'])]
    label_to_idx = {l: i for i, l in enumerate(class_labels)}
    action_idx = label_to_idx.get('action', 1)

    total = 0
    disagree = 0
    abs_diffs = []
    frame_idx = 0

    model.eval()
    # Determine output name once for efficient access
    try:
        out_name = mlmodel.user_defined_metadata.get("output_name")  # type: ignore[attr-defined]
    except Exception:
        out_name = None
    if not out_name:
        try:
            _spec = mlmodel.get_spec()
            _outs = [o.name for o in _spec.description.output]
            out_name = _outs[0] if len(_outs) > 0 else "probs"
        except Exception:
            out_name = "probs"
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            proc = frame
            if rotate != 0:
                proc = _apply_rotation(proc, rotate)
            flip_before_after = (crop_from.lower() == 'bottom')
            if flip_before_after:
                proc = cv2.flip(proc, 0)
            ## save images
            # save_path = pathlib.Path("debug_frames") / f"debug_frame_full_image_{frame_idx:06d}.png"
            # cv2.imwrite(str(save_path), proc)
            cropped = crop(proc)
            if flip_before_after:
                cropped = cv2.flip(cropped, 0)

            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Resize externally with cv2
            rgb_resized = cv2.resize(
                rgb,
                (config['input_width'], config['input_height']),
                interpolation=cv2.INTER_LINEAR,
            )

            # CoreML input: raw pixels NCHW float32 in [0,255] (model normalizes internally)
            cm_input = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1))[np.newaxis, :]
            
            # Debug: check tensor values for first frame
            if total == 0:
                print(f"    [CoreML tensor input] shape={cm_input.shape}, dtype={cm_input.dtype}, range=[{cm_input.min():.1f}, {cm_input.max():.1f}]")
                print(f"    [CoreML tensor sample] channel 0, pixel [0,0]={cm_input[0, 0, 0, 0]:.1f}, channel 1={cm_input[0, 1, 0, 0]:.1f}, channel 2={cm_input[0, 2, 0, 0]:.1f}")

            ml_preds = mlmodel.predict({"image": cm_input})
            probs_ml = np.array(ml_preds[out_name], dtype=np.float32).reshape(-1)
            prob_ml = float(probs_ml[action_idx])
            if total < 5:
                print(f"    [CoreML probs] no_action={probs_ml[0]:.4f}, action={probs_ml[1]:.4f}")
            label_ml = 'action' if prob_ml >= 0.5 else 'no_action'

            # PyTorch prediction: normalize explicitly
            x = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float()
            x = x / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            
            # Debug: check tensor values for first frame
            if total == 0:
                print(f"    [PyTorch tensor input] shape={x.shape}, dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")
                print(f"    [PyTorch tensor sample] channel 0, pixel [0,0]={x[0, 0, 0, 0]:.3f}, channel 1={x[0, 1, 0, 0]:.3f}, channel 2={x[0, 2, 0, 0]:.3f}")
                # Also test with same tensor on CoreML tensor_input from above to see if it's a model issue
                print(f"    [Testing PyTorch with CoreML's exact tensor...]")
                # For sanity: run PyTorch on normalized version of cm_input
                cm_norm = torch.from_numpy(cm_input).float() / 255.0
                cm_norm = (cm_norm - mean) / std
                test_logits = model(cm_norm)
                test_probs = torch.nn.functional.softmax(test_logits, dim=1)
                print(f"    [PyTorch with CoreML tensor] no_action={test_probs[0, 0].item():.4f}, action={test_probs[0, 1].item():.4f}")
            
            logits = model(x)
            torch_probs = torch.nn.functional.softmax(logits, dim=1)
            prob_torch = float(torch_probs[0, action_idx].item())
            label_torch = 'action' if prob_torch >= 0.5 else 'no_action'
            # Debug: print probabilities for first few frames
            if total < 5:
                print(f"    [PyTorch probs] no_action={torch_probs[0, 0].item():.4f}, action={torch_probs[0, 1].item():.4f}")
            print(f"  frame {frame_idx:6d}: CoreML label={label_ml}, p(action)={prob_ml:.4f} | PyTorch label={label_torch}, p(action)={prob_torch:.4f}")
            disagree += int(label_ml != label_torch)
            abs_diffs.append(abs(prob_ml - prob_torch))
            total += 1
            frame_idx += 1
            if max_frames > 0 and total >= max_frames:
                break

    cap.release()
    if total == 0:
        print("No frames evaluated; comparison aborted.")
        return {}
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    max_abs_diff = max(abs_diffs)
    disagree_rate = disagree / total
    print(f"Comparison summary: frames={total}, disagree={disagree} ({disagree_rate:.2%}), mean|Δp|={mean_abs_diff:.4f}, max|Δp|={max_abs_diff:.4f}")
    return {
        "frames": total,
        "disagree": disagree,
        "disagree_rate": disagree_rate,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
    }


def compare_on_image(
    mlmodel: ct.models.MLModel,
    model: torch.nn.Module,
    image_path: pathlib.Path,
    config: Dict[str, Any],
    apply_crop: bool = False,
) -> Dict[str, Any]:
    """Compare CoreML vs PyTorch on a single RGB image file.

    The image will be resized to the model input size (H, W). If you pass a 256x256
    PNG saved from the iOS app's DebugModelInputs, it will be used as-is (just converted to float).
    """
    print(f"\nComparing on image: {image_path}")
    if not image_path.exists():
        print(f"Warning: image not found: {image_path}")
        return {}

    input_height = config['input_height']
    input_width = config['input_width']

    # Load image as RGB, applying EXIF orientation if present
    pil = Image.open(image_path)
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    pil = pil.convert('RGB')
    if apply_crop:
        # Apply Python crop() semantics before resize
        np_img = np.array(pil)
        np_img = crop(np_img)
        pil = Image.fromarray(np_img)
    if pil.size != (input_width, input_height):
        pil = pil.resize((input_width, input_height), resample=Image.BILINEAR)
    arr = np.array(pil).astype(np.float32)

    # Prepare CoreML input: NCHW float32 [0,255]
    cm_input = np.transpose(arr, (2, 0, 1))[np.newaxis, :]

    # CoreML prediction
    try:
        preds = mlmodel.predict({"image": cm_input})
        # Resolve output name
        try:
            out_name = mlmodel.user_defined_metadata.get("output_name")  # type: ignore[attr-defined]
        except Exception:
            out_name = None
        if not out_name:
            try:
                _spec = mlmodel.get_spec()
                _outs = [o.name for o in _spec.description.output]
                out_name = _outs[0] if len(_outs) > 0 else next(iter(preds.keys()))
            except Exception:
                out_name = next(iter(preds.keys()))
        probs_ml = np.array(preds[out_name], dtype=np.float32).reshape(-1)
    except Exception as e:
        print(f"CoreML predict failed: {e}")
        return {}

    # PyTorch prediction (normalize here)
    with torch.no_grad():
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        logits = model(x)
        probs_torch = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy().reshape(-1)

    # Compare
    abs_diffs = np.abs(probs_ml - probs_torch)
    print(f"  CoreML probs:  {probs_ml}")
    print(f"  PyTorch probs: {probs_torch}")
    print(f"  mean|Δ|={abs_diffs.mean():.6f}, max|Δ|={abs_diffs.max():.6f}")
    return {
        "mean_abs_diff": float(abs_diffs.mean()),
        "max_abs_diff": float(abs_diffs.max()),
        "probs_coreml": probs_ml.tolist(),
        "probs_torch": probs_torch.tolist(),
    }


def validate_coreml_model(mlmodel: ct.models.MLModel, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the CoreML model with a test input"""
    print("\nValidating CoreML model...")
    
    import numpy as np
    
    input_height = config['input_height']
    input_width = config['input_width']
    
    # Create a random test RGB tensor (1, 3, H, W) float32 in [0, 255]
    test_tensor = (np.random.rand(1, 3, input_height, input_width) * 255).astype(np.float32)
    
    # Run prediction (CoreML tensor input expects NCHW float32 array)
    predictions = mlmodel.predict({"image": test_tensor})
    
    print("Validation successful!")
    print(f"Output keys: {list(predictions.keys())}")
    
    # Print details for each output
    for key, value in predictions.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch bowling classifier to CoreML format (.mlpackage) with float16 precision"
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
        help='Output path for CoreML model (.mlpackage). Default: same name as input with .mlpackage extension'
    )
    
    # Explicit precision flags
    prec_group = parser.add_mutually_exclusive_group()
    prec_group.add_argument(
        '--fp16',
        action='store_true',
        help='Export with float16 precision (default)'
    )
    prec_group.add_argument(
        '--fp32',
        action='store_true',
        help='Export with float32 precision'
    )
    parser.add_argument(
        '--scriptable',
        action='store_true',
        help='Build export-friendly (scriptable) variant of the timm model when possible'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['mlprogram', 'neuralnetwork'],
        default='mlprogram',
        help='CoreML backend to target (default: mlprogram)'
    )
    
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of the exported model'
    )

    parser.add_argument(
        '--video',
        type=pathlib.Path,
        default=pathlib.Path('line&length/line&length/IMG_8301.MOV'),
        help='Optional: validate exported CoreML model on this video (cropping external)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='Number of frames to evaluate from the video; -1 means evaluate all frames'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Evaluate every Nth frame; 1 means evaluate every frame'
    )
    parser.add_argument(
        '--rotate',
        type=int,
        choices=[-90, 0, 90, 180, 270],
        default=0,
        help='Rotate each frame before cropping to fix orientation (degrees)'
    )
    parser.add_argument(
        '--save-debug-frames',
        type=pathlib.Path,
        default=None,
        help='Optional directory to save a few cropped RGB frames for debugging'
    )
    parser.add_argument(
        '--coreml',
        action='store_true',
        help='Validate the exported CoreML model on the video'
    )
    parser.add_argument(
        '--validate-torch',
        action='store_true',
        help='Also validate the original PyTorch model on the video'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare CoreML vs PyTorch predictions on the video and report differences'
    )
    parser.add_argument(
        '--image',
        type=pathlib.Path,
        default=None,
        help='Optional: path to an RGB image to compare (PyTorch vs CoreML). If size differs, it will be resized to model input.'
    )
    parser.add_argument(
        '--apply-crop',
        action='store_true',
        help='Apply the same Python crop (remove top 32% then centered square) before resizing the image to model input size.'
    )
    parser.add_argument(
        '--crop-from',
        type=str,
        choices=['top', 'bottom'],
        default='top',
        help="Which side to treat as 'top' for the crop operation (default: top)"
    )
    
    args = parser.parse_args()
    
    # Print environment for debugging (helps diagnose NumPy/ABI issues)
    print_env_info()
    
    # Check if input file exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        output_path = args.model_path.with_suffix('.mlpackage')
    else:
        output_path = args.output
        if output_path.suffix != '.mlpackage':
            output_path = output_path.with_suffix('.mlpackage')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load checkpoint
        checkpoint, config = load_checkpoint(args.model_path)
        
        # Create model
        model = create_model(checkpoint, config, scriptable=args.scriptable)
        
        # Resolve precision: default fp16 unless --fp32 specified
        use_float16 = True if (args.fp16 or not args.fp32) else False

        # Export to CoreML with explicit backend and precision
        mlmodel = export_to_coreml(
            model,
            config,
            output_path,
            use_float16,
            scriptable=args.scriptable,
            backend=args.backend,
        )

        if not args.no_validate:
            validate_coreml_model(mlmodel, config)
            if args.video is not None:
                try:
                    if args.coreml:
                        validate_on_video(
                            mlmodel,
                            args.video,
                            config,
                            max_frames=args.max_frames,
                            stride=args.stride,
                            rotate=args.rotate,
                            debug_dir=args.save_debug_frames,
                            crop_from=args.crop_from,
                        )
                    if args.validate_torch:
                        validate_torch_on_video(
                            model,
                            args.video,
                            config,
                            max_frames=args.max_frames,
                            stride=args.stride,
                            rotate=args.rotate,
                            debug_dir=args.save_debug_frames,
                            crop_from=args.crop_from,
                        )
                    if args.compare:
                        compare_on_video(
                            mlmodel,
                            model,
                            args.video,
                            config,
                            max_frames=args.max_frames,
                            stride=args.stride,
                            rotate=args.rotate,
                            crop_from=args.crop_from,
                        )
                except Exception as ve:
                    print(f"Warning: video validation failed: {ve}")

        # Optional single-image parity check
        if args.image is not None:
            try:
                compare_on_image(mlmodel, model, args.image, config, apply_crop=args.apply_crop)
            except Exception as ie:
                print(f"Warning: image comparison failed: {ie}")

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
