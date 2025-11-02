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
from PIL import Image
from modellib import crop


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
    nn_fallback_threshold: float = 1e-2,
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
        sample_tensor_raw = np.transpose(sample_array, (2, 0, 1))[np.newaxis, :]  # HWC -> NCHW
        
        # Apply preprocessing (ImageNet normalization) to match what will be fed to CoreML
        sample_tensor = torch.from_numpy(sample_tensor_raw)
        sample_tensor = sample_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        example_input = (sample_tensor - mean) / std
        
        print(f"  Sample input shape: {example_input.shape}, range: [{example_input.min():.3f}, {example_input.max():.3f}]")
    else:
        print(f"Real sample not found, using random data for tracing")
        # Create example input with ImageNet normalization applied
        raw_input = torch.rand(1, 3, input_height, input_width) * 255.0
        raw_input = raw_input / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        example_input = (raw_input - mean) / std
    
    # Prepare JIT candidates: script (if requested) and trace; choose the one closest to original
    print("Preparing JIT candidates...")
    model.eval()
    jit_candidates = []
    with torch.no_grad():
        ref_out = model(example_input)
    
    scripted_model = None
    if scriptable:
        try:
            print("  Attempting torch.jit.script...")
            scripted_model = torch.jit.script(model)
            with torch.no_grad():
                s_out = scripted_model(example_input)
                s_diff = torch.max(torch.abs(ref_out - s_out)).item()
            print(f"  Parity (script vs original): max diff = {s_diff:.6f}")
            jit_candidates.append(("script", scripted_model, s_diff))
        except Exception as e:
            print(f"  Scripting failed: {e}")

    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input, strict=False, check_trace=True)
        t_out = traced_model(example_input)
        t_diff = torch.max(torch.abs(ref_out - t_out)).item()
    print(f"  Parity (trace vs original):  max diff = {t_diff:.6f}")
    jit_candidates.append(("trace", traced_model, t_diff))

    # Choose best JIT based on smallest parity diff
    jit_candidates.sort(key=lambda x: x[2])
    chosen_name, chosen_jit, chosen_diff = jit_candidates[0]
    print(f"  Selected JIT: {chosen_name} (max diff {chosen_diff:.6f})")
    
    ## print traced model
    # print("Traced model structure:")
    # print(traced_model)
    print("Converting JIT model to CoreML...")
    convert_kwargs = {
        "convert_to": "mlprogram",
        "inputs": [ct.TensorType(name="image", shape=tuple(example_input.shape))],
    }
    # Prefer a modern target to get better operator coverage
    try:
        target = getattr(ct.target, "iOS16", None) or getattr(ct.target, "macOS13", None)
        if target is not None:
            convert_kwargs["minimum_deployment_target"] = target
    except Exception:
        pass

    precision = getattr(ct, "precision", None)
    if precision is not None:
        convert_kwargs["compute_precision"] = precision.FLOAT16 if use_float16 else precision.FLOAT32
        exported_precision = "float16" if use_float16 else "float32"
    else:
        exported_precision = "float32"
        if use_float16:
            # Older coremltools versions fall back to float32; warn user to reduce precision manually later.
            print("  Warning: coremltools.precision not available; exporting in float32.")

    mlmodel = ct.convert(chosen_jit, **convert_kwargs)
    print("CoreML MLProgram conversion complete")
    # Quick parity check: compare raw logits on the example input
    try:
        with torch.no_grad():
            torch_logits = model(example_input).cpu().numpy().reshape(-1)
        coreml_out = mlmodel.predict({"image": example_input.cpu().numpy().astype(np.float32)})
        # Try to find first array-like output
        cm_values = None
        for v in coreml_out.values():
            try:
                arr = np.array(v, dtype=np.float32).reshape(-1)
                if arr.size == torch_logits.size:
                    cm_values = arr
                    break
            except Exception:
                continue
        if cm_values is not None:
            diff = float(np.max(np.abs(cm_values - torch_logits)))
            print(f"  MLProgram parity: max|Δlogits|={diff:.6f}")
            # If MLProgram parity is not great, try alternate MLProgram precision before NN fallback
            if diff > nn_fallback_threshold and precision is not None and "compute_precision" in convert_kwargs:
                try:
                    alt_precision = precision.FLOAT32 if convert_kwargs["compute_precision"] == precision.FLOAT16 else precision.FLOAT16
                    alt_kwargs = dict(convert_kwargs)
                    alt_kwargs["compute_precision"] = alt_precision
                    mlmodel_alt = ct.convert(chosen_jit, **alt_kwargs)
                    coreml_out_alt = mlmodel_alt.predict({"image": example_input.cpu().numpy().astype(np.float32)})
                    cm_values_alt = None
                    for v in coreml_out_alt.values():
                        try:
                            arr_alt = np.array(v, dtype=np.float32).reshape(-1)
                            if arr_alt.size == torch_logits.size:
                                cm_values_alt = arr_alt
                                break
                        except Exception:
                            continue
                    if cm_values_alt is not None:
                        diff_alt = float(np.max(np.abs(cm_values_alt - torch_logits)))
                        human_prec = "float32" if alt_precision == precision.FLOAT32 else "float16"
                        print(f"  MLProgram (alt {human_prec}) parity: max|Δlogits|={diff_alt:.6f}")
                        if diff_alt <= diff:
                            print("  Using MLProgram alt-precision model due to better parity.")
                            mlmodel = mlmodel_alt
                            exported_precision = human_prec
                            diff = diff_alt
                except Exception as e_alt:
                    print(f"  Skipping MLProgram alt-precision attempt due to: {e_alt}")
            if diff > nn_fallback_threshold:
                print(
                    f"  MLProgram parity exceeds threshold ({diff:.6f} > {nn_fallback_threshold:g}); trying neuralnetwork backend..."
                )
                # Try neuralnetwork convert
                nn_kwargs = dict(convert_kwargs)
                nn_kwargs["convert_to"] = "neuralnetwork"
                # compute_precision is not supported for neuralnetwork target
                nn_kwargs.pop("compute_precision", None)
                # For neuralnetwork backend, minimum deployment target must be <= iOS14/macOS11
                try:
                    older_target = (
                        getattr(ct.target, "iOS14", None)
                        or getattr(ct.target, "iOS13", None)
                        or getattr(ct.target, "macOS11", None)
                        or getattr(ct.target, "macOS10_15", None)
                    )
                    if older_target is not None:
                        nn_kwargs["minimum_deployment_target"] = older_target
                        print(f"  Using older deployment target for NN: {older_target}")
                except Exception:
                    pass
                try:
                    mlmodel_nn = ct.convert(chosen_jit, **nn_kwargs)
                    coreml_out_nn = mlmodel_nn.predict({"image": example_input.cpu().numpy().astype(np.float32)})
                    cm_values_nn = None
                    for v in coreml_out_nn.values():
                        try:
                            arr2 = np.array(v, dtype=np.float32).reshape(-1)
                            if arr2.size == torch_logits.size:
                                cm_values_nn = arr2
                                break
                        except Exception:
                            continue
                    if cm_values_nn is not None:
                        diff_nn = float(np.max(np.abs(cm_values_nn - torch_logits)))
                        print(f"  NeuralNetwork parity: max|Δlogits|={diff_nn:.6f}")
                        # Optional: compress NN weights to FP16 to reduce size when requested
                        if use_float16 and hasattr(ct, "utils") and hasattr(ct.utils, "convert_neural_network_weights_to_fp16"):
                            try:
                                mlmodel_nn_fp16 = ct.utils.convert_neural_network_weights_to_fp16(mlmodel_nn)
                                coreml_out_nn_fp16 = mlmodel_nn_fp16.predict({"image": example_input.cpu().numpy().astype(np.float32)})
                                cm_values_nn_fp16 = None
                                for v in coreml_out_nn_fp16.values():
                                    try:
                                        arr3 = np.array(v, dtype=np.float32).reshape(-1)
                                        if arr3.size == torch_logits.size:
                                            cm_values_nn_fp16 = arr3
                                            break
                                    except Exception:
                                        continue
                                if cm_values_nn_fp16 is not None:
                                    diff_nn_fp16 = float(np.max(np.abs(cm_values_nn_fp16 - torch_logits)))
                                    print(f"  NeuralNetwork FP16-weights parity: max|Δlogits|={diff_nn_fp16:.6f}")
                                    # Keep FP16 weights if parity is as good as (or nearly as good as) float32
                                    if diff_nn_fp16 <= diff_nn + 1e-4:
                                        mlmodel_nn = mlmodel_nn_fp16
                                        exported_precision = "float16"
                            except Exception as e_fp16:
                                print(f"  Skipping NN FP16 weight conversion due to: {e_fp16}")
                        if diff_nn <= diff:
                            print("  Using NeuralNetwork model due to better parity.")
                            mlmodel = mlmodel_nn
                        else:
                            print("  Keeping MLProgram model (parity better than NN).")
                except Exception as e2:
                    print(f"  NeuralNetwork conversion failed: {e2}")
        else:
            print("  Note: could not locate a comparable array output for parity check.")
    except Exception as e:
        print(f"  Skipping parity check due to: {e}")
    
    # Add metadata
    mlmodel.author = "Bowling Action Classifier"
    mlmodel.short_description = f"Bowling action detection model ({config['model_name']})"
    mlmodel.version = "1.0"
    
    # Add input/output descriptions
    mlmodel.input_description["image"] = (
        "RGB image input (cropped externally), shape (1,3,H,W) float32 in ImageNet-normalized space: "
        "x = (x/255 - mean)/std with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"
    )
    
    # Add descriptions for classifier outputs (classLabel and class probabilities)
    spec = mlmodel.get_spec()
    output_names = [output.name for output in spec.description.output]
    print(f"Output feature names: {output_names}")
    
    for output_name in output_names:
        if output_name in mlmodel.output_description:
            if "classLabel" in output_name or "Label" in output_name:
                mlmodel.output_description[output_name] = "Predicted class label"
            elif "Probability" in output_name or "probability" in output_name:
                mlmodel.output_description[output_name] = "Class probabilities dictionary"
    
    # Save the model
    print(f"\nSaving CoreML model to: {output_path}")
    mlmodel.save(str(output_path))
    
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


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_max = np.max(x)
    ex = np.exp(x - x_max)
    s = np.sum(ex)
    return ex / (s if s != 0 else 1.0)


def _extract_probs_from_coreml_preds(preds: Dict[str, Any], num_classes: int) -> Optional[np.ndarray]:
    """Best-effort extraction of class probabilities from CoreML prediction dict.

    Handles these cases:
    - preds contains a dict of probabilities (label->prob).
    - preds contains an array-like of probabilities.
    - preds contains raw logits (most likely for MLProgram from traced PyTorch) -> apply softmax.
    Returns None if nothing usable found.
    """
    # 1) Look for a probability dict
    for k, v in preds.items():
        lk = k.lower()
        if "prob" in lk or "classprob" in lk:
            if isinstance(v, dict) and len(v) > 0:
                arr = np.array(list(v.values()), dtype=np.float32)
                # If not normalized, normalize defensively
                s = np.sum(arr)
                if s <= 0 or s > 1.0001:
                    arr = _softmax_np(arr)
                return arr
            # array-like already
            try:
                arr = np.array(v, dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    s = np.sum(arr)
                    if s <= 0 or s > 1.0001:
                        arr = _softmax_np(arr)
                    return arr
            except Exception:
                pass

    # 2) No explicit probability key; pick the first array-like output as logits
    for k, v in preds.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.array(v, dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    # Treat as logits and softmax
                    return _softmax_np(arr)
            except Exception:
                continue

    # 3) Sometimes CoreML returns {name: MLMultiArray}; convert via np.array()
    for k, v in preds.items():
        try:
            arr = np.array(v, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return _softmax_np(arr)
        except Exception:
            continue

    return None


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

        # Prepare tensor (NCHW) and apply ImageNet normalization expected by the CoreML model
        tensor_input = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1))[np.newaxis, :]
        tensor_input = tensor_input / 255.0
        mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        tensor_input = (tensor_input - mean_np) / std_np

        preds = mlmodel.predict({"image": tensor_input})
        probs_ml = _extract_probs_from_coreml_preds(preds, len(class_labels))
        if probs_ml is None:
            prob_action = 0.0
            label = 'no_action'
        else:
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

    import math
    total = 0
    disagree = 0
    abs_diffs = []
    frame_idx = 0

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

            # Resize externally with cv2 for both CoreML and PyTorch paths to be identical
            rgb_resized = cv2.resize(
                rgb,
                (config['input_width'], config['input_height']),
                interpolation=cv2.INTER_LINEAR,
            )

            # Build a single normalized tensor to feed both CoreML and PyTorch
            tensor_input = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1))[np.newaxis, :]  # HWC -> NCHW
            tensor_input = tensor_input / 255.0
            mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
            std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            tensor_input = (tensor_input - mean_np) / std_np
            
            # Debug: check tensor values for first frame
            if total == 0:
                print(f"    [CoreML tensor input] shape={tensor_input.shape}, dtype={tensor_input.dtype}, range=[{tensor_input.min():.3f}, {tensor_input.max():.3f}]")
                print(f"    [CoreML tensor sample] channel 0, pixel [0,0]={tensor_input[0, 0, 0, 0]:.3f}, channel 1={tensor_input[0, 1, 0, 0]:.3f}, channel 2={tensor_input[0, 2, 0, 0]:.3f}")
            
            ml_preds = mlmodel.predict({"image": tensor_input})
            probs_ml = _extract_probs_from_coreml_preds(ml_preds, len(class_labels))
            if probs_ml is None:
                prob_ml = 0.0
            else:
                prob_ml = float(probs_ml[action_idx])
                if total < 5:
                    print(f"    [CoreML probs] no_action={probs_ml[0]:.4f}, action={probs_ml[1]:.4f}")
            label_ml = 'action' if prob_ml >= 0.5 else 'no_action'

            # PyTorch prediction: reuse the exact same normalized tensor
            x = torch.from_numpy(tensor_input.copy())
            
            # Debug: check tensor values for first frame
            if total == 0:
                print(f"    [PyTorch tensor input] shape={x.shape}, dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")
                print(f"    [PyTorch tensor sample] channel 0, pixel [0,0]={x[0, 0, 0, 0]:.3f}, channel 1={x[0, 1, 0, 0]:.3f}, channel 2={x[0, 2, 0, 0]:.3f}")
                # Also test with same tensor on CoreML tensor_input from above to see if it's a model issue
                print(f"    [Testing PyTorch with CoreML's exact tensor...]")
                test_logits = model(torch.from_numpy(tensor_input))
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
    
    parser.add_argument(
        '--float32',
        action='store_true',
        help='Use float32 precision instead of float16 (larger model size)'
    )
    parser.add_argument(
        '--scriptable',
        action='store_true',
        help='Build export-friendly (scriptable) variant of the timm model when possible'
    )
    parser.add_argument(
        '--nn-fallback-threshold',
        type=float,
        default=1e-2,
        help='Only attempt NeuralNetwork backend if MLProgram max|Δlogits| exceeds this threshold (default: 1e-2)'
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
        
        # Export to CoreML
        use_float16 = not args.float32
        mlmodel = export_to_coreml(
            model,
            config,
            output_path,
            use_float16,
            scriptable=args.scriptable,
            nn_fallback_threshold=args.nn_fallback_threshold,
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
