# CoreML Export Guide

This guide explains how to export your trained PyTorch classifier to Core ML, validate it on sample videos, and choose explicit backends/precisions for predictable deployment.

## What gets exported

- A Core ML model that:
  - Expects RGB input as NCHW float32 with pixel values in [0, 255]
  - Applies ImageNet normalization internally: x = (x/255 - mean)/std with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Outputs class probabilities (softmax)
- The exported model is produced by tracing a wrapper around your trained model that performs normalization and softmax.

## Requirements

- Python 3.9+
- PyTorch, timm
- coremltools (tested with 6.x)
- OpenCV, Pillow, NumPy
- macOS with Xcode for on-device testing (optional)

Install (minimal):

```bash
pip install torch torchvision timm coremltools opencv-python pillow numpy
```

## CLI

Script: `export_coreml.py`

Required:
- `model_path`: path to the PyTorch checkpoint (.pth) that was saved by training

Common flags:
- `--output`: output path (.mlpackage). Default is `<model_path>.mlpackage`.
- `--backend {mlprogram, neuralnetwork}`: select conversion backend (default: mlprogram)
- `--fp16` or `--fp32`: explicit precision (default: fp16)
- `--scriptable`: try building a scriptable timm variant before loading weights (optional)
- `--no-validate`: skip post-export validations

Validation and debugging flags:
- `--video PATH`: run validation/comparison on this video
- `--coreml`: run Core ML-only validation on video
- `--validate-torch`: run PyTorch model on video
- `--compare`: run both and compare probabilities frame-by-frame
- `--max-frames N`: evaluate only the first N frames
- `--stride N`: evaluate every Nth frame
- `--rotate { -90,0,90,180,270 }`: fix orientation before cropping
- `--save-debug-frames DIR`: save a few cropped frames
- `--crop-from {top,bottom}`: flip before/after crop if you need the crop origin reversed

## Examples

Export MLProgram (fp16, default) and compare on 10 frames:

```bash
python export_coreml.py trainings/.../best_model.pth --backend mlprogram --fp16 --compare --max-frames 10
```

Export MLProgram (fp32):

```bash
python export_coreml.py trainings/.../best_model.pth --backend mlprogram --fp32
```

Export NeuralNetwork backend (fp16 weights compressed after conversion):

```bash
python export_coreml.py trainings/.../best_model.pth --backend neuralnetwork --fp16 --compare
```

Validate Core ML only on a specific video:

```bash
python export_coreml.py trainings/.../best_model.pth --coreml --video line&length/line&length/IMG_8301.MOV --max-frames 50
```

## Input/Output contract (Core ML)

- Input key: `image`
- Type: NCHW float32 (1, 3, H, W)
- Range: [0, 255] per pixel (the model normalizes internally)
- Output key: `probs`
- Output type: class probabilities (softmax) as a dense array ordered by class index

Note: The exporter standardizes the output name to `probs` for a simpler API. In Python: `preds = mlmodel.predict({"image": x}); probs = preds["probs"]`.

## Backend and precision notes

- MLProgram
  - Requires newer OS targets (iOS 15+/macOS 12+, commonly iOS 16+).
  - Supports `compute_precision` directly (fp16/fp32).
  - Often yields smaller models and can be faster on modern devices.
- NeuralNetwork
  - Supports older targets (down to iOS 14/macOS 11 in this exporter).
  - Ignores `compute_precision` at conversion time; this script compresses weights to fp16 after conversion when `--fp16` is set.

## On-device integration tips

- The exporter uses a tensor input in the Core ML spec. In Python, call `predict` with an NCHW float32 array in [0,255] under the key `image`.
- In iOS/macOS apps, prefer passing an `MLMultiArray` (NCHW) at the `image` input. If you use `CVPixelBuffer`, convert to the expected layout and dtype before inference.

## Troubleshooting

- Large parity differences with MLProgram fp32
  - Try `--fp16` with MLProgram or use the NeuralNetwork backend.
- NN backend and fp16 input complaints
  - NN with older targets may not accept fp16 input types; this exporter keeps fp32 input but compresses weights to fp16 after conversion when `--fp16` is specified.
- Editor warnings like "Import X could not be resolved"
  - Those are environment (Pylance) messages; ensure your Python env has torch, coremltools, numpy, cv2, PIL installed.
- Preprocessing mismatches
  - The exported model performs normalization internally. Feed raw RGB pixels in [0,255], NCHW.

## Reproducibility

- The script prints environment/version info at start to help diagnose ABI issues (e.g., NumPy versions).
- For consistent comparisons, the video compare path now resizes with OpenCV and matches exactly the model’s normalization math.
