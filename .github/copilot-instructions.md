# Bowling Action Labelling & Classification - AI Agent Instructions

This is a computer vision pipeline for bowling action detection with manual annotation tools and automated classification.

## Architecture Overview

**3-Stage Pipeline:**
1. **Manual Annotation** (`video_labeler.py`) → Creates JSON files with temporal segments
2. **Crop Extraction** (`extract_crops.py`) → Generates left/right image crops with binary labels  
3. **Model Training** (`train_classifier.py`) → Trains PyTorch classifier on crops

**Data Flow:**
```
Videos (.MOV) + JSON annotations → Crop images + labels.txt → Trained PyTorch model
```

## Key Patterns & Conventions

### Annotation Format Evolution
- **Legacy format**: Single `start_frame`/`end_frame` per video
- **Current format**: `temporal_events` array supporting multiple action instances
- Use `convert_annotations.py` when migrating between formats
- JSON files always match video names: `IMG_8286.MOV` ↔ `IMG_8286.json`

### Video-Based Data Splits
- Training split by video, not individual frames (prevents data leakage)
- Last N videos become validation set (default: 12 videos)
- Pattern: `video_name = img_path.split('/')[0]` extracts video ID from crop paths

### Crop Processing Pipeline
- **Fixed crop strategy**: Remove top 570px, keep 1350px height, split left/right halves
- **Label generation**: Binary classification per crop (0=no action/wrong direction, 1=action present)
- **Output format**: `<relative_path>,<binary_label>,<side>` in `labels.txt`

### Model Training Specifics
- Uses `timm` models with pretrained weights (default: `efficientnet_b0`)
- **Class imbalance handling**: 4 strategies via `--class_balance` (weights/sampling/focal/none)
- **Input shape**: 128×320 (width×height) to match crop aspect ratio
- **Augmentation**: ColorJitter + horizontal flip enabled by default

## Critical Workflows

### Interactive Video Labelling
```bash
python video_labeler.py --video_dir videos-sept7th --action_class bowling
```
**Navigation**: Arrow keys (1 frame), W/X (10 frames), S/E (mark start/end), N (next instance)

### Full Pipeline Execution
```bash
# 1. Label videos interactively
python video_labeler.py --video_dir videos-sept7th

# 2. Extract crops from annotations  
python extract_crops.py --video_dir videos-sept7th --output_dir crops_dataset

# 3. Train classifier
python train_classifier.py --data_dir crops_dataset --epochs 50
```

### Training Monitoring
```bash
tensorboard --logdir trainings/logs
```

## Project-Specific Quirks

### Frame Coordinate System
- Video labeller uses 0-based indexing internally
- Display shows 1-based frame numbers to users
- Bounding boxes include top offset compensation (`top_offset = 100`)

### File Naming Conventions
- Videos: `IMG_XXXX.MOV` format expected
- Crops: `{video_name}_frame_{frame:06d}_{side}.jpg`
- Models saved to: `trainings/checkpoints/best_model.pth`

### Error Handling Patterns
- Missing annotation files → skip video with warning
- Corrupt images → fallback to black 128×320 image
- Invalid frame ranges → print error but continue processing

## Integration Points

- **OpenCV**: Real-time video display and frame extraction
- **PyTorch + timm**: Model architecture and training
- **TensorBoard**: Training visualization and metrics
- **sklearn**: Train/val splits and evaluation metrics

## Development Commands

```bash
# Quick training test
bash run_train.sh

# Resume from checkpoint
python train_classifier.py --checkpoint trainings/checkpoints/best_model.pth

# Validation only mode
python train_classifier.py --validate_only --checkpoint path/to/model.pth

# Convert old annotation format
python convert_annotations.py --video_dir videos-aug31st --backup
```