# Bowling Action Labelling and Classification

This project provides tools for labelling bowling actions in videos and training a deep learning classifier to automatically detect bowling actions. The system is designed to work with video files and their corresponding JSON annotations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Video Labelling](#video-labelling)
  - [Training Classifier](#training-classifier)
  - [Additional Tools](#additional-tools)
  - [CoreML Export](#coreml-export)
- [Project Files](#project-files)
- [Output](#output)
- [Requirements](#requirements)

## Overview

This project consists of two main components:

1. **Video Labelling Tool** (`video_labeler.py`): An interactive OpenCV-based tool for manually labelling bowling actions in videos
2. **Classifier Training** (`train_classifier.py`): A PyTorch-based training pipeline for learning to classify bowling actions from image crops

The workflow typically involves:
1. Recording bowling videos (.MOV files)
2. Using the video labeller to mark temporal segments containing bowling actions
3. Extracting image crops from the labelled segments
4. Training a deep learning model to classify these crops automatically

## Installation

### Quick start

```bash
pip install -r requirements.txt
```

This installs all runtime and tooling dependencies defined in `requirements.txt`.

### Prerequisites

- Python 3.7+
- OpenCV (cv2)
- PyTorch
- timm (PyTorch Image Models)
- scikit-learn
- matplotlib
- seaborn
- tensorboard
- PIL/Pillow
- pandas
- numpy
 - coremltools (for Core ML export)
 - altair (optional, for visualization utilities)

### Install Dependencies

```bash
# Install PyTorch (visit pytorch.org for system-specific installation)
pip install torch torchvision torchaudio

# Install other dependencies
# Or install packages individually
pip install opencv-python timm scikit-learn matplotlib seaborn tensorboard pillow pandas numpy coremltools altair
```

## Dataset Structure

The project expects the following directory structure:

```
bowling-labelling/
├── videos-aug31st/          # Video files and annotations
│   ├── IMG_8286.MOV        # Video file
│   ├── IMG_8286.json       # Corresponding annotation file
│   └── ...
├── videos-sept7th/         # Additional video directory
│   └── ...
├── crops_dataset/          # Generated crop images (created by extract_crops.py)
│   ├── IMG_8286/          # Crops from specific video
│   └── labels.txt         # Labels file for training
├── trainings/             # Training outputs (created by train_classifier.py)
│   ├── logs/              # TensorBoard logs
│   ├── checkpoints/       # Model checkpoints
│   └── results/           # Training results and plots
└── classes.txt            # Action class definitions
```

## Usage

### Video Labelling

The `video_labeler.py` tool provides an interactive interface for labelling bowling actions in videos.

#### Basic Usage

```bash
python video_labeler.py --video_dir videos-sept7th --action_class bowling
```

#### Parameters

- `--video_dir`: Directory containing video files (default: "videos-sept7th")
- `--action_class`: Action class name (default: "bowling", choices: ["bowling"])
- `--box_class`: Bounding box class name (default: "bowler", choices: ["bowler"])

#### Interactive Controls

When running the video labeller:

- **Arrow Keys**: Navigate frames (←/→ for single frame, ↑/↓ for 10 frames)
- **Space**: Start/End action instance marking
- **Enter**: Confirm and save current instance
- **'s'**: Save annotations
- **'q'**: Quit and move to next video
- **'r'**: Reset current instance
- **'d'**: Delete last instance

#### Annotation Format

The tool saves annotations in JSON format:

```json
{
  "actions": [
    {
      "class": "bowling",
      "start_frame": 45,
      "end_frame": 120
    }
  ]
}
```

### Training Classifier

The `train_classifier.py` script trains a deep learning model to classify bowling actions from image crops.

#### Basic Usage

```bash
python train_classifier.py --data_dir crops_dataset --labels_file crops_dataset/labels.txt --output_dir trainings
```

#### Key Parameters

**Data & Model:**
- `--data_dir`: Dataset directory (default: "crops_dataset")
- `--labels_file`: Labels file path (default: "crops_dataset/labels.txt")
- `--output_dir`: Output directory for results (default: "trainings")
- `--model_name`: timm model name (default: "efficientnet_b0")
- `--num_classes`: Number of classes (default: 2)

**Training:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--optimizer`: Optimizer type (default: "adamw", choices: ["adamw", "adam", "sgd"])
- `--scheduler`: Learning rate scheduler (default: "cosine", choices: ["plateau", "cosine"])

**Data Handling:**
- `--val_videos`: Number of videos for validation (default: 12)
- `--input_height`: Input image height (default: 320)
- `--input_width`: Input image width (default: 128)
- `--augment`: Enable data augmentation (enabled by default)

**Class Balancing:**
- `--class_balance`: Class balancing method (default: "weights", choices: ["none", "weights", "sampling", "focal"])
- `--focal_alpha`: Focal loss alpha parameter (default: 0.25)
- `--focal_gamma`: Focal loss gamma parameter (default: 2.0)

#### Advanced Usage Examples

**Train with focal loss for class imbalance:**
```bash
python train_classifier.py --class_balance focal --focal_alpha 0.25 --focal_gamma 2.0 --epochs 100
```

**Train with different model architecture:**
```bash
python train_classifier.py --model_name resnet50 --batch_size 32 --lr 2e-4
```

**Resume training from checkpoint:**
```bash
python train_classifier.py --checkpoint trainings/checkpoints/best_model.pth
```

**Validation only mode:**
```bash
python train_classifier.py --validate_only --checkpoint trainings/checkpoints/best_model.pth
```

### Additional Tools

#### Extract Crops (`extract_crops.py`)

Extracts left/right image crops from labelled video segments:

```bash
python extract_crops.py --input_dir videos-sept7th --output_dir crops_dataset
```

#### Convert Annotations (`convert_annotations.py`)

Converts between different annotation formats if needed.

#### Analyze Labels (`analyze_labels.py`)

Provides statistics and analysis of the labelled dataset.

## Project Files

- **`video_labeler.py`**: Interactive video labelling tool
- **`train_classifier.py`**: Deep learning model training pipeline
- **`extract_crops.py`**: Image crop extraction from videos
- **`convert_annotations.py`**: Annotation format conversion utilities
- **`analyze_labels.py`**: Dataset analysis and statistics
- **`classes.txt`**: Defines action classes (currently: "bowling")
- **`run_train.sh`**: Bash script for automated training runs

### CoreML Export

Export trained PyTorch checkpoints to Core ML for on-device use and validate on sample videos. See the dedicated guide:

- CoreML Export Guide: [docs/export_coreml.md](docs/export_coreml.md)

## Output

### Training Outputs

The training script produces:

- **Model checkpoints**: Best and latest model weights
- **Training logs**: TensorBoard logs for monitoring
- **Metrics plots**: Training/validation curves, confusion matrices
- **Results summary**: Final performance metrics and analysis

### TensorBoard Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir trainings/logs
```

### Model Performance

The trained model outputs:
- Classification accuracy
- Precision, recall, and F1-score
- Confusion matrix
- Training/validation loss curves

## Requirements

The project requires the following Python packages (see `requirements.txt` for the authoritative list and versions provided by pip):

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
timm>=0.6.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
tensorboard>=2.7.0
pillow>=8.0.0
pandas>=1.3.0
numpy>=1.21.0

Additional:

```
coremltools
altair
pytest
```
```

## Tips for Best Results

1. **Consistent Labelling**: Ensure consistent labelling across all videos for better model performance
2. **Balanced Dataset**: Try to have roughly equal amounts of positive and negative examples
3. **Video Quality**: Higher quality videos generally produce better training data
4. **Augmentation**: Enable data augmentation for better generalization
5. **Model Selection**: Experiment with different model architectures using the `--model_name` parameter
6. **Monitoring**: Use TensorBoard to monitor training progress and detect overfitting

## Troubleshooting

- **OpenCV Display Issues**: Ensure you have a display available when running the video labeller
- **Memory Issues**: Reduce batch size if encountering out-of-memory errors
- **Missing Files**: Check that video files and corresponding JSON annotations exist
- **Permission Errors**: Ensure write permissions for output directories

For more specific issues, check the error messages and ensure all dependencies are properly installed.