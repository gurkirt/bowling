#!/usr/bin/env python3
"""
train_classifier.py - Train a PyTorch image classification model using timm
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BowlingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, root_dir=""):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback with rectangular dimensions
            image = Image.new('RGB', (128, 320), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def load_dataset(labels_file, data_dir, val_videos=12, random_state=42):
    """Load and split the dataset based on videos"""
    print(f"Loading dataset from {labels_file}")
    
    # Read labels file and group by video
    video_data = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    img_path, action_label, side = parts
                    # Extract video name from path (e.g., "IMG_8286/IMG_8286_frame_000001_left.jpg" -> "IMG_8286")
                    video_name = img_path.split('/')[0] if '/' in img_path else img_path.split('\\')[0]
                    
                    if video_name not in video_data:
                        video_data[video_name] = []
                    
                    video_data[video_name].append({
                        'image_path': img_path,
                        'label': int(action_label),
                        'side': side
                    })
    
    if not video_data:
        raise ValueError("No valid data found in labels file")
    
    # Sort videos by name and split
    video_names = sorted(video_data.keys())
    train_videos = video_names[:-val_videos]
    val_videos_list = video_names[-val_videos:]
    
    print(f"Total videos: {len(video_names)}")
    print(f"Train videos: {len(train_videos)} ({train_videos[:3]}...{train_videos[-3:]})")
    print(f"Val videos: {len(val_videos_list)} ({val_videos_list})")
    
    # Separate train and validation data
    train_data = []
    val_data = []
    
    for video_name in train_videos:
        train_data.extend(video_data[video_name])
    
    for video_name in val_videos_list:
        val_data.extend(video_data[video_name])
    
    # Convert to lists
    X_train = [item['image_path'] for item in train_data]
    y_train = [item['label'] for item in train_data]
    X_val = [item['image_path'] for item in val_data]
    y_val = [item['label'] for item in val_data]
    
    # Print statistics
    train_pos = sum(y_train)
    val_pos = sum(y_val)
    
    print(f"\nDataset Statistics:")
    print(f"Train: {len(X_train)} samples, positive: {train_pos} ({train_pos/len(X_train)*100:.1f}%)")
    print(f"Val: {len(X_val)} samples, positive: {val_pos} ({val_pos/len(X_val)*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val)

def get_transforms(input_height=320, input_width=128, augment=True):
    """Get data transforms for rectangular images"""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # Reduced rotation for rectangular images
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduced translation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(model_name, num_classes=2, pretrained=True):
    """Create timm model"""
    print(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, writer=None, epoch=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
            # Log batch-level metrics to TensorBoard
            if writer is not None and epoch is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
                writer.add_scalar('Train/Batch_Accuracy', 100.*correct/total, global_step)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Action', 'Action'], 
                yticklabels=['No Action', 'Action'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, writer=None):
    """Main training loop"""
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    start_epoch = args.start_epoch
    
    print(f"\nStarting training from epoch {start_epoch+1} to {args.epochs}...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Metrics/Precision', val_precision, epoch)
            writer.add_scalar('Metrics/Recall', val_recall, epoch)
            writer.add_scalar('Metrics/F1', val_f1, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'args': args,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'args': args,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    return train_losses, val_losses, train_accs, val_accs, best_val_acc

def validate_model(model, val_loader, criterion, device, args):
    """Run validation only"""
    print("\nRunning validation...")
    val_loss, val_acc, val_precision, val_recall, val_f1, val_pred, val_true = validate_epoch(
        model, val_loader, criterion, device
    )
    
    print(f"Validation Results:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")
    print(f"Val Precision: {val_precision:.4f}")
    print(f"Val Recall: {val_recall:.4f}")
    print(f"Val F1-Score: {val_f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(val_true, val_pred, 
                         os.path.join(args.output_dir, 'validation_confusion_matrix.png'))
    
    return val_loss, val_acc, val_precision, val_recall, val_f1, val_pred, val_true

def save_checkpoint(state, filepath):
    """Save checkpoint"""
    torch.save(state, filepath, _use_new_zipfile_serialization=False)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    loaded_data = {
        'epoch': checkpoint.get('epoch', 0),
        'val_acc': checkpoint.get('val_acc', 0.0),
        'best_val_acc': checkpoint.get('best_val_acc', 0.0),
        'args': checkpoint.get('args', None),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'train_accs': checkpoint.get('train_accs', []),
        'val_accs': checkpoint.get('val_accs', [])
    }
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return loaded_data

def main():
    parser = argparse.ArgumentParser(description='Train image classification model with timm')
    
    # Data arguments
    parser.add_argument('--data_dir', default='crops_dataset', help='Dataset directory')
    parser.add_argument('--labels_file', default='crops_dataset/labels.txt', help='Labels file path')
    parser.add_argument('--output_dir', default='trainings', help='Output directory')
    
    # Model arguments
    parser.add_argument('--model_name', default='efficientnet_b0', help='timm model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='cosine', help='LR scheduler')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'adam', 'sgd'], help='Optimizer type')
    
    # Data arguments
    parser.add_argument('--val_videos', type=int, default=12, help='Number of videos for validation')
    parser.add_argument('--input_height', type=int, default=320, help='Input image height')
    parser.add_argument('--input_width', type=int, default=128, help='Input image width')
    parser.add_argument('--augment', action='store_true', default=True, help='Use data augmentation')
    
    # Class balancing arguments
    parser.add_argument('--class_balance', choices=['none', 'weights', 'sampling', 'focal'], default='weights', 
                       help='Class balancing method: none, weights (weighted loss), sampling (weighted sampler), focal (focal loss)')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    
    # Checkpoint and validation arguments
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation (requires --checkpoint)')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint for validation or resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (for resuming training)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Use TensorBoard logging')
    parser.add_argument('--log_dir', default=None, help='TensorBoard log directory (auto-generated if None)')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create experiment name
    exp_name = f"{args.model_name}_bs{args.batch_size}_lr{args.lr:.0e}_ep{args.epochs}_{args.optimizer}_{args.scheduler}"
    if args.weight_decay != 1e-4:
        exp_name += f"_wd{args.weight_decay:.0e}"
    if args.augment:
        exp_name += "_aug"
    if args.class_balance != 'none':
        exp_name += f"_bal{args.class_balance}"
        if args.class_balance == 'focal':
            exp_name += f"_a{args.focal_alpha}_g{args.focal_gamma}"
    print(f"Experiment name: {exp_name}")
    
    # Create output directory with experiment name
    exp_output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    print(f"Experiment output directory: {exp_output_dir}")
    
    # Overwrite output_dir with experiment directory so everything gets stored there
    args.output_dir = exp_output_dir
    args.exp_name = exp_name
    
    # Setup TensorBoard
    writer = None
    if args.tensorboard:
        if args.log_dir is None:
            # Auto-generate log directory name
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(args.output_dir, f"tensorboard_{timestamp}")
        else:
            log_dir = args.log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Run 'tensorboard --logdir {log_dir}' to view logs")
    
    # Load dataset
    (X_train, y_train), (X_val, y_val) = load_dataset(
        args.labels_file, args.data_dir, args.val_videos, args.seed
    )
    
    # Calculate class weights for balancing
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    print(f"\nClass distribution in training set:")
    print(f"Class 0 (No Action): {class_counts[0]} samples ({class_counts[0]/total_samples*100:.1f}%)")
    print(f"Class 1 (Action): {class_counts[1]} samples ({class_counts[1]/total_samples*100:.1f}%)")
    print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
    
    # Calculate class weights (inverse frequency)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"Class weights: {class_weights}")
    print(f"Using class balancing method: {args.class_balance}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(args.input_height, args.input_width, args.augment)
    
    # Create datasets
    train_dataset = BowlingDataset(X_train, y_train, train_transform, args.data_dir)
    val_dataset = BowlingDataset(X_val, y_val, val_transform, args.data_dir)
    
    # Create weighted sampler for class balancing
    train_sampler = None
    if args.class_balance == 'sampling':
        # Calculate sample weights for weighted random sampling
        sample_weights = [class_weights[label] for label in y_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"Using WeightedRandomSampler for class balancing")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    model = create_model(args.model_name, args.num_classes, args.pretrained)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture to TensorBoard
    if writer is not None:
        # Log hyperparameters
        hparams = {
            'model': args.model_name,
            'optimizer': args.optimizer,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'input_height': args.input_height,
            'input_width': args.input_width,
            'epochs': args.epochs,
            'val_videos': args.val_videos,
            'augment': args.augment,
            'exp_name': exp_name
        }
        writer.add_hparams(hparams, {'hparam/placeholder': 0})
        
        # Log model graph (try with a sample batch)
        try:
            sample_batch = next(iter(train_loader))[0][:1].to(device)  # Take first image only
            writer.add_graph(model, sample_batch)
        except Exception as e:
            print(f"Could not log model graph: {e}")
    
    # Loss function with class balancing
    if args.class_balance == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    elif args.class_balance == 'weights':
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
        print(f"Using weighted CrossEntropyLoss with class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (no class balancing)")
    
    # Create optimizer based on args
    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    print(f"Using optimizer: {args.optimizer} with lr={args.lr}, weight_decay={args.weight_decay}")
    
    # Scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint_data = load_checkpoint(args.checkpoint, model, optimizer, scheduler)
        args.start_epoch = checkpoint_data['epoch'] + 1
        best_val_acc = checkpoint_data['best_val_acc']
        train_losses = checkpoint_data['train_losses']
        val_losses = checkpoint_data['val_losses']
        train_accs = checkpoint_data['train_accs']
        val_accs = checkpoint_data['val_accs']
        print(f"Resumed from epoch {args.start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # Validation only mode
    if args.validate_only:
        if not args.checkpoint:
            # Use best model if no specific checkpoint provided
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                load_checkpoint(best_model_path, model)
                print("Loaded best model for validation")
            else:
                raise ValueError("No checkpoint provided and no best model found. Use --checkpoint to specify model.")
        
        val_loss, val_acc, val_precision, val_recall, val_f1, val_pred, val_true = validate_model(
            model, val_loader, criterion, device, args
        )
        
        # Save validation results
        results = {
            'exp_name': exp_name,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'args': vars(args)
        }
        
        import json
        with open(os.path.join(args.output_dir, 'validation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Validation completed! Results saved to {args.output_dir}")
        return
    
    # Training mode
    train_losses, val_losses, train_accs, val_accs, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, writer
    )
    
    # Log final metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Final/Best_Validation_Accuracy', best_val_acc, args.epochs)
        
        # Log training curves as images
        if train_losses and val_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(train_losses, label='Train')
            ax1.plot(val_losses, label='Validation')
            ax1.set_title('Loss')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(train_accs, label='Train')
            ax2.plot(val_accs, label='Validation')
            ax2.set_title('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            writer.add_figure('Training_Curves', fig, args.epochs)
            plt.close(fig)
    
    # Load best model for final evaluation
    checkpoint_data = load_checkpoint(os.path.join(args.output_dir, 'best_model.pth'), model)
    
    # Final validation evaluation
    print("\nFinal Validation Evaluation:")
    print("=" * 50)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_pred, val_true = validate_epoch(
        model, val_loader, criterion, device
    )
    
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Val Accuracy: {val_acc:.2f}%")
    print(f"Final Val Precision: {val_precision:.4f}")
    print(f"Final Val Recall: {val_recall:.4f}")
    print(f"Final Val F1-Score: {val_f1:.4f}")
    
    # Log final validation results to TensorBoard
    if writer is not None:
        writer.add_scalar('Final/Validation_Loss', val_loss, 0)
        writer.add_scalar('Final/Validation_Accuracy', val_acc, 0)
        writer.add_scalar('Final/Validation_Precision', val_precision, 0)
        writer.add_scalar('Final/Validation_Recall', val_recall, 0)
        writer.add_scalar('Final/Validation_F1', val_f1, 0)
        
        # Log confusion matrix as image
        try:
            cm = confusion_matrix(val_true, val_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Action', 'Action'], 
                       yticklabels=['No Action', 'Action'], ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            writer.add_figure('Confusion_Matrix', fig, 0)
            plt.close(fig)
        except Exception as e:
            print(f"Could not log confusion matrix to TensorBoard: {e}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         os.path.join(args.output_dir, 'training_history.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(val_true, val_pred, 
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save training results
    results = {
        'exp_name': exp_name,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'final_val_precision': val_precision,
        'final_val_recall': val_recall,
        'final_val_f1': val_f1,
        'args': vars(args)
    }
    
    import json
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {writer.log_dir}")
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
