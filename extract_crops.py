#!/usr/bin/env python3
"""
extract_crops.py - Extract left/right crops from video frames with binary labels
"""

import os
import json
import glob
import cv2
import argparse
from pathlib import Path

def load_annotation(json_path):
    """Load annotation from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def extract_and_crop_frames(video_path: str, annotation, output_dir, top_crop=570, height_remain=1350):
    """Extract frames from video and create left/right crops with labels"""
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    start_frame = annotation.get('start_frame')
    end_frame = annotation.get('end_frame')
    direction_label = annotation.get('direction_labels')
    
    if start_frame is None or end_frame is None:
        print(f"Skipping {video_name}: missing start/end frame")
        return None
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {video_name}: {total_frames} frames, action from {start_frame} to {end_frame}, direction: {direction_label}")
    
    # Prepare label entries
    label_entries = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop from top (remove top_crop pixels, keep height_remain pixels)
        original_height, original_width = frame.shape[:2]
        cropped_frame = frame[top_crop:, :]
        
        # Check if crop is valid
        if cropped_frame.shape[0] != height_remain:
            print(f"Warning: Frame {frame_idx} height after crop is {cropped_frame.shape[0]}, expected {height_remain}")
            frame_idx += 1
            continue
        
        crop_height, crop_width = cropped_frame.shape[:2]
        half_width = crop_width // 2
        
        # Split into left and right crops
        left_crop = cropped_frame[:, :half_width]
        right_crop = cropped_frame[:, half_width:]
        
        # Determine if action is present (within start_frame-2 to end_frame)
        action_present = (start_frame - 2) <= frame_idx <= end_frame
        
        # Determine action labels (1 if action present and direction matches, 0 otherwise)
        left_action_label = 1 if (action_present and direction_label == 'left') else 0
        right_action_label = 1 if (action_present and direction_label == 'right') else 0
        
        # Save crops
        left_filename = f"{video_name}_frame_{frame_idx:06d}_left.jpg"
        right_filename = f"{video_name}_frame_{frame_idx:06d}_right.jpg"
        
        left_path = os.path.join(video_output_dir, left_filename)
        right_path = os.path.join(video_output_dir, right_filename)
        
        cv2.imwrite(left_path, left_crop)
        cv2.imwrite(right_path, right_crop)
        
        # Create relative paths for the label file
        left_relative_path = os.path.join(video_name, left_filename)
        right_relative_path = os.path.join(video_name, right_filename)
        
        # Add entries in format: <path_image_crop.jpg>,<action_label>,<left or right>
        label_entries.append(f"{left_relative_path},{left_action_label},left")
        label_entries.append(f"{right_relative_path},{right_action_label},right")
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    print(f"  Saved {frame_idx} left crops to {video_output_dir}")
    print(f"  Saved {frame_idx} right crops to {video_output_dir}")
    print(f"  Left positives: {sum(1 for entry in label_entries if 'left' in entry and ',1,' in entry)}")
    print(f"  Right positives: {sum(1 for entry in label_entries if 'right' in entry and ',1,' in entry)}")
    
    return label_entries

def process_videos(video_dir, output_dir, top_crop=570, height_remain=1350):
    """Process all videos in the directory"""
    
    # Find all video files
    video_files = glob.glob(os.path.join(video_dir, "*.MOV"))
    video_files.extend(glob.glob(os.path.join(video_dir, "*.mp4")))
    video_files.extend(glob.glob(os.path.join(video_dir, "*.avi")))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    total_frames = 0
    all_label_entries = []
    
    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(video_dir, f"{video_name}.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: No annotation file found for {video_name}, skipping")
            continue
        
        annotation = load_annotation(json_path)
        if annotation is None:
            continue
        
        try:
            label_entries = extract_and_crop_frames(
                video_path, annotation, output_dir, top_crop, height_remain
            )
            if label_entries:
                all_label_entries.extend(label_entries)
                processed_count += 1
                total_frames += len(label_entries) // 2  # Each frame produces 2 crops
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    # Write all labels to a single file
    labels_file = os.path.join(output_dir, "labels.txt")
    with open(labels_file, 'w') as f:
        f.write('\n'.join(all_label_entries))
    
    print(f"\nCompleted!")
    print(f"Processed {processed_count} videos")
    print(f"Total frames extracted: {total_frames}")
    print(f"Total crops created: {len(all_label_entries)}")
    print(f"Labels saved to: {labels_file}")
    
    # Print label statistics
    left_positives = sum(1 for entry in all_label_entries if 'left' in entry and ',1,' in entry)
    right_positives = sum(1 for entry in all_label_entries if 'right' in entry and ',1,' in entry)
    left_total = sum(1 for entry in all_label_entries if 'left' in entry)
    right_total = sum(1 for entry in all_label_entries if 'right' in entry)
    
    print(f"Left crops: {left_positives} positive / {left_total - left_positives} negative (total: {left_total})")
    print(f"Right crops: {right_positives} positive / {right_total - right_positives} negative (total: {right_total})")

def create_dataset_info(output_dir):
    """Create a summary of the dataset"""
    info_path = os.path.join(output_dir, "dataset_info.txt")
    labels_file = os.path.join(output_dir, "labels.txt")
    
    if not os.path.exists(labels_file):
        print("No labels.txt file found, skipping dataset info creation")
        return
    
    # Read all labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    total_left_pos = 0
    total_left_neg = 0
    total_right_pos = 0
    total_right_neg = 0
    
    video_stats = {}
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) != 3:
            continue
        
        path, action_label, side = parts
        video_name = path.split('/')[0] if '/' in path else path.split('\\')[0]
        action_label = int(action_label)
        
        if video_name not in video_stats:
            video_stats[video_name] = {'left_pos': 0, 'left_neg': 0, 'right_pos': 0, 'right_neg': 0}
        
        if side == 'left':
            if action_label == 1:
                video_stats[video_name]['left_pos'] += 1
                total_left_pos += 1
            else:
                video_stats[video_name]['left_neg'] += 1
                total_left_neg += 1
        elif side == 'right':
            if action_label == 1:
                video_stats[video_name]['right_pos'] += 1
                total_right_pos += 1
            else:
                video_stats[video_name]['right_neg'] += 1
                total_right_neg += 1
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("Dataset Information\n")
        f.write("==================\n\n")
        f.write("Crop parameters:\n")
        f.write("- Top crop: 570 pixels removed\n")
        f.write("- Remaining height: 1350 pixels\n")
        f.write("- Split: left/right halves\n")
        f.write("- Final crop size: 540 x 1350 (width x height)\n\n")
        
        f.write("Label format: <path_image_crop.jpg>,<action_label>,<left or right>\n")
        f.write("- action_label: 0 (no action or wrong direction), 1 (action present and correct direction)\n")
        f.write("- side: 'left' or 'right'\n\n")
        
        f.write("Videos processed:\n")
        
        for video_name in sorted(video_stats.keys()):
            stats = video_stats[video_name]
            total_frames = (stats['left_pos'] + stats['left_neg'])  # Should be same as right
            f.write(f"- {video_name}: {total_frames} frames, "
                   f"left +{stats['left_pos']}/-{stats['left_neg']}, "
                   f"right +{stats['right_pos']}/-{stats['right_neg']}\n")
        
        f.write("\nSummary:\n")
        f.write(f"Total videos: {len(video_stats)}\n")
        f.write(f"Left crops: +{total_left_pos} / -{total_left_neg} (total: {total_left_pos + total_left_neg})\n")
        f.write(f"Right crops: +{total_right_pos} / -{total_right_neg} (total: {total_right_pos + total_right_neg})\n")
        f.write(f"Total crops: {(total_left_pos + total_left_neg + total_right_pos + total_right_neg)}\n")
    
    print(f"Dataset info saved to {info_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract left/right crops from video frames with binary labels")
    parser.add_argument("--video_dir", default="videos-aug31st", 
                       help="Directory containing video files and annotations")
    parser.add_argument("--output_dir", default="crops_dataset", 
                       help="Directory to save extracted crops")
    parser.add_argument("--top_crop", type=int, default=570,
                       help="Pixels to crop from top (default: 570)")
    parser.add_argument("--height_remain", type=int, default=1350,
                       help="Height to remain after top crop (default: 1350)")
    args = parser.parse_args()
    
    print(f"Processing videos from: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Crop parameters: remove top {args.top_crop}px, keep {args.height_remain}px height")
    
    process_videos(args.video_dir, args.output_dir, args.top_crop, args.height_remain)
    create_dataset_info(args.output_dir)

if __name__ == "__main__":
    main()
