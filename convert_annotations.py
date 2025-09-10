#!/usr/bin/env python3
"""
convert_annotations.py - Convert old annotation format to new format with instances
"""

import os
import json
import argparse
from glob import glob


def convert_annotation(old_annotation):
    """Convert old format annotation to new format"""
    new_annotation = {
        "video": old_annotation.get("video", ""),
        "class": old_annotation.get("class", "bowling"),
        "total_frames": old_annotation.get("total_frames", 0),
        "instances": []
    }
    
    # Convert single start_frame/end_frame to instances array
    if "start_frame" in old_annotation and "end_frame" in old_annotation:
        start_frame = old_annotation["start_frame"]
        end_frame = old_annotation["end_frame"]
        middle_frame = (start_frame + end_frame) // 2
        
        instance = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "middle_frame": middle_frame
        }
        new_annotation["instances"].append(instance)
    
    # Keep bounding_boxes if they exist
    if "bounding_boxes" in old_annotation:
        new_annotation["bounding_boxes"] = old_annotation["bounding_boxes"]
    
    # Note: direction_labels are intentionally removed as they will be inferred from bounding boxes
    
    return new_annotation


def main():
    parser = argparse.ArgumentParser(description="Convert old annotation format to new format")
    parser.add_argument("--video_dir", default="videos-aug31st", help="Directory containing video files and annotations")
    parser.add_argument("--backup", action="store_true", help="Create backup of original files")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be converted without making changes")
    
    args = parser.parse_args()
    
    # Find all JSON files
    json_files = glob(os.path.join(args.video_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {args.video_dir}")
        return
    
    print(f"Found {len(json_files)} annotation files")
    
    converted_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        try:
            # Read original annotation
            with open(json_file, 'r', encoding='utf-8') as f:
                old_annotation = json.load(f)
            
            # Check if already in new format
            if "instances" in old_annotation:
                print(f"SKIP: {os.path.basename(json_file)} - already in new format")
                skipped_count += 1
                continue
            
            # Check if it has the old format fields
            if "start_frame" not in old_annotation or "end_frame" not in old_annotation:
                print(f"SKIP: {os.path.basename(json_file)} - missing start_frame/end_frame")
                skipped_count += 1
                continue
            
            # Convert to new format
            new_annotation = convert_annotation(old_annotation)
            
            if args.dry_run:
                print(f"DRY RUN: Would convert {os.path.basename(json_file)}")
                print(f"  Old: start={old_annotation.get('start_frame')}, end={old_annotation.get('end_frame')}")
                print(f"  New: {len(new_annotation['instances'])} instances")
                if old_annotation.get('direction_labels'):
                    print(f"  Removing direction_labels: {old_annotation['direction_labels']}")
                converted_count += 1
                continue
            
            # Create backup if requested
            if args.backup:
                backup_file = json_file + ".backup"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(old_annotation, f, indent=2)
                print(f"BACKUP: Created {os.path.basename(backup_file)}")
            
            # Write converted annotation
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(new_annotation, f, indent=2)
            
            print(f"CONVERTED: {os.path.basename(json_file)}")
            print(f"  Old: start={old_annotation.get('start_frame')}, end={old_annotation.get('end_frame')}")
            print(f"  New: {len(new_annotation['instances'])} instances")
            if old_annotation.get('direction_labels'):
                print(f"  Removed direction_labels: {old_annotation['direction_labels']}")
            
            converted_count += 1
            
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"ERROR: Failed to convert {os.path.basename(json_file)}: {e}")
    
    print("\nSummary:")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(json_files)}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry_run to actually convert files.")
        print("Recommend using --backup to create backups of original files.")


if __name__ == "__main__":
    main()
