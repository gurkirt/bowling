#!/usr/bin/env python3
"""
Test script to verify the new annotation format
"""
import json

# Test the new annotation format
test_annotation = {
    "temporal_events": [
        {
            "start_frame": 10,
            "end_frame": 50,
            "middle_frame": 30,
            "class": "bowler"
        },
        {
            "start_frame": 100,
            "end_frame": 140,
            "middle_frame": 120,
            "class": "bowler"
        }
    ],
    "bounding_boxes": {
        "30": [{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "class": "bowler"}],
        "120": [{"x1": 150, "y1": 250, "x2": 350, "y2": 450, "class": "bowler"}]
    }
}

print("New annotation format example:")
print(json.dumps(test_annotation, indent=2))

print("\nFormat verification:")
print(f"- Has temporal_events: {'temporal_events' in test_annotation}")
print(f"- Has bounding_boxes: {'bounding_boxes' in test_annotation}")
print(f"- Number of temporal events: {len(test_annotation['temporal_events'])}")
print(f"- Number of annotated frames: {len(test_annotation['bounding_boxes'])}")

# Verify each temporal event has required fields
for i, event in enumerate(test_annotation['temporal_events']):
    required_fields = ['start_frame', 'end_frame', 'middle_frame', 'class']
    has_all_fields = all(field in event for field in required_fields)
    print(f"- Temporal event {i+1} has all required fields: {has_all_fields}")

# Verify bounding boxes have class labels
for frame, boxes in test_annotation['bounding_boxes'].items():
    for j, box in enumerate(boxes):
        has_class = 'class' in box
        print(f"- Frame {frame} box {j+1} has class label: {has_class}")
