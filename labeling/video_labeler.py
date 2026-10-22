# video_labeler.py
import os
import json
from glob import glob
import cv2
import argparse
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from autolabel.sam3_tracks import SequentialSeekCapture  # noqa: E402


def get_video_files(video_dir):
    ## .MOV or .mp4 
    return sorted(glob(os.path.join(video_dir, '*.MOV')) + glob(os.path.join(video_dir, '*.mp4')))

def annotation_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(os.path.dirname(video_path), f"{base}.json")


def label_video(video_path, class_name, previous=None):
    print(f"\nLabeling: {video_path}")
    cap = SequentialSeekCapture(lambda: cv2.VideoCapture(video_path))  # frame i as the pipeline decodes it
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    print(f"Classes: {class_name}")

    current_frame = 0
    instances = [dict(ev) for ev in (previous or [])]  # List of action instances
    current_start = None
    current_end = None

    def show_frame(frame_idx, current_start, current_end, instances):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return False
        display = frame.copy()
        top_offset = 100
        bottom_offset = 10
        scale_factor_1 = 2.0
        max_width = 640*scale_factor_1
        max_height = 480*scale_factor_1
        scale_factor = max_width/display.shape[1] if display.shape[1] > max_width else 1
        scale_factor = min(scale_factor, max_height/display.shape[0] if display.shape[0] > max_height else 1)
        # Display current instance info
        text = f"Frame: {frame_idx+1}/{total_frames} | Current: Start={current_start} End={current_end}"
        # print(scale_factor)
        cv2.putText(display, text, (10, top_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, max(1,int(2*scale_factor)), (0,255,0), int(1/scale_factor))

        # Display existing instances
        inside = [i + 1 for i, ev in enumerate(instances) if ev["start_frame"] <= frame_idx <= ev["end_frame"]]
        instances_text = f"Instances: {len(instances)}" + (f" | IN #{inside[0]}" if inside else "")
        cv2.putText(display, instances_text, (10, top_offset + 160), cv2.FONT_HERSHEY_SIMPLEX, max(1,int(2*scale_factor)), (255,0,0), int(1/scale_factor))

        # Crop and resize
        display = display[top_offset:-bottom_offset, :]
        
        display = cv2.resize(display, (0, 0), fx=scale_factor, fy=scale_factor)
        print(f"Scale factor: {scale_factor}", display.shape)
        cv2.imshow("Label Video", display)
        return True

    def add_instance(current_start, current_end, instances):
        if current_start <= current_end:
            instance = {
                "start_frame": current_start,
                "end_frame": current_end,
                "middle_frame": (current_start + current_end) // 2,
                "class": class_name
            }
            instances.append(instance)
            print(f"Added instance {len(instances)}: frames {current_start}-{current_end} (middle: {instance['middle_frame']})")
            current_start = None
            current_end = None
            print("Ready for next instance. Set start frame (S) and end frame (E).")
        else:
            print("Error: Start frame must be <= end frame. Fix frames or press 'N' again to skip.")

    print("Instructions:")
    print("  Right/Left arrow: Next/Prev frame")
    print("  S: Set start frame")
    print("  E: Set end frame")
    print("  N: Next instance (auto-saves current start/end if both are set)")
    print("  D: Delete last instance")
    print("  L: List all instances")
    print("  J: Jump to the next instance's start   R: Remove the instance nearest this frame")
    print("  Q: Quit and save annotation")

    while True:
        if not show_frame(current_frame, current_start, current_end, instances):
            print("Could not read frame.")
            break
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            if current_start is not None and current_end is not None:
                add_instance(current_start, current_end, instances)  # Try to add current instance before quitting
            break
        elif key == ord('s'):
            current_start = current_frame
            print(f"Start frame set to {current_start}")
        elif key == ord('e'):
            current_end = current_frame
            print(f"End frame set to {current_end}")
            # Auto-check if we can move to next instance
            if current_start is not None and current_end is not None:
                print("Both start and end frames set. Press 'N' for next instance or continue labeling.")
        elif key == ord('n'):
            # Move to next instance - auto-save current if both start and end are set
            if current_start is not None and current_end is not None:
                add_instance(current_start, current_end, instances)
                current_start = None
                current_end = None
            else:
                print("Cannot move to next instance: Set both start and end frames first")
        elif key == ord('a'):
            # Keep old 'A' functionality for manual add
            if current_start is not None and current_end is not None:
                if current_start <= current_end:
                    instance = {
                        "start_frame": current_start,
                        "end_frame": current_end,
                        "middle_frame": (current_start + current_end) // 2,
                        "class": class_name
                    }
                    instances.append(instance)
                    print(f"Added instance {len(instances)}: frames {current_start}-{current_end} (middle: {instance['middle_frame']})")
                    current_start = None
                    current_end = None
                else:
                    print("Error: Start frame must be <= end frame")
            else:
                print("Error: Set both start and end frames first")
        elif key == ord('d'):
            if instances:
                removed = instances.pop()
                print(f"Removed instance: frames {removed['start_frame']}-{removed['end_frame']}")
            else:
                print("No instances to remove")
        elif key == ord('j') and instances:
            later = sorted(ev["start_frame"] for ev in instances if ev["start_frame"] > current_frame)
            current_frame = later[0] if later else min(ev["start_frame"] for ev in instances)
        elif key == ord('r') and instances:
            near = min(instances, key=lambda ev: abs(ev["middle_frame"] - current_frame))
            instances.remove(near)
            print(f"Removed instance: frames {near['start_frame']}-{near['end_frame']}")
        elif key == ord('l'):
            print(f"Current instances ({len(instances)}):")
            for i, inst in enumerate(instances):
                print(f"  {i+1}: frames {inst['start_frame']}-{inst['end_frame']} (middle: {inst['middle_frame']})")
        elif key == 81:  # left arrow
            current_frame = max(0, current_frame - 1)
        elif key == 83:  # right arrow
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('w'):  # step forward 10 frames
            current_frame = min(total_frames - 1, current_frame + 10)
        elif key == ord('x'):  # step backward 10 frames
            current_frame = max(0, current_frame - 10)
            
    cv2.destroyAllWindows()
    cap.release()
    return {
        "video": os.path.basename(video_path),
        "total_frames": total_frames,
        "temporal_events": sorted(instances, key=lambda ev: ev["start_frame"])
    }






def label_bounding_box(video_path, current_frame, current_box = None, class_name="bowler", context=()):
    print(f"\nBounding box labeling: {video_path} at frame {current_frame}")
    cap = SequentialSeekCapture(lambda: cv2.VideoCapture(video_path))  # frame i as the pipeline decodes it
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("Could not read start frame.")
        cap.release()
        return {}
    
    # Variables for bounding box
    drawing = False
    boxes = []
    
    ix, iy = -1, -1
    
    # Create a copy for drawing
    display_frame = frame.copy()
    top_offset = 100
    for b in context:  # other boxes already on this frame (stumps, batters), for reference
        cv2.rectangle(display_frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 200, 255), 2)
        cv2.putText(display_frame, b.get("class", ""), (b["x1"], b["y1"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    display_frame = display_frame[top_offset:, :]
    max_width, max_height = 1280, 960
    scale_factor = min(1.0, max_width / display_frame.shape[1], max_height / display_frame.shape[0])
    display_frame = cv2.resize(display_frame, (0, 0), fx=scale_factor, fy=scale_factor)
    print(f"Scale factor: {scale_factor}", display_frame.shape)
    original_frame = display_frame.copy()
    
    def draw_existing_boxes():
        temp_frame = original_frame.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(temp_frame, f"Box {i+1} {class_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return temp_frame
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, current_box, display_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            current_box = [ix, iy, ix, iy]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                display_frame = draw_existing_boxes()
                current_box[2], current_box[3] = x, y
                cv2.rectangle(display_frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (255, 0, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_box[2], current_box[3] = x, y
            if abs(current_box[2] - current_box[0]) > 10 and abs(current_box[3] - current_box[1]) > 10:
                # Normalize coordinates (ensure x1 < x2, y1 < y2)
                x1, y1 = min(current_box[0], current_box[2]), min(current_box[1], current_box[3])
                x2, y2 = max(current_box[0], current_box[2]), max(current_box[1], current_box[3])
                boxes.append([x1, y1, x2, y2])
                print(f"Added box {len(boxes)}: [{x1}, {y1}, {x2}, {y2}]")
            display_frame = draw_existing_boxes()
    
    cv2.namedWindow("Bounding Box Annotation")
    cv2.setMouseCallback("Bounding Box Annotation", mouse_callback)
    
    print("Instructions:")
    print("  Click and drag to draw bounding boxes")
    print("  u: Undo last box")
    print("  c: Clear all boxes")
    print("  q: Quit and save boxes")
    
    display_frame = draw_existing_boxes()
    
    while True:
        cv2.imshow("Bounding Box Annotation", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('u') and boxes:
            boxes.pop()
            print(f"Removed last box. {len(boxes)} boxes remaining.")
            display_frame = draw_existing_boxes()
        elif key == ord('c'):
            boxes.clear()
            print("Cleared all boxes.")
            display_frame = draw_existing_boxes()
    
    cv2.destroyAllWindows()
    cap.release()
    
    # Convert boxes back to original frame coordinates and format
    formatted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Scale back up
        x1, x2 = int(x1 / scale_factor), int(x2 / scale_factor)
        y1, y2 = int(y1 / scale_factor), int(y2 / scale_factor)
        # Add back the offset
        y1, y2 = y1 + top_offset, y2 + top_offset
        
        # Format as required with class field
        formatted_box = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class": class_name
        }
        formatted_boxes.append(formatted_box)
    
    return {str(current_frame): formatted_boxes}


    
def label_actions(args):
    class_name = args.action_class
    video_files = get_video_files(args.video_dir)
    for video_path in video_files:
        anno_path = annotation_path(video_path)
        needs_label = False
        annotation = None
        if os.path.exists(anno_path):
            with open(anno_path, 'r', encoding='utf-8') as f:
                anno = json.load(f)
            # Check if start/end frame are valid
            temporal_events = anno.get('temporal_events')
            # An empty list is a labelled clip with no deliveries; only a missing key or a
            # malformed event means it still needs labelling.
            if temporal_events is None or any('start_frame' not in ev or 'end_frame' not in ev for ev in temporal_events):
                print(f"Annotation for {video_path} is incomplete. Relabeling...")
                needs_label = True
            else:
                annotation = anno

        else:
            needs_label = True

        if needs_label:
            annotation = label_video(video_path, class_name)
            if os.path.exists(anno_path):  # keep boxes and anything else already in the file
                with open(anno_path, 'r', encoding='utf-8') as f:
                    annotation = {**json.load(f), **annotation}
            with open(anno_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2)
            print(f"Saved annotation to {anno_path}")
        
        # Always reload annotation after possible updat
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

def relabel_video(video_path, class_name):
    """Redo one video's deliveries, starting from the saved ones, without losing its boxes.

    The annotation is backed up to labels/relabel_backup/ first. Unchanged deliveries keep
    their boxes. A changed delivery takes the stumps/striker/non_striker boxes of the old
    delivery it overlaps (or is nearest to), moved to its new middle frame, and you redraw
    its bowler box there. Boxes of deleted deliveries are dropped.
    """
    anno_path = annotation_path(video_path)
    with open(anno_path, 'r', encoding='utf-8') as f:
        old = json.load(f)
    backup = os.path.join("labels", "relabel_backup",
                          f"{os.path.splitext(os.path.basename(anno_path))[0]}.{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    shutil.copy(anno_path, backup)
    print(f"Backed up {anno_path} -> {backup}")

    old_events = old.get('temporal_events', [])
    old_boxes = old.get('bounding_boxes', {}) or {}

    def box_frame(ev):  # the frame an event's boxes are keyed on: its middle, else any boxed frame inside
        frames = [ev['middle_frame']] + list(range(ev['start_frame'], ev['end_frame'] + 1))
        return next((str(f) for f in frames if str(f) in old_boxes), None)

    new = label_video(video_path, class_name, previous=old_events)
    boxes, redraw, moved_from = {}, [], set()
    unchanged = {(ev['start_frame'], ev['end_frame']) for ev in old_events}
    for ev in new['temporal_events']:
        if (ev['start_frame'], ev['end_frame']) in unchanged:
            src = next(o for o in old_events if (o['start_frame'], o['end_frame']) == (ev['start_frame'], ev['end_frame']))
            key = box_frame(src)
            if key:
                boxes[key] = old_boxes[key]
            continue
        src = min(old_events, key=lambda o: abs(o['middle_frame'] - ev['middle_frame']), default=None)
        if src and abs(src['middle_frame'] - ev['middle_frame']) > 60:
            src = None
        key = box_frame(src) if src else None
        kept = [b for b in old_boxes.get(key, []) if b.get('class', 'bowler') != 'bowler']
        boxes[str(ev['middle_frame'])] = kept
        moved_from.add(key)
        redraw.append(ev)
        print(f"Changed delivery {ev['start_frame']}-{ev['end_frame']}: moved {[b['class'] for b in kept]} "
              f"from frame {key} to {ev['middle_frame']}; bowler box to redraw")
    dropped = set(old_boxes) - set(boxes) - moved_from
    if dropped:
        print(f"Dropping boxes on frames {sorted(dropped, key=int)} (no delivery uses them any more)")

    for ev in redraw:
        key = str(ev['middle_frame'])
        drawn = label_bounding_box(video_path, ev['middle_frame'], class_name="bowler", context=boxes[key]).get(key, [])
        if drawn:
            boxes[key] = boxes[key] + drawn[:1]
        else:
            print(f"No bowler box drawn for {ev['start_frame']}-{ev['end_frame']}; it will have no side / no SAM3 prompt")

    old['temporal_events'] = new['temporal_events']
    old['bounding_boxes'] = dict(sorted(boxes.items(), key=lambda kv: int(kv[0])))
    with open(anno_path, 'w', encoding='utf-8') as f:
        json.dump(old, f, indent=2)
    print(f"Saved {anno_path}")


def label_boxes(args):
    box_class = args.box_class
    video_files = get_video_files(args.video_dir)
    for video_path in video_files:
        anno_path = annotation_path(video_path)
        if os.path.exists(anno_path):
            with open(anno_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
        else:
            print(f"No annotation file found for {video_path}. Skipping...")
            continue

        # Check if temporal_events exist
        temporal_events = annotation.get('temporal_events', [])
        if not temporal_events:
            print(f"No temporal events found for {video_path}. Skipping...")
            continue
        
        bbox_labels = annotation.get('bounding_boxes', {})
        
        # Loop over each temporal event
        for event_idx, event in enumerate(temporal_events):
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            middle_frame = event['middle_frame']
            event_class = event.get('class', 'bowling')
            
            print(f"\nProcessing temporal event {event_idx + 1}/{len(temporal_events)} for {video_path}")
            print(f"Event: frames {start_frame}-{end_frame}, class: {event_class}")
            
            # Check if bounding box exists for any frame in the event range with the correct class
            needs_bbox = True
            
            # Loop through all frames in the temporal event range
            for frame_idx in range(start_frame, end_frame + 1):
                frame_str = str(frame_idx)
                if frame_str in bbox_labels:
                    # Check if any bounding box has the correct class
                    for bbox in bbox_labels[frame_str]:
                        if bbox.get('class') == box_class:
                            needs_bbox = False
                            print(f"Bounding box with class '{box_class}' already exists for frame {frame_idx}")
                            break
                    if not needs_bbox:
                        break
            
            if needs_bbox:
                print(f"No bounding box with class '{box_class}' found for frame {middle_frame}. Adding bbox...")
                bbox_labels_new = label_bounding_box(video_path, middle_frame, class_name=box_class)
                
                if bbox_labels_new:  # Only update if we got new boxes
                    if 'bounding_boxes' not in annotation:
                        annotation['bounding_boxes'] = {}
                    annotation['bounding_boxes'].update(bbox_labels_new)
                    
                    # Save annotation after each event
                    with open(anno_path, 'w', encoding='utf-8') as f:
                        json.dump(annotation, f, indent=2)
                    print(f"Saved bounding boxes to {anno_path}")
                else:
                    print(f"No bounding boxes were created for frame {middle_frame}")
            
        print(f"Completed processing all temporal events for {video_path}")


def main():

    parser = argparse.ArgumentParser(description="Label start/end frames for actions in videos.")
    parser.add_argument("--video_dir", help="Directory containing video files", default="videos")
    parser.add_argument("--action_class", help="Action class name", default="bowling", choices=["bowling"])
    parser.add_argument("--box_class", help="Bounding box class name", default="bowler", choices=["bowler"])
    parser.add_argument("--relabel", default="",
                        help="Comma-separated video stems to redo (deliveries + bowler box of changed ones), e.g. IMG_9398_val")
    args = parser.parse_args()
    if args.relabel:
        stems = [s.strip() for s in args.relabel.split(",") if s.strip()]
        for video_path in get_video_files(args.video_dir):
            if os.path.splitext(os.path.basename(video_path))[0] in stems:
                relabel_video(video_path, args.action_class)
        return
    label_actions(args)
    # label_boxes(args)

if __name__ == "__main__":
    main()
