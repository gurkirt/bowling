# training_set offers the write() function to extract frames from videos and read() function to read the extracted frames.
# training_set also offers a main program to extract frames from videos in a directory.
# 
# extract frames from the default directory and save them to the default output directory "training_set/".
# $ python3 training_set.py 
#
# Summarizes the training set in the given directory.
# $ python3 training_set.py --summary /mnt/chromeos/removable/usbdisk/training_set/
# Total frames: 22474, In action: 5% (1263), Not in action: 94% (21211).

from pathlib import Path
from typing import Iterator, Tuple, TypeAlias

import argparse
import collections
import cv2
import json
import os
import numpy as np
import random 
import shutil
import threading
from queue import Queue

Frame: TypeAlias = np.ndarray 

def add_symlinks(output_dir: Path) -> None:
    """Create train/eval splits by creating symlinks to frames."""
    for fold in range(10):
        train_dir = output_dir / 'dataloader' / str(fold) / 'train'
        eval_dir = output_dir / 'dataloader' / str(fold)/ 'eval'
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_counter, eval_counter = 0, 0
        for i, video in enumerate((output_dir / 'frames').iterdir()):
            if i + fold % 10:
                for frame in video.glob('*.jpg'):
                    (train_dir / f'frame_{train_counter:05d}.jpg').symlink_to(frame)
                    train_counter += 1
            else:
                for frame in video.glob('*.jpg'):
                   (eval_dir / f'frame_{eval_counter:05d}.jpg').symlink_to(frame)
                   eval_counter += 1

def worker(queue: Queue, frames_dir: Path):
    while True:
        video = queue.get()
        if video is None:
            queue.task_done()
            break
        write_one(video, frames_dir)
        queue.task_done()

def write(output_dir: Path, video_dir: Path, istest: bool) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True)
    video_queue = Queue()
    num_threads = os.cpu_count() 
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(video_queue, frames_dir))
        t.start()
        threads.append(t)
    for ext in ('*.MOV', '*.mp4', '*.avi'):
        for video in Path(video_dir).glob(ext):
            video_queue.put(video)
            if istest:
                print("test mode: processing a single video and exiting")
                break
    for _ in threads:
        video_queue.put(None)
    video_queue.join()
    for t in threads:
        t.join()


def write_one(video: Path, frames_dir: Path) -> None:
    """
    Extract frames from videos in video_dir, save to frames_dir.
    Only 10% of 'not in action' frames are saved.
    """
    try:
        with open(video.with_suffix('.json')) as f:
            events = json.load(f)['temporal_events']
    except Exception as e:
        print(f"Warning: Skipping {video.name}: could not read annotation {e}")
        return

    for i, frame in enumerate(_frames(video)):
        in_action = any(e["start_frame"] - 2 <= i <= e['end_frame'] for e in events)
        if not in_action and random.randint(0, 10) % 10 != 0:
            # Save only 10% of the frames "not in action".
            continue
        framefilename = Path(video.stem) / f'frame_{i:05d}_{str(in_action).lower()}.jpg'
        framepath = frames_dir / framefilename
        framepath.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(framepath), _crop(frame))


def _crop(frame: Frame) -> Frame:
    """Remove 32% from the top vertically, and crop the frame to a square centered horizontally."""
    h, w, _ = frame.shape
    top_offset = int(h * 0.32)
    new_h = h - top_offset
    side = min(new_h, w)
    center_x = w // 2
    center_y = top_offset + new_h // 2
    left = max(center_x - side // 2, 0)
    right = left + side
    top = max(center_y - side // 2, top_offset)
    bottom = top + side
    return frame[top:bottom, left:right]


def _frames(video: Path) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"extracting frames from {video.name}: {frame_count} frames")
    try:
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          yield frame
    finally:
        cap.release()


def _summary(frame_dir: Path) -> str:
    frame_counts = collections.defaultdict(collections.Counter)
    for frame_path in frame_dir.glob("*/*.jpg"):
        in_action= frame_path.stem.lower().endswith('true')
        frame_counts[frame_path.parent.name][in_action] += 1
    total, in_action, not_in_action = 0, 0, 0
    for counts in frame_counts.values():
        in_action += counts[True]
        not_in_action += counts[False]
        total += counts[False] + counts[True]
    return f"Total frames: {total}, In action: {int(in_action*100/total)}% ({in_action}), Not in action: {int(100 *not_in_action/total)}% ({not_in_action})."

def main():
    parser = argparse.ArgumentParser(description="Extract the frames annotated with whether the baller is in balling action.")
    parser.add_argument("--video_dir", default="videos-aug31st", 
                       help="Directory containing video files and annotations")
    parser.add_argument("--output_dir", default="training_set", 
                       help="Directory to save the extracted frames. The filename includes whether the baller is in balling action.")
    parser.add_argument("--test", default=False, action='store_true',
                       help="If --test is set, then the frames of a single video is processed.")
    parser.add_argument("--summary", default=False, action='store_true',
                       help="When --summary is set, the script displays stats on the produced training set.")
    parser.add_argument("--summary_dir", type=str, default=None,
                       help="The frames directory for the summary is the first positional arg.")

    args = parser.parse_args()
    if args.summary:
        if args.summary_dir is None:
            parser.error("When --summary is set, the frames directory must be provided as the first positional arg.")
        print(_summary(Path(args.summary_dir)))
        return

    write(Path(args.output_dir), Path(args.video_dir), args.test)
    add_symlinks(Path(args.output_dir))

if __name__ == '__main__':
    main()
