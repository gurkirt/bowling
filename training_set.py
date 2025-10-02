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
import numpy as np
import random 
import shutil

Frame: TypeAlias = np.ndarray 

def write(video_dir: Path, output_dir: Path, istest: bool) -> None:
    """Extract frames from videos in video_dir, and save them to output_dir.
    The filename of each frame includes whether the baller is in balling action.
    The frames are randomly split into train and eval sets.
    If istest is True, only the frames of a single video is processed.

    Example output:

    training_set/frames/
             .../frames/IMG_8286
             .../frames/IMG_8286/frame_00130_false.jpg
             .../frames/IMG_8286/frame_00140_false.jpg
             .../frames/IMG_8286/frame_00149_true.jpg
             .../frames/IMG_8286/frame_00150_true.jpg
             .../frames/IMG_8286/frame_00310_false.jpg
    training_set/dataloader/
             .../dataloader/train
                        .../train/frame_00000.jpg -> training_set/frames/IMG_8286/frame_00130_false.jpg
                        .../train/frame_00001.jpg -> training_set/frames/IMG_8286/frame_00140_false.jpg
                        .../train/frame_00002.jpg -> training_set/frames/IMG_8286/frame_00150_true.jpg
             .../dataloader/eval
                        .../eval/frame_00000.jpg -> training_set/frames/IMG_8286/frame_00149_true.jpg
                        .../eval/frame_00001.jpg -> training_set/frames/IMG_8286/frame_00310_false.jpg
    """
    frames_dir = output_dir / 'frames'
    train_dir = output_dir / 'dataloader' / 'train'
    eval_dir = output_dir / 'dataloader' / 'eval'

    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    eval_dir.mkdir()
    eval_counter, train_counter = 0,0
    for i, (framefilename, frame, in_action) in enumerate(_extract(Path(video_dir), istest)):
        if not in_action and  i % 10 != 0:
            # Save only 10% of the frames "not in action".
            continue
        framepath = frames_dir / framefilename
        framepath.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(framepath), _crop(frame))
        if random.choice([True, False]):
            (train_dir / f'frame_{train_counter:05d}.jpg').symlink_to(framepath)
            train_counter += 1
        else:
            (eval_dir / f'frame_{eval_counter:05d}.jpg').symlink_to(framepath)
            eval_counter += 1
        


def _extract(video_dir: Path, istest: bool) -> Iterator[Tuple[Path, Frame, bool]]:
     for ext in '*.MOV' , '*.mp4', '*.avi':
        for video in video_dir.glob(ext):
           try:
               with open(video.with_suffix('.json')) as f:
                   events = json.load(f)['temporal_events']
           except Exception as e:
               print(f"Warning: Skipping {video.name}: could not read annotation {e}")
               continue

           for i, frame in enumerate(_frames(video)):
               in_action = any(e["start_frame"] - 2 <= i <= e['end_frame'] for e in events)
               yield Path(video.stem) / f'frame_{i:05d}_{str(in_action).lower()}.jpg', frame, in_action

           if istest:
               print("Test mode: processed a single video and exiting.")
               break


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
    print(f"Extracting frames from {video.name}: {frame_count} frames")
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


if __name__ == '__main__':
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
    else:
        write(Path(args.video_dir), Path(args.output_dir), args.test)   