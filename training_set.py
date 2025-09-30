# training_set offers the write() function to extract frames from videos and read() function to read the extracted frames.
# training_set also offers a main program to extract frames from videos in a directory.
# 
# extract frames from the default directory and save them to the default output directory "training_set/".
# $ python3 training_set.py 
#
# Summarizes the training set in the given directory.
# $ python3 training_set.py --summary /mnt/chromeos/removable/usbdisk/training_set/
# Total frames: 22474, In action: 5% (1263), Not in action: 94% (21211).

import argparse
import collections
from typing import Callable, Iterator, Tuple, TypeAlias, Optional
import numpy as np
from pathlib import Path
import json
import cv2
import shutil

Frame: TypeAlias = np.ndarray 


def write(video_dir: Path, output_dir: Path, istest: bool) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    for index, (video_path, frame, in_action) in enumerate(_extract(Path(video_dir), istest)):
        frames_dir = output_dir / video_path.stem
        frames_dir.mkdir(exist_ok=True)
        frame_path = frames_dir / Path(f'frame_{index:05d}_{str(in_action).lower()}.jpg' )
        if in_action:
          cv2.imwrite(str(frame_path), frame)
        elif index % 10 == 0:
          # Save only 10% of the frames not in action.
          # this should still save 2X more frames not in action than in action.
          cv2.imwrite(str(frame_path), frame)


def read(frame_dir: Path) -> Iterator[Tuple[Path, Frame, bool]]:
    for frame_path, in_action in _read_paths(frame_dir):
        yield frame_path, cv2.imread(str(frame_path)), in_action


def _read_paths(frame_dir: Path) -> Iterator[Tuple[Path, bool]]:
    for frame_path in frame_dir.glob("*/*.jpg"):
        yield frame_path, frame_path.stem.lower().endswith('true')


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
               yield video, frame, in_action

           if istest:
               print("Test mode: processed a single video and exiting.")
               break


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
    for path, in_action in _read_paths(frame_dir):
        frame_counts[path.parent.name][in_action] += 1
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
    parser.add_argument("summary_dir", type=str, default=None,
                       help="The frames directory for the summary is the first positional arg.")


    args = parser.parse_args()
    if args.summary:
        if args.summary_dir is None:
            parser.error("When --summary is set, the frames directory must be provided as the first positional arg.")
        print(_summary(Path(args.summary_dir)))
    else:
        write(Path(args.video_dir), Path(args.output_dir), args.test)