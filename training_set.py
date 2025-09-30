# training_set offers the write() function to extract frames from videos and read() function to read the extracted frames.

import argparse
from typing import Callable, Iterator, Tuple, TypeAlias, Optional
import numpy as np
from pathlib import Path
import json
import cv2
import shutil

Frame: TypeAlias = np.ndarray 

def read(frame_dir: Path) -> Iterator[Tuple[Frame, bool]]:
    for frame_path in sorted(frame_dir.glob("*/*.jpg")):
        frame = cv2.imread(str(frame_path))
        in_action = frame_path.stem.lower().endswith('true')
        yield frame, in_action


def write(video_dir: Path, output_dir: Path, istest: bool) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    for index, (video_path, frame, in_action) in enumerate(_extract(Path(video_dir), istest)):
        frames_dir = output_dir / video_path.stem
        frames_dir.mkdir(exist_ok=True)
        frame_path = frames_dir / Path(f'frame_{index:05d}_{str(in_action).lower()}.jpg' )
        cv2.imwrite(str(frame_path), frame)


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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the frames annotated with whether the baller is in balling action.")
    parser.add_argument("--video_dir", default="videos-aug31st", 
                       help="Directory containing video files and annotations")
    parser.add_argument("--output_dir", default="training_set", 
                       help="Directory to save the extracted frames. The filename includes whether the baller is in balling action.")
    parser.add_argument("--test", default=False, action='store_true',
                       help="If --test is set, then the frames of a single video is processed.")

    args = parser.parse_args()

    write(Path(args.video_dir), Path(args.output_dir), args.test)