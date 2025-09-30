# training_set offers the write() function to extract frames from videos and read() function to read the extracted frames.

import argparse
from typing import Callable, Dict, Iterator, List, Tuple, TypeAlias, Optional
import numpy as np
from pathlib import Path
import json
import cv2

Frame: TypeAlias = np.ndarray 

def read(frame_dir: Path) -> Iterator[Tuple[Frame, bool]]:
    for frame_path in sorted(frame_dir.glob("*/*.jpg")):  
        frame = cv2.imread(str(frame_path))
        in_action = frame_path.stem.lower().endswith('true')
        yield frame, in_action


def write(video_dir: Path, output_dir: Path, istest: bool) -> None:
    output_dir.mkdir(exist_ok=True)
    for index, content in enumerate(_extract(Path(video_dir), istest)):
        video_path, frame, in_action =  content
        video_frames_dir = output_dir / video_path.stem
        video_frames_dir.mkdir(exist_ok=True)
        framepath = video_frames_dir / Path(f'frame_{index:05d}_{str(in_action).lower()}.jpg' )
        cv2.imwrite(str(framepath), frame)


def _extract(video_dir: Path, istest: bool) -> Iterator[Tuple[Path, Frame, bool]]:
    for video in _glob_videos(video_dir):
        try:
           is_in_action = _is_in_action_maker(video)
        except Exception as e:
            print(f"Warning: Skipping {video.name}: could not read annotation {e}")
            continue

        for index, frame in enumerate(_frames(video)):
            yield video, frame, is_in_action(index)
        
        if istest:
            break


def _glob_videos(video_dir: Path) -> Iterator[Path]:
    for ext in '*.MOV' , '*.mp4', '*.avi':
        paths = sorted(video_dir.glob(ext))
        print('Found {} videos with extension {}'.format(len(paths), ext))
        yield from paths


def _is_in_action_maker(video: Path) -> Callable[[int], bool]:
    annotation = video.with_suffix('.json')
    with open(annotation) as f:
        events = json.load(f)['temporal_events']
    def is_in_action(frame_index: int) -> bool:
        return any(e["start_frame"] - 2 <= frame_index <= e['end_frame'] for e in events)
    return is_in_action
    

def _frames(video: Path) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video.name}: {frame_count} frames")
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