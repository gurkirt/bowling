# training_set offers the write() function to extract frames from videos and read() function to read the extracted frames.
# training_set also offers a main program to extract frames from videos in a directory.
# 
# extract frames from the default directory and save them to the default output directory "training_set/".
# $ python3 training_set.py 
#
# Summarizes the training set in the given directory.
# $ python3 training_set.py --summary /mnt/chromeos/removable/usbdisk/training_set/
# Total frames: 22474, In action: 5% (1263), Not in action: 94% (21211).

from typing import Iterator, Tuple, TypeAlias

import argparse
import collections
import cv2
import json
import time
import numpy as np
import pathlib
import shutil
import modellib
import multiprocessing as mp
from multiprocessing import Pool

Frame: TypeAlias = np.ndarray 

def glob_videos(video_dir: pathlib.Path, istest: bool) -> Iterator[pathlib.Path]:
    video_found = False
    for ext in ('*.MOV', '*.mp4', '*.avi'):
        for video in pathlib.Path(video_dir).glob(ext):
            video_found = True
            yield video
            if istest:
                print("test mode: processing a single video and exiting")
                break
    if not video_found:
        raise ValueError("No video files found in the directory: %s." % video_dir)

def write_all(output_dir: pathlib.Path, video_dir: pathlib.Path, istest: bool) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True)
    with Pool(processes=min(8, mp.cpu_count())) as pool:
        pool.starmap(write_one, [(video, frames_dir) 
                                 for video in glob_videos(video_dir, istest)])



def write_one(video: pathlib.Path, frames_dir: pathlib.Path) -> None:
    """
    Decode every frame of a video to frames_dir/<video_stem>/, cropped, with the
    in-action label encoded in the filename. No subsampling happens here so the
    dataset can build multi-frame windows and sample negatives on the fly.
    """
    try:
        events = modellib.read_start_end(video)
    except Exception as e:
        print(f"Warning: Skipping {video.name}: could not read annotation {e}")
        return

    out_dir = frames_dir / video.stem
    out_dir.mkdir(exist_ok=True, parents=True)
    for i, frame in enumerate(_frames(video)):
        in_action = any(start - 2 <= i <= end for (start, end) in events)
        framepath = out_dir / f'frame_{i:05d}_{str(in_action).lower()}.jpg'
        cv2.imwrite(str(framepath), modellib.crop(frame))


def _frames(video: pathlib.Path) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.perf_counter()
    try:
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          yield frame
    finally:
        cap.release()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"extracted frames from {video.name}: {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} fps)")


def _summary(frame_dir: pathlib.Path) -> str:
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
    parser.add_argument("--video_dir", default="videos", type=pathlib.Path,
                       help="Directory containing video files and annotations")
    parser.add_argument("--output_dir", default="training_set", type=pathlib.Path, 
                       help="Directory to save the extracted frames. The filename includes whether the baller is in balling action.")
    parser.add_argument("--test", default=False, action='store_true',
                       help="If --test is set, then the frames of a single video is processed.")
    parser.add_argument("--summary", default=False, action='store_true',
                       help="When --summary is set, the script displays stats on the produced training set.")
    parser.add_argument("--summary_dir", type=pathlib.Path, default=None,
                       help="The frames directory for the summary is the first positional arg.")

    args = parser.parse_args()
    if args.summary:
        if args.summary_dir is None:
            parser.error("When --summary is set, the frames directory must be provided as the first positional arg.")
        print(_summary(args.summary_dir))
        return
    if not args.video_dir:
        parser.error("The video directory does not exist: %s." % args.video_dir)
    write_all(args.output_dir, args.video_dir, args.test)

if __name__ == '__main__':
    main()
