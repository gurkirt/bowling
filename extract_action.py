#!/usr/bin/env python3
"""
video_scan.py - Scan a video, classify each frame, and find the start
of a run of at least 4 consecutive action frames. Outputs two frame
numbers: one half a second before X and one two seconds after X.
"""

import argparse
from pathlib import Path
from typing import Iterator, Optional, TypeAlias, Callable
import cv2
from PIL import Image

from modellib import (
    Model,
    Device,
    Frame,
    crop,
    Stats,
)
import itertools
import json
import webbrowser



def label_frames(cap: cv2.VideoCapture, is_action: Model, stats: Stats) -> Iterator[tuple[bool, Frame]]:
    """Returns 2-tuples composed of whether the frame is in_action and the frame itself."""
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = Image.fromarray(crop(frame))
        answer,  confidence = is_action(image)
        print(f'index: {index}, answer: {int(answer)}, confidence: {confidence:.4f}')
        stats.add(index, answer, confidence)
        yield answer, frame
        index += 1


def window4(it: Iterator, fillvalue=None) -> Iterator:
    """Returns an iterator that yields the tuples of 4 consecutives elements."""
    WINDOW_SIZE = 4
    peekers = itertools.tee(it, WINDOW_SIZE)
    for i in range(WINDOW_SIZE):
        for _ in range(i):
            next(peekers[i], None)
    return itertools.zip_longest(*peekers, fillvalue=fillvalue)


def runs_2sec(labelled_frames: Iterator[tuple[bool, Frame]], fps: float) -> Iterator[Optional[Frame]]:
    """Returns an iterator that of frames. Each action starts with None, then 2 seconds of frames."""
    run_length = max(1, int(round(2.0 * fps)))
    it = window4(labelled_frames, fillvalue=(False, None)) 
    for next4 in it:
        if all(e[0] for e in next4):
            print("\nfound start of bowling")
            yield None
            yield next4[0][1]
            yield from (next4[0][1] for next4 in itertools.islice(it, run_length-1))        


def write_actions(action_frames: Iterator[Optional[Frame]], input_path: Path) -> None:
    """Writes each extracted bowling action in a separate file of 2 seconds."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.release()
    action_count = 0 
    output_path = str(input_path.parent / (input_path.stem + "_{:02d}" + '.mp4v'))
    out = None
    for f in action_frames:
        if f is None:
            if out is not None:
                out.release()
            out = cv2.VideoWriter(output_path.format(action_count), fourcc, fps, (width, height))
        else:
            out.write(f)
    if out is not None:
        out.release()

def extract_fps(video: Path) -> float:
    """Reads the fps out of a video."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"failed to open input video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def main() -> None:
    parser = argparse.ArgumentParser(description="Scan video for action start frame")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("model", type=Path, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=Device.from_string, default=Device.CPU, choices=list(Device), help="Device to use (cpu/cuda)")
    args = parser.parse_args()
 
    is_action = Model(args.model, args.device)
    with open(args.video.with_suffix('.json')) as f:
        events = [(e["start_frame"], e["end_frame"]) for e in json.load(f)['temporal_events']]

    stats = Stats(args.video, events)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"failed to open input video: {args.video}")
    frames_and_labels = label_frames(cap, is_action, stats)
    fps = extract_fps(args.video)
    write_actions(runs_2sec(frames_and_labels, fps), args.video)
    cap.release()
    chart_path = "chart.html"
    stats.to_chart(chart_path)
    print(f"Created an SVG interactive chart of the prediction and correct values in {chart_path}. Open your browser on this file.")

if __name__ == "__main__":
    main()


