#!/usr/bin/env python3
"""
video_scan.py - Scan a video, classify each frame, and find the start
of a run of at least 4 consecutive action frames. Outputs two frame
numbers: one half a second before X and one two seconds after X.
"""

import argparse
from pathlib import Path
from typing import Iterator, Optional, Tuple, TypeAlias
import numpy as np

import cv2
from PIL import Image

from test import (
    load_model,
    get_transforms,
    predict_is_in_action,
    Device,
    path,
)

Frame: TypeAlias = np.ndarray 


def crop(frame: Frame) -> Frame:
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



def predictions(video_path: Path, model, transform, device: Device) -> Iterator[bool]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    found_action = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = Image.fromarray(crop(frame))
            p = predict_is_in_action(model, image, transform, device)
            if p:
                found_action = True
            yield p
    finally:
        cap.release()
    print(f"found an action? {found_action}")


def find_start_of_run(flags: Iterator[bool], min_run: int = 4) -> Optional[int]:
    run_start: Optional[int] = None
    run_len = 0
    idx = 0
    for flag in flags:
        if flag:
            if run_len == 0:
                run_start = idx
            run_len += 1
            if run_len >= min_run:
                return run_start if run_start is not None else idx
        else:
            run_start = None
            run_len = 0
        idx += 1
    return None


def compute_window_frames(frame_index: int, fps: float, total_frames: int) -> Tuple[int, int]:
    """Returns two frame indexes 0.5 seconds before, and 2 secondss after the detected start of bowling."""
    half_sec = max(1, int(round(0.5 * fps)))
    two_sec = max(1, int(round(2.0 * fps)))
    before = max(0, frame_index - half_sec)
    after = min(total_frames - 1, frame_index + two_sec)
    return before, after


def get_video_metadata(video_path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not determine FPS from video: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Could not determine total frame count from video: {video_path}")
    return float(fps), int(total_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan video for action start frame")
    parser.add_argument("video", type=path, help="Path to input video")
    parser.add_argument("model_path", type=Path, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=Device.from_string, default=Device.CPU, choices=list(Device), help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    model, input_height, input_width = load_model(args.model_path, args.device)
    fps, total_frames = get_video_metadata(args.video)
    print(fps, total_frames, total_frames/fps)
    ts = get_transforms(input_height, input_width)
    flags_iter = predictions(args.video, model, ts, args.device)
    frame_index = find_start_of_run(flags_iter, min_run=4)
    msg = "index of the frame 0.5s before bowling: %s %s"
    if frame_index is None:
        print(msg % (-1, 1))
        return

    before, after = compute_window_frames(frame_index, fps, total_frames)
    print(msg %  (before,after))


if __name__ == "__main__":
    main()


