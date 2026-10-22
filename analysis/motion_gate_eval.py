#!/usr/bin/env python3
"""motion_gate_eval.py - Can a near-free motion gate decide when to run the bowling
classifier, without missing deliveries?

Per video, the release zone is a box around the hand-labelled non-striker (who stands
next to where the bowler releases and barely moves between balls). Each frame, that zone
is downsampled to a tiny grayscale patch and differenced against the frame ~1/15 s
earlier. The gate opens when motion exceeds a threshold and stays open for --hold
seconds. Reports, per threshold: fraction of frames the classifier would run on, and
fraction of labelled bowling events whose frames the gate covers.

    python analysis/motion_gate_eval.py --workers 16
"""
import argparse
import json
import pathlib
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from autolabel.sam3_tracks import open_video  # noqa: E402

ROI_W, ROI_H, ROI_UP = 5.0, 2.6, 0.7  # in non-striker heights; holds 95% of bowler boxes
PATCH = (64, 32)                      # gate works on a 64x32 grayscale patch


def release_zone(anno: dict):
    for fr, boxes in sorted((anno.get("bounding_boxes") or {}).items(), key=lambda kv: int(kv[0])):
        for b in boxes:
            if b.get("class") == "non_striker":
                h = b["y2"] - b["y1"]
                cx = (b["x1"] + b["x2"]) / 2
                y1 = b["y1"] - ROI_UP * h
                return cx - ROI_W / 2 * h, y1, cx + ROI_W / 2 * h, y1 + ROI_H * h
    return None


def motion_series(path: pathlib.Path):
    anno = json.load(open(path, encoding="utf-8"))
    zone = release_zone(anno)
    video = path.parent / anno["video"]
    if zone is None or not video.exists():
        return None
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lag = max(1, round(fps / 15))
    hist, motion = [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in zone]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        g = cv2.cvtColor(cv2.resize(img[y1:y2, x1:x2], PATCH, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.int16)
        hist.append(g)
        motion.append(float(np.mean(np.abs(g - hist[-1 - lag]) > 18)) if len(hist) > lag else 0.0)
        if len(hist) > lag + 1:
            hist.pop(0)
    cap.release()
    events = [(e["start_frame"], e["end_frame"]) for e in anno.get("temporal_events", [])]
    return {"stem": path.stem, "fps": fps, "motion": motion, "events": events}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hold", type=float, default=1.0, help="Seconds the gate stays open after motion.")
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("analysis/motion_gate_cache.json"))
    args = ap.parse_args()

    if args.cache.exists():
        series = json.load(open(args.cache))
    else:
        paths = sorted(args.video_dir.glob("*.json"))
        with Pool(args.workers) as pool:
            series = [s for s in pool.imap_unordered(motion_series, paths) if s]
        json.dump(series, open(args.cache, "w"))
    n_ev = sum(len(s["events"]) for s in series)
    n_fr = sum(len(s["motion"]) for s in series)
    print(f"{len(series)} videos with a non-striker label, {n_ev} events, {n_fr} frames\n")
    print(" thresh | classifier runs on | events fully covered | events >=half covered")
    for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        on_frames, full, half = 0, 0, 0
        for s in series:
            m = np.array(s["motion"]) > thr
            hold = int(round(args.hold * s["fps"]))
            gate = np.zeros_like(m)
            idx = np.flatnonzero(m)
            for i in idx:
                gate[i:i + hold + 1] = True
            on_frames += int(gate.sum())
            for a, b in s["events"]:
                cov = gate[a:b + 1].mean() if b < len(gate) else 0
                full += cov == 1.0
                half += cov >= 0.5
        print(f" {thr:6.3f} | {100 * on_frames / n_fr:17.1f}% | {100 * full / n_ev:19.1f}% | {100 * half / n_ev:20.1f}%")


if __name__ == "__main__":
    main()
