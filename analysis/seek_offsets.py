#!/usr/bin/env python3
"""seek_offsets.py - Does the frame the labeller showed match the frame the pipeline decodes?

video_labeler.py shows frame i with cap.set(CAP_PROP_POS_FRAMES, i); every training-set
builder (and the app) decodes frames in order. On variable-frame-rate clips the seek
lands on a different frame. For every labelled frame (event start/middle/end, box
frames) this seeks like the labeller, finds the identical frame in a sequential decode,
and reports the offset (sequential index - labelled index).

    python analysis/seek_offsets.py --workers 16 --out analysis/seek_offsets.json
"""
import argparse
import json
import pathlib
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np

THUMB = (48, 85)
SEARCH = 150  # frames either side of the labelled index to look for the seeked frame


def thumb(img):
    return cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA).astype(np.int16)


def offsets(anno_path: pathlib.Path):
    anno = json.load(open(anno_path, encoding="utf-8"))
    video = anno_path.parent / anno.get("video", "")
    if not video.is_file():
        return {"stem": anno_path.stem, "skipped": "video missing"}
    labelled = set()
    for ev in anno.get("temporal_events", []):
        labelled.update(ev[k] for k in ("start_frame", "middle_frame", "end_frame") if k in ev)
    labelled.update(int(f) for f in (anno.get("bounding_boxes") or {}))
    if not labelled:
        return {"stem": anno_path.stem, "skipped": "nothing labelled"}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    lo, hi = min(labelled) - SEARCH, max(labelled) + SEARCH
    seq, i = {}, 0
    while i <= hi:
        ok, img = cap.read()
        if not ok:
            break
        if i >= lo:
            seq[i] = thumb(img)
        i += 1
    n_seq = i
    cap.release()

    cap = cv2.VideoCapture(str(video))
    out = {}
    for t in sorted(labelled):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if not ok:
            out[t] = None
            continue
        g = thumb(img)
        cand = sorted((float(np.abs(seq[k] - g).mean()), k) for k in range(t - SEARCH, t + SEARCH + 1) if k in seq)
        (d0, k0), d1 = cand[0], cand[1][0] if len(cand) > 1 else 99.0
        # A clean match is identical (d0 ~ 0) and clearly better than the runner-up.
        out[t] = {"offset": k0 - t, "diff": round(d0, 2), "runner_up": round(d1, 2)}
    cap.release()
    return {"stem": anno_path.stem, "fps": fps, "frames": n_seq, "offsets": out,
            "events": [(ev["start_frame"], ev["end_frame"]) for ev in anno.get("temporal_events", [])]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--videos", default="", help="Comma-separated stems; default all.")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    paths = sorted(args.video_dir.glob("*.json"))
    wanted = {s.strip() for s in args.videos.split(",") if s.strip()}
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(offsets, paths))
    results.sort(key=lambda r: r["stem"])

    done = [r for r in results if "skipped" not in r]
    ev_off, unsure, per_video = [], [], []
    for r in done:
        offs = {int(t): o for t, o in r["offsets"].items()}
        for t, o in offs.items():
            if o is None or o["diff"] > 0.5 or o["runner_up"] - o["diff"] < 0.5:
                unsure.append((r["stem"], t, o))
        starts = [offs[a]["offset"] for a, _ in r["events"] if offs.get(a)]
        ev_off += starts
        vals = [o["offset"] for o in offs.values() if o]
        per_video.append((r["stem"], r["fps"], min(vals, default=0), max(vals, default=0)))

    print(f"{len(done)} videos checked, {len(results) - len(done)} skipped")
    c = Counter(ev_off)
    print(f"delivery start offsets (decoded index - labelled index), {len(ev_off)} deliveries:")
    print("  " + "  ".join(f"{k:+d}: {c[k]}" for k in sorted(c)))
    print(f"  |offset| <= 1: {sum(abs(o) <= 1 for o in ev_off)}   2-4: {sum(2 <= abs(o) <= 4 for o in ev_off)}   "
          f">= 5: {sum(abs(o) >= 5 for o in ev_off)}")
    bad = [v for v in per_video if v[2] != 0 or v[3] != 0]
    print(f"\nvideos with any non-zero offset: {len(bad)} / {len(per_video)}")
    for stem, fps, lo, hi in sorted(bad, key=lambda v: -max(abs(v[2]), abs(v[3]))):
        print(f"  {stem:45s} fps {fps:7.3f}  offsets {lo:+d} .. {hi:+d}")
    if unsure:
        print(f"\n{len(unsure)} labelled frames without a clean match (static scene or decode issue):")
        for s, t, o in unsure[:30]:
            print(f"  {s} frame {t}: {o}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
