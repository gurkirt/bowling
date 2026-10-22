#!/usr/bin/env python3
"""stump_training_set.py - Decode every annotated video into the two stump-anchored
crops from stump_crops.py, in the layout dataset.py reads:

    <out>/frames/<stem>__L/frame_<i>_<true|false>.jpg
    <out>/frames/<stem>__R/frame_<i>_<true|false>.jpg   (mirrored)

Labels: a crop is positive only on the bowler's side of the stumps during a delivery
(start_frame - 2 .. end_frame, the same window training_set.py uses); the other side
of the same frame is a negative, as is every crop outside a delivery. The bowler's
side comes from the hand-labelled bowler box vs the hand-confirmed stumps.

Each crop is saved with --margin extra context per side (as a fraction of the crop),
so training can jitter the crop like an imperfect stump calibration would; at
validation and in the app the centre (unmargined) region is the model input. Pixels
outside the frame are black, as they would be in the app.

Each frame uses the stumps of the nearest labelled event of its video, so a camera
that moved between deliveries is followed. Videos with no stumps labelled are skipped;
frames around a delivery with no bowler box (side unknown) are left out of both crops,
and so is a wider span around a delivery marked occluded in label_stumps.py ('o').
Videos with no deliveries are written as all-negative using their labelled stumps.

Also writes <out>/meta.json with the crop geometry, margin and the calibration
overlay (median stump box as a fraction of the frame), for the app.

    python stump_training_set.py --workers 16
"""
import argparse
import json
import pathlib
import statistics
from multiprocessing import Pool

import cv2

import stump_crops
from autolabel.sam3_tracks import bowler_prompts, load_annotation, open_video
from stump_crops import bowler_side, crop_image, side_crops, stumps_for_event

POS_LEAD = 2  # frames before start_frame that count as positive, as in training_set.py
EXCLUDE_PAD = 10  # frames dropped either side of a delivery whose side is unknown
OCCLUDED_PAD_S = 2.0  # seconds dropped either side of a delivery marked occluded


def expand(rect, margin):
    x1, y1, x2, y2 = rect
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def video_events(anno: dict):
    """([(start, end, middle, stumps, side)] for every event with a bowler box and stumps,
    [(start, end)] for events without, whose side is unknown)."""
    usable, used = [], set()
    for event_index, frame, bowler in bowler_prompts(anno):
        stumps = stumps_for_event(anno, frame)
        if stumps is None:
            continue
        ev = anno["temporal_events"][event_index]
        usable.append((ev["start_frame"], ev["end_frame"], frame, stumps, bowler_side(stumps, bowler)))
        used.add(event_index)
    unknown = [(ev["start_frame"], ev["end_frame"]) for i, ev in enumerate(anno.get("temporal_events", []))
               if i not in used]
    return usable, unknown


def write_one(job) -> dict:
    anno_path, frames_dir, margin, size, occluded = job
    anno = load_annotation(anno_path)
    if anno is None:
        return {"stem": anno_path.stem, "skipped": "video missing"}
    events, unknown = video_events(anno)
    if not events and not anno.get("temporal_events"):
        # No deliveries: every crop is a negative; use the stumps labelled on any frame.
        stumps = next(((f, (b["x1"], b["y1"], b["x2"], b["y2"]))
                       for f, bs in (anno.get("bounding_boxes") or {}).items()
                       for b in bs if b.get("class") == "stumps"), None)
        if stumps:
            events = [(-1, -1 - POS_LEAD - 1, int(stumps[0]), stumps[1], None)]  # never active
    if not events:
        return {"stem": anno_path.stem, "skipped": "no stumps labelled"}

    stem = anno_path.stem
    dirs = {s: frames_dir / f"{stem}__{s[0].upper()}" for s in ("left", "right")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jpg"):  # labels may have changed: a stale frame_i_<label> would duplicate frame i
            old.unlink()
    out_w, out_h = size

    cap = open_video(anno["_video_path"])
    occ_pad = round(OCCLUDED_PAD_S * (cap.get(cv2.CAP_PROP_FPS) or 30.0))
    excluded = [(a - POS_LEAD - EXCLUDE_PAD, b + EXCLUDE_PAD) for a, b in unknown]
    excluded += [(ev["start_frame"] - POS_LEAD - occ_pad, ev["end_frame"] + occ_pad)
                 for i, ev in enumerate(anno.get("temporal_events", [])) if i in occluded]
    i, n_pos, n_dropped, frame_wh = 0, 0, 0, [0, 0]
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frame_wh = [img.shape[1], img.shape[0]]
        # A delivery we can't assign to a side would otherwise become a false negative.
        if any(a <= i <= b for a, b in excluded):
            n_dropped += 1
            i += 1
            continue
        nearest = min(events, key=lambda e: abs(e[2] - i))
        crops = dict(zip(("left", "right"), side_crops(nearest[3])))
        active = [e for e in events if e[0] - POS_LEAD <= i <= e[1]]
        for side in ("left", "right"):
            label = any(e[4] == side for e in active)
            n_pos += label
            patch = crop_image(img, expand(crops[side], margin), mirror=side == "right")
            patch = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dirs[side] / f"frame_{i:05d}_{str(label).lower()}.jpg"), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        i += 1
    cap.release()
    return {"stem": stem, "frames": i - n_dropped, "dropped": n_dropped, "positives": n_pos,
            "events": len(events), "unknown_events": len(unknown), "occluded_events": len(occluded),
            "stumps": [e[3] for e in events], "frame_wh": frame_wh}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("training_set_stumps"))
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Extra context saved per side, as a fraction of the crop size.")
    ap.add_argument("--core_width", type=int, default=208, help="Width of the unmargined crop in saved pixels.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--videos", default="", help="Comma-separated stems to (re)write; default all.")
    ap.add_argument("--stumps_review", type=pathlib.Path, default=pathlib.Path("labels/stumps_review.json"),
                    help="label_stumps.py progress file; deliveries marked 'occluded' there are left out.")
    args = ap.parse_args()

    aspect = (stump_crops.ABOVE + stump_crops.BELOW) / (stump_crops.OUTER + stump_crops.INNER)
    core = (args.core_width, round(args.core_width * aspect))
    size = (round(core[0] * (1 + 2 * args.margin)), round(core[1] * (1 + 2 * args.margin)))
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.video_dir.glob("*.json"))
    wanted = {s.strip() for s in args.videos.split(",") if s.strip()}
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    print(f"{len(paths)} annotations -> {frames_dir}  core {core[0]}x{core[1]}, saved {size[0]}x{size[1]}")

    review = json.load(open(args.stumps_review)) if args.stumps_review.exists() else {}
    occluded = {stem: {int(k) for k, v in d.items() if v.get("status") == "occluded" and k != "none"}
                for stem, d in review.items()}
    jobs = [(p, frames_dir, args.margin, size, occluded.get(p.stem, set())) for p in paths]
    results = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(write_one, jobs):
            results.append(r)
            print(f"  {r['stem']}: " + (f"skipped ({r['skipped']})" if "skipped" in r
                                       else f"{r['frames']} frames, {r['positives']} positive crops"
                                       + (f", {r['dropped']} frames dropped around {r['unknown_events']} "
                                          f"deliveries without stumps ({r['occluded_events']} occluded)"
                                          if r["dropped"] else "")))

    done = [r for r in results if "skipped" not in r]
    # Calibration overlay: where people already put the stumps, as fractions of the frame.
    norm = [((s[0] + s[2]) / 2 / r["frame_wh"][0], s[3] / r["frame_wh"][1], (s[3] - s[1]) / r["frame_wh"][1],
             (s[2] - s[0]) / (s[3] - s[1])) for r in done for s in r["stumps"]]
    overlay = {k: round(statistics.median(v[j] for v in norm), 4)
               for j, k in enumerate(("centre_x", "base_y", "height", "width_over_height"))}
    meta = {"geometry_stump_heights": {"outer": stump_crops.OUTER, "inner": stump_crops.INNER,
                                       "above": stump_crops.ABOVE, "below": stump_crops.BELOW},
            "margin": args.margin, "saved_size": size, "core_size": core,
            "positive_window": f"start_frame-{POS_LEAD} .. end_frame, bowler's side only",
            "right_crop_mirrored": True,
            "calibration_overlay_fraction_of_frame": overlay,
            "videos": len(done), "frames": sum(r["frames"] for r in done),
            "positive_crops": sum(r["positives"] for r in done),
            "dropped_frames": {r["stem"]: r["dropped"] for r in done if r["dropped"]},
            "skipped": {r["stem"]: r["skipped"] for r in results if "skipped" in r}}
    json.dump(meta, open(args.out / "meta.json", "w"), indent=1)
    print(f"\n{meta['videos']} videos, {meta['frames']} frames x 2 crops, {meta['positive_crops']} positive crops, "
          f"{len(meta['skipped'])} skipped")
    print(f"calibration overlay (fraction of frame): {overlay}")


if __name__ == "__main__":
    main()
