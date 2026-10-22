#!/usr/bin/env python3
"""crop_extent.py - How much of the stump crop does the bowler actually use?

For every delivery with hand-confirmed stumps and a bowler box, picks the SAM3 person
track that matches the bowler box and follows it over the training positive window
(start_frame-2 .. end_frame). Each box is expressed in stump heights relative to the
stump centre-x and base-y, mirrored so the bowler is always left of the stumps:

    outer  how far the bowler reaches away from the stumps
    inner  how far past the stump centre it reaches (negative: stays short of it)
    above  how far above the stump base its top is
    below  how far below the stump base its bottom is

Prints the distribution of each edge's worst frame per delivery and how many
deliveries a candidate geometry keeps fully inside, exactly and under a runtime stump
calibration error, with the bowler's share of the crop and the pixel cost.

    python analysis/crop_extent.py
"""
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import stump_crops  # noqa: E402
from autolabel.sam3_tracks import bowler_prompts, interpolate_box, iou  # noqa: E402
from stump_crops import bowler_side, stumps_for_event  # noqa: E402

POS_LEAD = 2
MATCH_IOU = 0.5


def deliveries():
    """[(per-frame [outer, inner, above, below] rows, stump height px)] per delivery."""
    out, unmatched = [], 0
    for path in sorted(glob.glob("videos/*.json")):
        anno = json.load(open(path, encoding="utf-8"))
        sc = pathlib.Path("sam3_out") / f"{pathlib.Path(path).stem}.json"
        if not sc.exists():
            continue
        tracks = json.load(open(sc)).get("person_tracks", {})
        for ev_i, frame, bowler in bowler_prompts(anno):
            st = stumps_for_event(anno, frame)
            if st is None:
                continue
            ev = anno["temporal_events"][ev_i]
            best = max(((iou(b, bowler), t) for t in tracks.values()
                        if (b := interpolate_box(t["boxes"], frame))), default=(0, None), key=lambda x: x[0])
            if best[0] < MATCH_IOU:
                unmatched += 1
                continue
            h = st[3] - st[1]
            cx, base = (st[0] + st[2]) / 2, st[3]
            mirror = bowler_side(st, bowler) == "right"
            rows = []
            for f in range(ev["start_frame"] - POS_LEAD, ev["end_frame"] + 1):
                b = interpolate_box(best[1]["boxes"], f)
                if not b:
                    continue
                x1, x2 = ((cx - b[2]) / h, (cx - b[0]) / h) if not mirror else ((b[0] - cx) / h, (b[2] - cx) / h)
                # in "bowler left of stumps" coordinates: distance from centre, positive = away
                rows.append([x2, -x1, (base - b[1]) / h, (b[3] - base) / h])
            if rows:
                out.append((np.array(rows), h))
    return out, unmatched


def contained(rows, g, scale=1.0, dx=0.0, dy=0.0):
    """All frames inside geometry g=(outer, inner, above, below) when the stump estimate
    is `scale` x the true height and shifted by (dx, dy) true stump heights."""
    o, i, a, b = (v * scale for v in g)
    return bool(np.all((rows[:, 0] <= o + dx) & (rows[:, 1] <= i - dx)
                       & (rows[:, 2] <= a - dy) & (rows[:, 3] <= b + dy)))


def report(name, g, dels):
    exact = np.mean([contained(r, g) for r, _ in dels])
    jit = [(s, dx, dy) for s in (0.9, 1.0, 1.1) for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1)]
    robust = np.mean([all(contained(r, g, *j) for j in jit) for r, _ in dels])
    w, hgt = g[0] + g[1], g[2] + g[3]
    share = np.median([np.median((r[:, 0] + r[:, 1]) * (r[:, 2] + r[:, 3])) / (w * hgt) for r, _ in dels])
    px = round(208 * 320 * (w * hgt) / (2.6 * 4.0))  # same pixels per stump height as now
    print(f"  {name:34s} {w:.2f} x {hgt:.2f} h (aspect {hgt / w:.2f})  inside: exact {100 * exact:5.1f}%  "
          f"+-10%/0.1h {100 * robust:5.1f}%  bowler share {share:.2f}  pixels {px:6d} ({px / 65536:.2f}x exp21 per crop)")


def main() -> None:
    dels, unmatched = deliveries()
    print(f"{len(dels)} deliveries with stumps and a matching SAM3 bowler track "
          f"({unmatched} without a track at IoU >= {MATCH_IOU})\n")
    worst = np.array([[r[:, 0].max(), r[:, 1].max(), r[:, 2].max(), r[:, 3].max()] for r, _ in dels])
    print("worst frame per delivery, in stump heights   p50    p90    p95    p99    max")
    for j, n in enumerate(("outer (away from stumps)", "inner (past stump centre)", "above base", "below base")):
        q = np.percentile(worst[:, j], [50, 90, 95, 99, 100])
        print(f"  {n:28s}  " + "  ".join(f"{v:5.2f}" for v in q))
    allr = np.vstack([r for r, _ in dels])
    print(f"\nbowler size over all frames: width p50 {np.median(allr[:, 0] + allr[:, 1]):.2f} h, "
          f"height p50 {np.median(allr[:, 2] + allr[:, 3]):.2f} h")

    cur = (stump_crops.OUTER, stump_crops.INNER, stump_crops.ABOVE, stump_crops.BELOW)
    print("\ncandidate geometries (outer, inner, above, below):")
    report(f"current {cur}", cur, dels)
    for g in [(2.3, 0.0, 3.1, 0.9), (2.3, 0.1, 3.1, 0.9), (2.2, 0.1, 3.0, 0.8),
              (2.1, 0.1, 2.9, 0.7), (2.0, 0.0, 2.8, 0.6)]:
        report(str(g), g, dels)


if __name__ == "__main__":
    main()
