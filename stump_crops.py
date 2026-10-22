#!/usr/bin/env python3
"""stump_crops.py - Two classifier crops anchored on the bowler's-end stumps.

The camera stands where the umpire would, so the bowler releases right beside the
near (non-striker's end) stumps: left of them over the wicket, right of them round
the wicket, never straddling. Measured in stump heights from the stump centre-x and
base-y, the bowler at release is 1.8-3.1 tall, its outer edge within 2.05, its
head at most 3.0 above the base. So one crop per side, sized in stump heights, holds the
bowler at a consistent scale regardless of zoom. The right crop is mirrored so both
look like "bowler left of stumps" and one classifier scores both in a batch of two.

The stumps are static for a session, so at runtime they are found once (tap or a
one-shot detector) and the two crops are fixed rectangles after that.

    python stump_crops.py            # containment check against current labels
"""
from typing import Optional, Tuple

Rect = Tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels

# In stump heights, relative to (stump centre-x, stump base-y); left-side crop.
# Sized on 313 hand-confirmed stump boxes against the SAM3 bowler track over every
# event frame (start-2 .. end), not just the release frame: the bowler is fully inside
# for 100% of events with exact stumps, and 96% under +-10% stump height / +-0.1h shift
# (what a one-shot runtime stump estimate might get wrong). Bowler share of crop: 0.20.
OUTER = 2.3   # how far the crop reaches away from the stumps (worst frame: 2.03)
INNER = 0.3   # how far past the stump centre it reaches (worst frame: 0.03)
ABOVE = 3.1   # above the stump base (worst frame: 3.0)
BELOW = 0.9   # below the stump base (worst frame: 0.85)


def side_crops(stumps: Rect) -> Tuple[Rect, Rect]:
    """(left, right) crop rectangles for a stump box. Unclipped; may leave the frame."""
    h = stumps[3] - stumps[1]
    cx = (stumps[0] + stumps[2]) / 2
    base = stumps[3]
    y1, y2 = base - ABOVE * h, base + BELOW * h
    return (cx - OUTER * h, y1, cx + INNER * h, y2), (cx - INNER * h, y1, cx + OUTER * h, y2)


def bowler_side(stumps: Rect, bowler: Rect) -> str:
    return "left" if (bowler[0] + bowler[2]) / 2 < (stumps[0] + stumps[2]) / 2 else "right"


def contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def crop_image(img, rect: Rect, mirror: bool):
    """Crop `rect` out of a BGR image, padding with black where it leaves the frame."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    h, w = img.shape[:2]
    pad = [max(0, -y1), max(0, y2 - h), max(0, -x1), max(0, x2 - w)]
    out = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if any(pad):
        out = cv2.copyMakeBorder(out, *pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.flip(out, 1) if mirror else out


def stumps_for_event(anno: dict, frame: int) -> Optional[Rect]:
    for b in (anno.get("bounding_boxes") or {}).get(str(frame), []):
        if b.get("class") == "stumps":
            return b["x1"], b["y1"], b["x2"], b["y2"]
    return None


def _check() -> None:
    import glob
    import json
    import pathlib
    n = ok = labelled = 0
    share = []
    for path in sorted(glob.glob("videos/*.json")):
        anno = json.load(open(path, encoding="utf-8"))
        sidecar = pathlib.Path("sam3_out") / f"{pathlib.Path(path).stem}.json"
        sam = None
        if sidecar.exists():
            dets = [d["box"] for s in json.load(open(sidecar)).get("stumps", {}).get("samples", [])
                    for d in s.get("detections", [])]
            sam = sorted(dets, key=lambda b: b[3] - b[1])[len(dets) // 2] if dets else None
        for frame, boxes in (anno.get("bounding_boxes") or {}).items():
            bowler = next(((b["x1"], b["y1"], b["x2"], b["y2"]) for b in boxes if b.get("class") == "bowler"), None)
            stumps = stumps_for_event(anno, int(frame))
            labelled += stumps is not None
            stumps = stumps or sam
            if not bowler or not stumps:
                continue
            left, right = side_crops(stumps)
            crop = left if bowler_side(stumps, bowler) == "left" else right
            n += 1
            ok += contains(crop, bowler)
            share.append((bowler[2] - bowler[0]) * (bowler[3] - bowler[1])
                         / ((crop[2] - crop[0]) * (crop[3] - crop[1])))
    share.sort()
    print(f"{n} bowler events with stumps ({labelled} hand-confirmed, rest SAM3)")
    print(f"bowler fully inside its side crop: {ok}/{n} ({100 * ok / max(n, 1):.1f}%)")
    if share:
        print(f"bowler share of crop area: median {share[len(share) // 2]:.2f}, "
              f"5th pct {share[len(share) // 20]:.2f}  (current fixed crop: median 0.17)")


if __name__ == "__main__":
    _check()
