#!/usr/bin/env python3
"""review_bowler_iou.py - Manually mark low best_person_iou events as 'good' (the
SAM3 person track is actually fine, IoU is just loose/tight-box mismatch) or 'bad'
(genuinely wrong -- candidate for re-labelling or a --passes bowler re-run), so
Phase 1 QC ends with a concrete keep/relabel list instead of only a numeric cutoff.

Controls:
  Right/Left : step frame forward/back (look at neighbouring frames for context)
  g          : mark GOOD, move to next event
  b          : mark BAD, move to next event
  q          : quit and save progress (resumable -- already-reviewed events are skipped)
"""
import argparse
import json
import pathlib
from typing import Optional

import cv2

from sam3_tracks import load_annotation, interpolate_box, open_video

REVIEW_PATH_DEFAULT = "labels/sam3_review.json"


def load_review(path: pathlib.Path) -> dict:
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return {}


def save_review(path: pathlib.Path, review: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=1)


def collect_low_iou_events(video_dir: pathlib.Path, out_dir: pathlib.Path,
                           threshold: float, review: dict, force_videos: set = frozenset()) -> list:
    items = []
    for sc_path in sorted(out_dir.glob("*.json")):
        sidecar = json.load(open(sc_path, encoding="utf-8"))
        stem = sc_path.stem
        forced = stem in force_videos
        eval_rows = sidecar.get("eval", {}).get("per_event", [])
        if not eval_rows:
            continue
        anno_path = video_dir / f"{stem}.json"
        if not anno_path.exists():
            continue
        already = {} if forced else review.get(stem, {})
        for row in eval_rows:
            if not forced and row["best_person_iou"] >= threshold:
                continue
            if str(row["event_index"]) in already:
                continue
            items.append((stem, anno_path, sidecar, row))
    return items


def review_event(stem: str, sidecar: dict, anno: dict, row: dict) -> Optional[str]:
    """Interactive single-event review. Returns 'good'/'bad', or None on quit."""
    cap = open_video(anno["_video_path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = row["frame"]
    gt = None
    for b in (anno.get("bounding_boxes") or {}).get(str(frame_idx), []):
        if b.get("class", "bowler") == "bowler":
            gt = (b["x1"], b["y1"], b["x2"], b["y2"])
            break
    persons = sidecar.get("person_tracks", {})
    best_oid = row.get("best_person_id")

    win = "Review bowler IoU"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    window_sized = False

    def show(idx: int) -> bool:
        nonlocal window_sized
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            return False
        scale = min(1.0, 1400 / max(img.shape[:2]))
        disp = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if not window_sized:
            cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
            window_sized = True
        if gt:
            x1, y1, x2, y2 = [int(v * scale) for v in gt]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, "GT", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if best_oid is not None:
            box = interpolate_box(persons.get(str(best_oid), {}).get("boxes", {}), idx)
            if box:
                x1, y1, x2, y2 = [int(v * scale) for v in box]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(disp, f"p{best_oid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(disp, f"{stem}  event {row['event_index']}  frame {idx}/{total}  "
                          f"iou={row['best_person_iou']}  best_id={best_oid}  "
                          f"n_people={row['n_people_on_frame']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(disp, "g=good  b=bad  arrows=step frame  q=quit&save",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow(win, disp)
        return True

    idx = frame_idx
    verdict = None
    while show(idx):
        key = cv2.waitKey(0) & 0xFF
        if key == ord('g'):
            verdict = "good"
            break
        if key == ord('b'):
            verdict = "bad"
            break
        if key == ord('q'):
            break
        if key == 81:  # left arrow
            idx = max(0, idx - 1)
        elif key == 83:  # right arrow
            idx = min(total - 1, idx + 1)
    cap.release()
    cv2.destroyWindow(win)
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--review_path", type=pathlib.Path, default=pathlib.Path(REVIEW_PATH_DEFAULT))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--videos", default="",
                    help="Comma-separated video stems to force re-review (e.g. IMG_8538,IMG_8352), "
                         "regardless of IoU threshold or prior 'good'/'bad' verdict -- for re-checking "
                         "clips after a rotation/tracking fix.")
    args = ap.parse_args()

    review = load_review(args.review_path)
    force_videos = {s.strip() for s in args.videos.split(",") if s.strip()}
    items = collect_low_iou_events(args.video_dir, args.out_dir, args.threshold, review, force_videos)
    print(f"{len(items)} events to review (threshold={args.threshold}, forced={sorted(force_videos)})")

    for i, (stem, anno_path, sidecar, row) in enumerate(items, 1):
        anno = load_annotation(anno_path)
        if anno is None:
            continue
        print(f"[{i}/{len(items)}] {stem} event {row['event_index']} iou={row['best_person_iou']}")
        verdict = review_event(stem, sidecar, anno, row)
        if verdict is None:
            print("Quit, progress saved.")
            break
        review.setdefault(stem, {})[str(row["event_index"])] = verdict
        save_review(args.review_path, review)

    n_good = sum(v == "good" for d in review.values() for v in d.values())
    n_bad = sum(v == "bad" for d in review.values() for v in d.values())
    print(f"\nTotal reviewed so far: good={n_good} bad={n_bad}  -> {args.review_path}")


if __name__ == "__main__":
    main()
