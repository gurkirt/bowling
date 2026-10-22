#!/usr/bin/env python3
"""label_stumps.py - Confirm or draw the bowler's-end (non-striker's end) stumps for
every bowling event, on the frame the bowler box was hand-labelled.

The proposal (yellow) is the stump box already confirmed for an earlier event of the
same video (the camera rarely moves), else SAM3's "cricket stumps" detection. The two
classifier crops from stump_crops.py are drawn from it (cyan; the one on the bowler's
side is solid), so a bad stump box shows up as a crop that misses the bowler.

Videos with no deliveries (pure negatives) get one item on their middle frame, with
no bowler drawn, so their crops can be written too.

Draw the box tight around the three stumps, base at the ground: the crops are scaled
by its height and anchored on its base.

The box is written into videos/<stem>.json under bounding_boxes[<frame>] with class
"stumps". Progress (incl. 'not visible') is kept in --review_path, so it is resumable.

Controls:
  Enter / Space  : accept the proposal
  m              : draw the stumps manually, Enter to confirm
  x              : stumps not visible on this event
  o              : bowler occluded at release - leave this delivery out of training
                   (no stumps box, so stump_training_set.py drops its frames)
  s              : skip for now (not recorded, comes back next run)
  b              : back to the previous item (shows what you saved there, to fix it)
  Right/Left     : step frame for context (label is always stored on the bowler frame)
  q              : quit and save
"""
import argparse
import json
import pathlib
import sys
from typing import Optional

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
from autolabel.sam3_tracks import bowler_prompts, load_annotation, open_video  # noqa: E402
from labeling.label_batter import load_json, save_json, write_role_box  # noqa: E402
from stump_crops import bowler_side, contains, side_crops  # noqa: E402

REVIEW_PATH_DEFAULT = "labels/stumps_review.json"
GREEN, YELLOW, CYAN, RED = (0, 220, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255)


def sam3_stumps(sidecar: Optional[dict], frame: int) -> Optional[tuple]:
    """SAM3 stump detection from the sample nearest `frame`, with its score."""
    if not sidecar:
        return None
    samples = [s for s in sidecar.get("stumps", {}).get("samples", []) if s.get("detections")]
    if not samples:
        return None
    s = min(samples, key=lambda s: abs(s["frame"] - frame))
    d = max(s["detections"], key=lambda d: d["score"])
    return tuple(d["box"]), d["score"]


def label_event(cap, stem: str, anno: dict, event_index: int, frame: int, bowler: tuple,
                proposal: Optional[tuple], source: str, max_display: int) -> Optional[dict]:
    """Interactive single-event labelling. Returns a verdict dict, {} for skip,
    {"back": True} to revisit the previous item, None on quit."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = "Label stumps"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {"scale": 1.0, "img": None}

    def rect(img, box, color, thick, label=""):
        s = state["scale"]
        x1, y1, x2, y2 = [int(v * s) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if label:
            cv2.putText(img, label, (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def overlay(disp, stumps):
        if bowler:
            rect(disp, bowler, GREEN, 2, "bowler")
        if stumps is None:
            return
        rect(disp, stumps, YELLOW, 2, "stumps")
        side = bowler_side(stumps, bowler) if bowler else None
        for name, crop in zip(("left", "right"), side_crops(stumps)):
            on_side = name == side
            color = (CYAN if contains(crop, bowler) else RED) if on_side else CYAN
            rect(disp, crop, color, 2 if on_side else 1, f"{name} crop" if on_side else "")

    def show(idx: int) -> bool:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            return False
        state["img"] = img
        s = min(1.0, max_display / max(img.shape[:2]))
        state["scale"] = s
        disp = cv2.resize(img, (0, 0), fx=s, fy=s)
        overlay(disp, proposal)
        hud = [f"{stem}  " + (f"event {event_index}" if bowler else "no deliveries") + f"  frame {idx}/{total}"
               + ("" if idx == frame else f"  (bowler frame {frame})"),
               f"proposal: {source}" if proposal else "no proposal - press m to draw",
               "Enter accept  m draw  x none  o occluded  s skip  b back  arrows step  q quit"]
        for j, line in enumerate(hud):
            cv2.putText(disp, line, (10, 25 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
        cv2.imshow(win, disp)
        return True

    idx = frame
    verdict = None
    while show(idx):
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, ord(' ')) and proposal:
            verdict = {"status": source.split()[0], "frame": frame, "box": [round(v, 1) for v in proposal]}
            break
        elif key == ord('m'):
            idx = frame
            show(idx)
            s = state["scale"]
            disp = cv2.resize(state["img"], (0, 0), fx=s, fy=s)
            if bowler:
                rect(disp, bowler, GREEN, 2, "bowler")
            r = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
            if r[2] > 0 and r[3] > 0:
                proposal = (r[0] / s, r[1] / s, (r[0] + r[2]) / s, (r[1] + r[3]) / s)
                source = "manual (Enter to accept, m to redraw)"
        elif key == ord('x'):
            verdict = {"status": "none", "frame": frame}
            break
        elif key == ord('o') and bowler:
            verdict = {"status": "occluded", "frame": frame}
            break
        elif key == ord('s'):
            verdict = {}
            break
        elif key == ord('b'):
            verdict = {"back": True}
            break
        elif key == ord('q'):
            break
        elif key == 81:  # left arrow
            idx = max(0, idx - 1)
        elif key == 83:  # right arrow
            idx = min(total - 1, idx + 1)
    return verdict


def clear_stumps(anno_path: pathlib.Path, frame: int) -> None:
    """Remove the stumps box on `frame` (an event re-marked as not visible)."""
    raw = json.load(open(anno_path, encoding="utf-8"))
    boxes = raw.get("bounding_boxes") or {}
    kept = [b for b in boxes.get(str(frame), []) if b.get("class") != "stumps"]
    if kept:
        boxes[str(frame)] = kept
    else:
        boxes.pop(str(frame), None)
    save_json(anno_path, raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--review_path", type=pathlib.Path, default=pathlib.Path(REVIEW_PATH_DEFAULT))
    ap.add_argument("--videos", default="", help="Comma-separated video stems to (re-)label, even if already done.")
    ap.add_argument("--max_display", type=int, default=1200, help="Longest display side in pixels.")
    args = ap.parse_args()

    review = load_json(args.review_path)
    forced = {s.strip() for s in args.videos.split(",") if s.strip()}

    items = []
    for anno_path in sorted(args.video_dir.glob("*.json")):
        stem = anno_path.stem
        if forced and stem not in forced:
            continue
        anno = load_annotation(anno_path)
        if anno is None:
            continue
        done = {} if stem in forced else review.get(stem, {})
        boxes = anno.get("bounding_boxes") or {}

        def in_json(frame):
            return any(b.get("class") == "stumps" for b in boxes.get(str(frame), []))
        for event_index, frame, bowler in bowler_prompts(anno):
            v = done.get(str(event_index))
            # Done only if the annotation agrees with the review: a box the review
            # remembers but the JSON lost (overwritten elsewhere) is asked again.
            if v and v.get("box") and not in_json(frame):
                print(f"  {stem} event {event_index}: reviewed but no stumps box in the JSON - asking again")
                v = None
            if not v:
                items.append((stem, anno_path, event_index, frame, bowler))
        v = done.get("none")
        if v and v.get("box") and not in_json(v["frame"]):
            print(f"  {stem}: reviewed but no stumps box in the JSON - asking again")
            v = None
        if not anno.get("temporal_events") and not v:
            cap = open_video(anno["_video_path"])
            middle = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            cap.release()
            items.append((stem, anno_path, "none", middle, None))
    print(f"{len(items)} events to label")

    caps = {}  # one capture per video: seeks decode forward, so reopening per event would restart at frame 0
    i = 0
    while i < len(items):
        stem, anno_path, event_index, frame, bowler = items[i]
        sc_path = args.out_dir / f"{stem}.json"
        sidecar = json.load(open(sc_path, encoding="utf-8")) if sc_path.exists() else None
        saved = review.get(stem, {}).get(str(event_index))
        earlier = [v for k, v in review.get(stem, {}).items() if v.get("box") and k != str(event_index)]
        sam = sam3_stumps(sidecar, frame)
        if saved and saved.get("box"):
            proposal, source = tuple(saved["box"]), "saved earlier (Enter keeps it, m redraws, x none)"
        elif earlier:
            proposal, source = tuple(earlier[-1]["box"]), "previous event of this video"
        elif sam:
            proposal, source = sam[0], f"sam3 (score {sam[1]:.2f})"
        else:
            proposal, source = None, ""
        anno = load_annotation(anno_path)
        print(f"[{i + 1}/{len(items)}] {stem} event {event_index} frame {frame}  proposal: {source or 'none'}"
              + (f"  (saved: {saved['status']})" if saved else ""))
        if stem not in caps:
            for c in caps.values():
                c.release()
            caps = {stem: open_video(anno["_video_path"])}
        verdict = label_event(caps[stem], stem, anno, event_index, frame, bowler, proposal, source, args.max_display)
        if verdict is None:
            print("Quit, progress saved.")
            break
        if verdict.get("back"):
            i = max(0, i - 1)
            continue
        i += 1
        if not verdict:
            continue
        if verdict.get("status") == "saved":  # accepted unchanged: keep how it was first labelled
            verdict["status"] = saved["status"]
        if verdict.get("box"):
            write_role_box(anno_path, frame, verdict["box"], None, "stumps")
        else:
            clear_stumps(anno_path, frame)
        review.setdefault(stem, {})[str(event_index)] = verdict
        save_json(args.review_path, review)
    for c in caps.values():
        c.release()
    cv2.destroyAllWindows()

    from collections import Counter
    print(f"\nTotal so far: {dict(Counter(v['status'] for d in review.values() for v in d.values()))}"
          f"  -> {args.review_path}")


if __name__ == "__main__":
    main()
