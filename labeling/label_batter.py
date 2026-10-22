#!/usr/bin/env python3
"""label_batter.py - Label the striker or the non-striker (one role per run) for every
bowling event by picking one of the SAM3 'person' tracks, on the same frame the bowler
box was hand-labelled.

For each event the bowler box frame is shown with the hand-labelled bowler (green),
the other role if it is already labelled (magenta), and every other SAM3 person track
on that frame, numbered by box size (1 = largest, i.e. usually closest to camera).
One candidate is pre-selected as the guess (yellow): the largest track for --role
striker, the track nearest the bowler for --role non_striker. Confirm it, or pick
another one.

The chosen box is written into videos/<stem>.json under bounding_boxes[<frame>] with
class "striker" / "non_striker" (same format as the bowler box, plus sam3_person_id so
the whole SAM3 track can be pulled later). Progress (incl. 'not visible' verdicts) is
kept per role in --review_path, so the tool is resumable.

    python labeling/label_batter.py --role striker
    python labeling/label_batter.py --role non_striker

Controls:
  click / 1-9    : select a candidate (click picks the smallest box under the cursor)
  Tab            : cycle selection
  Enter / Space  : accept selection as batter, go to next event
  m              : draw the batter box manually (SAM3 missed them), Enter to confirm
  x              : this role is not visible on this event
  s              : skip for now (not recorded, comes back next run)
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
from autolabel.sam3_tracks import bowler_prompts, interpolate_box, iou, load_annotation, open_video  # noqa: E402

REVIEW_PATH_DEFAULT = "labels/batter_review.json"
BOWLER_IOU = 0.3  # person track overlapping the bowler box this much is the bowler, not a candidate

ROLES = ("striker", "non_striker")
GREEN, YELLOW, GREY, CYAN, MAGENTA = (0, 220, 0), (0, 255, 255), (170, 170, 170), (255, 255, 0), (255, 0, 255)


def load_json(path: pathlib.Path) -> dict:
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return {}


def save_json(path: pathlib.Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def labelled_box(anno: dict, frame: int, cls: str) -> Optional[dict]:
    for b in (anno.get("bounding_boxes") or {}).get(str(frame), []):
        if b.get("class") == cls:
            return b
    return None


def candidates_at(persons: dict, frame: int, exclude: list) -> list:
    """[(oid, box)] of person tracks on `frame` not overlapping any `exclude` box
    (bowler, other role), largest box first."""
    out = []
    for oid, t in persons.items():
        box = interpolate_box(t["boxes"], frame)
        if box and all(iou(tuple(box), tuple(e)) < BOWLER_IOU for e in exclude):
            out.append((oid, box))
    out.sort(key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]), reverse=True)
    return out


def initial_guess(role: str, cands: list, bowler: tuple) -> int:
    """Striker: largest track (closest to camera). Non-striker: track nearest the
    bowler, since they stand at the bowler's end."""
    if not cands:
        return -1
    if role == "striker":
        return 0
    bx, by = (bowler[0] + bowler[2]) / 2, (bowler[1] + bowler[3]) / 2
    return min(range(len(cands)), key=lambda i: ((cands[i][1][0] + cands[i][1][2]) / 2 - bx) ** 2
                                                + ((cands[i][1][1] + cands[i][1][3]) / 2 - by) ** 2)


def write_role_box(anno_path: pathlib.Path, frame: int, box: list, person_id: Optional[str], role: str) -> None:
    """Replace any existing `role` box on `frame` in videos/<stem>.json."""
    raw = json.load(open(anno_path, encoding="utf-8"))
    boxes = raw.setdefault("bounding_boxes", {})
    kept = [b for b in boxes.get(str(frame), []) if b.get("class") != role]
    entry = {"x1": int(round(box[0])), "y1": int(round(box[1])),
             "x2": int(round(box[2])), "y2": int(round(box[3])), "class": role}
    if person_id is not None:
        entry["sam3_person_id"] = person_id
    kept.append(entry)
    boxes[str(frame)] = kept
    save_json(anno_path, raw)


def label_event(stem: str, anno: dict, persons: dict, event_index: int, bowler_frame: int,
                bowler: tuple, role: str, max_display: int) -> Optional[dict]:
    """Interactive single-event labelling. Returns a verdict dict, {} for skip, None on quit."""
    cap = open_video(anno["_video_path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    other_role = ROLES[1 - ROLES.index(role)]
    other = labelled_box(anno, bowler_frame, other_role)
    other_box = (other["x1"], other["y1"], other["x2"], other["y2"]) if other else None
    exclude = [bowler] + ([other_box] if other_box else [])
    cands = candidates_at(persons, bowler_frame, exclude)
    sel = initial_guess(role, cands, bowler)
    tag = role.upper().replace("_", "-")

    win = f"Label {role}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {"scale": 1.0, "boxes": [], "img": None, "clicked": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicked"] = (x / state["scale"], y / state["scale"])

    cv2.setMouseCallback(win, on_mouse)

    def draw(img, box, color, label, thick):
        s = state["scale"]
        x1, y1, x2, y2 = [int(v * s) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, label, (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def show(idx: int) -> bool:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            return False
        state["img"] = img
        s = min(1.0, max_display / max(img.shape[:2]))
        state["scale"] = s
        disp = cv2.resize(img, (0, 0), fx=s, fy=s)
        if idx == bowler_frame:
            draw(disp, bowler, GREEN, "bowler", 2)
            if other_box:
                draw(disp, other_box, MAGENTA, other_role, 2)
            frame_cands = cands
        else:  # context frames: draw where the same candidate tracks are now
            frame_cands = [(oid, interpolate_box(persons[oid]["boxes"], idx)) for oid, _ in cands]
        for i, (oid, box) in enumerate(frame_cands):
            if box is None:
                continue
            chosen = i == sel
            draw(disp, box, YELLOW if chosen else GREY, f"{i + 1}" + (f" {tag}?" if chosen else ""),
                 3 if chosen else 1)
        hud = [f"[{tag}]  {stem}  event {event_index}  frame {idx}/{total}"
               + ("" if idx == bowler_frame else f"  (bowler frame {bowler_frame})"),
               "click/1-9 select  Tab cycle  Enter accept  m manual  x none  s skip  q quit"]
        for j, line in enumerate(hud):
            cv2.putText(disp, line, (10, 25 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
        cv2.imshow(win, disp)
        return True

    def pick_at(pt):
        hits = [i for i, (_, b) in enumerate(cands) if b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]]
        if hits:
            return min(hits, key=lambda i: (cands[i][1][2] - cands[i][1][0]) * (cands[i][1][3] - cands[i][1][1]))
        return None

    idx = bowler_frame
    verdict = None
    while show(idx):
        key = cv2.waitKey(50) & 0xFF
        if state["clicked"] is not None:
            hit = pick_at(state["clicked"])
            state["clicked"] = None
            if hit is not None:
                sel = hit
            continue
        if key == 255:
            continue
        if key in (13, 10, ord(' ')):
            if sel >= 0:
                oid, box = cands[sel]
                verdict = {"status": "track", "person_id": oid, "frame": bowler_frame,
                           "box": [round(v, 1) for v in box], "rank": sel + 1,
                           "guess_right": sel == initial_guess(role, cands, bowler)}
                break
        elif key == 9 and cands:  # Tab
            sel = (sel + 1) % len(cands)
        elif ord('1') <= key <= ord('9') and key - ord('1') < len(cands):
            sel = key - ord('1')
        elif key == ord('m'):
            idx = bowler_frame
            show(idx)
            s = state["scale"]
            disp = cv2.resize(state["img"], (0, 0), fx=s, fy=s)
            draw(disp, bowler, GREEN, "bowler", 2)
            if other_box:
                draw(disp, other_box, MAGENTA, other_role, 2)
            r = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
            cv2.setMouseCallback(win, on_mouse)
            if r[2] > 0 and r[3] > 0:
                box = [r[0] / s, r[1] / s, (r[0] + r[2]) / s, (r[1] + r[3]) / s]
                verdict = {"status": "manual", "person_id": None, "frame": bowler_frame,
                           "box": [round(v, 1) for v in box]}
                break
        elif key == ord('x'):
            verdict = {"status": "none", "frame": bowler_frame}
            break
        elif key == ord('s'):
            verdict = {}
            break
        elif key == ord('q'):
            break
        elif key == 81:  # left arrow
            idx = max(0, idx - 1)
        elif key == 83:  # right arrow
            idx = min(total - 1, idx + 1)
    cap.release()
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=ROLES, required=True, help="Which batter to label this run.")
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--review_path", type=pathlib.Path, default=pathlib.Path(REVIEW_PATH_DEFAULT))
    ap.add_argument("--videos", default="",
                    help="Comma-separated video stems to (re-)label, even if already done.")
    ap.add_argument("--max_display", type=int, default=1200, help="Longest display side in pixels.")
    args = ap.parse_args()

    all_reviews = load_json(args.review_path)
    review = all_reviews.setdefault(args.role, {})
    forced = {s.strip() for s in args.videos.split(",") if s.strip()}

    items = []
    for sc_path in sorted(args.out_dir.glob("*.json")):
        stem = sc_path.stem
        if forced and stem not in forced:
            continue
        anno_path = args.video_dir / f"{stem}.json"
        if not anno_path.exists():
            continue
        done = {} if stem in forced else review.get(stem, {})
        anno = load_annotation(anno_path)
        if anno is None:
            continue
        for event_index, frame, box in bowler_prompts(anno):
            if str(event_index) not in done:
                items.append((stem, sc_path, anno_path, event_index, frame, box))
    print(f"{len(items)} events to label as {args.role}")

    sidecar_cache = {}
    for i, (stem, sc_path, anno_path, event_index, frame, bowler) in enumerate(items, 1):
        if stem not in sidecar_cache:
            sidecar_cache.clear()
            sidecar_cache[stem] = json.load(open(sc_path, encoding="utf-8"))
        persons = sidecar_cache[stem].get("person_tracks", {})
        anno = load_annotation(anno_path)  # reload: earlier events may have written to it
        print(f"[{i}/{len(items)}] {stem} event {event_index} frame {frame} ({len(persons)} tracks)")
        verdict = label_event(stem, anno, persons, event_index, frame, bowler, args.role, args.max_display)
        if verdict is None:
            print("Quit, progress saved.")
            break
        if not verdict:
            continue
        if verdict["status"] in ("track", "manual"):
            write_role_box(anno_path, frame, verdict["box"], verdict.get("person_id"), args.role)
        review.setdefault(stem, {})[str(event_index)] = verdict
        save_json(args.review_path, all_reviews)
    cv2.destroyAllWindows()

    counts = {}
    for d in review.values():
        for v in d.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
    guesses = [v["guess_right"] for d in review.values() for v in d.values() if v["status"] == "track"]
    print(f"\n{args.role} total so far: {counts}  -> {args.review_path}")
    if guesses:
        print(f"auto guess was right {sum(guesses)}/{len(guesses)} ({100 * sum(guesses) / len(guesses):.0f}%)")


if __name__ == "__main__":
    main()
