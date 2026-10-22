#!/usr/bin/env python3
"""val_errors.py - Where does a classifier go wrong on the _val clips?

For a stump-crop checkpoint (training_set_stumps/frames, both __L/__R crops) and,
optionally, the fixed-crop baseline (exp21, run on the same clips' full frames):

  * false positives: per crop, on the bowler's side of the nearest delivery or the
    other side, and how many frames from that delivery's labelled window
    (start_frame-2 .. end_frame, the positive window used in training)
  * misses: where in the delivery window the missed frames sit
  * deliveries: the app's trigger (>= --need of the last --window frames over the
    threshold; a stump-crop frame counts if either crop does) replayed per clip:
    deliveries caught, how late, and triggers outside any delivery

    python analysis/val_errors.py --checkpoint experiments/trainings/<round>/<tag>/.../best_model.pth \
        --baseline models/fastvit_sa12_exp21.pt
"""
import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
from autolabel.sam3_tracks import load_annotation  # noqa: E402
from stump_training_set import POS_LEAD, video_events  # noqa: E402

FRAME_RE = re.compile(r"frame_(\d+)_(true|false)\.jpg$")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_model(path: pathlib.Path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "config" in ck:
        cfg = ck["config"]
        name, h, w = cfg["model_name"], cfg["input_height"], cfg["input_width"]
    else:
        a = ck["args"]
        name, h, w = a.model_name, a.input_height, a.input_width
    model = timm.create_model(name, pretrained=False, num_classes=2)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval().to(DEVICE), (h, w), ck.get("args")


@torch.no_grad()
def score(model, images, batch=64):
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack(images[i:i + batch]).to(DEVICE)
        out.append(model(x).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def centre_crop(img: Image.Image, margin: float) -> Image.Image:
    W, H = img.size
    cw, ch = W / (1 + 2 * margin), H / (1 + 2 * margin)
    x1, y1 = (W - cw) / 2, (H - ch) / 2
    return img.crop((round(x1), round(y1), round(x1 + cw), round(y1 + ch)))


def video_fps(path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def fixed_crop(frame):
    """modellib.crop (exp21's input): drop the top 32%, centred square. Copied to avoid modellib's plotting deps."""
    h, w = frame.shape[:2]
    top = int(h * 0.32)
    side = min(h - top, w)
    left = max(w // 2 - side // 2, 0)
    y = max(top + (h - top) // 2 - side // 2, top)
    return frame[y:y + side, left:left + side]


def trigger(frame_scores: np.ndarray, thr: float, window: int, need: int) -> list:
    """Frame indices where the app's sliding-window trigger fires (it resets after firing)."""
    fires, recent = [], []
    for i, s in enumerate(frame_scores):
        recent.append(s >= thr)
        recent = recent[-window:]
        if len(recent) == window and sum(recent) >= need:
            fires.append(i)
            recent = []
    return fires


def dist_to_window(i: int, windows: list) -> int:
    return min(0 if a <= i <= b else (a - i if i < a else i - b) for a, b in windows)


def delivery_report(name, per_clip, thr, window, need, tol):
    """per_clip: {stem: (frame_scores, [(a, b)] positive windows, fps)}"""
    caught, late, fp_trig, n_ev, seconds = 0, [], [], 0, 0.0
    missed = []
    for stem, (fs, wins, fps) in per_clip.items():
        fires = trigger(fs, thr, window, need)
        seconds += len(fs) / fps
        used = set()
        for a, b in wins:
            n_ev += 1
            hit = [f for f in fires if a - tol <= f <= b + tol]
            if hit:
                caught += 1
                late.append(hit[0] - a)
                used.update(hit)
            else:
                missed.append((stem, a, b, float(fs[a:b + 1].max()) if b < len(fs) else None))
        for f in fires:
            if f not in used:
                fp_trig.append((stem, f, dist_to_window(f, wins) if wins else None))
    print(f"\n[{name}] app trigger: >= {need} of last {window} frames over {thr}")
    print(f"  deliveries caught {caught}/{n_ev}"
          + (f", fired at frame {np.median(late):+.0f} (median; range {min(late):+d}..{max(late):+d}) "
             f"relative to start_frame-{POS_LEAD}" if late else ""))
    for m in missed:
        print(f"  MISSED {m[0]} window {m[1]}..{m[2]}  max score in window {m[3]:.2f}")
    print(f"  false triggers: {len(fp_trig)} in {seconds / 60:.1f} min of footage")
    for s, f, d in fp_trig:
        print(f"    {s} frame {f} ({d} frames from the nearest delivery window)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--frames", type=pathlib.Path, default=pathlib.Path("training_set_stumps/frames"))
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--baseline", type=pathlib.Path, default=None, help="Fixed-crop checkpoint, e.g. exp21.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--tol", type=int, default=10, help="Frames around a delivery a trigger may land and still count.")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Save per-crop scores as JSON.")
    args = ap.parse_args()
    torch.set_num_threads(16)

    model, (h, w), _ = load_model(args.checkpoint)
    tf = T.Compose([T.Resize((h, w)), T.ToTensor(), NORM])
    stems = sorted({d.name[:-3] for d in args.frames.iterdir() if d.name.endswith(("__L", "__R"))
                    and d.name[:-3].endswith("_val")})
    print(f"{args.checkpoint}\n{len(stems)} _val clips, on {DEVICE}")

    rows, per_clip = [], {}
    for stem in stems:
        anno = load_annotation(args.video_dir / f"{stem}.json")
        events, _ = video_events(anno)
        wins = [(a - POS_LEAD, b) for a, b, *_ in events]
        sides = {(a - POS_LEAD, b): s for a, b, _, _, s in events}
        fps = video_fps(anno["_video_path"])
        side_scores = {}
        for side, suf in (("left", "__L"), ("right", "__R")):
            files = sorted(args.frames.glob(f"{stem}{suf}/*.jpg"))
            idx = [int(FRAME_RE.search(f.name).group(1)) for f in files]
            labels = [FRAME_RE.search(f.name).group(2) == "true" for f in files]
            probs = score(model, [tf(centre_crop(Image.open(f).convert("RGB"), args.margin)) for f in files])
            side_scores[side] = probs
            for i, y, p in zip(idx, labels, probs):
                nearest = min(wins, key=lambda wd: dist_to_window(i, [wd]))
                rows.append({"stem": stem, "frame": i, "side": side, "label": y, "p": float(p),
                             "bowler_side": side == sides[nearest], "dist": dist_to_window(i, [nearest]),
                             "pos_in_window": i - nearest[0], "window_len": nearest[1] - nearest[0] + 1})
        per_clip[stem] = (np.maximum(side_scores["left"], side_scores["right"]), wins, fps)
        print(f"  {stem}: {len(side_scores['left'])} frames scored")

    thr = args.threshold
    fp = [r for r in rows if not r["label"] and r["p"] >= thr]
    fn = [r for r in rows if r["label"] and r["p"] < thr]
    tp = [r for r in rows if r["label"] and r["p"] >= thr]
    print(f"\n[stump crops] per crop at {thr}: TP {len(tp)}  FP {len(fp)}  FN {len(fn)}  "
          f"precision {len(tp) / max(1, len(tp) + len(fp)):.3f}  recall {len(tp) / max(1, len(tp) + len(fn)):.3f}")
    print("  false positives by side and distance from the delivery window (frames):")
    buckets = [(0, 0, "inside window"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 30, "11-30"), (31, 10**9, ">30")]
    for bs in (True, False):
        sub = [r for r in fp if r["bowler_side"] == bs]
        c = Counter(next(n for lo, hi, n in buckets if lo <= r["dist"] <= hi) for r in sub)
        print(f"    {'bowler side' if bs else 'other side '}: {len(sub):3d}  " +
              "  ".join(f"{n}: {c.get(n, 0)}" for _, _, n in buckets))
    by_clip = Counter(r["stem"] for r in fp)
    print("  false positives per clip:", dict(by_clip.most_common()))
    print("  misses by position in the window (0 = start_frame-2):")
    pos = Counter((r["pos_in_window"], r["window_len"]) for r in fn)
    rel = Counter("first 3 frames" if r["pos_in_window"] < 3 else
                  "last 2 frames" if r["pos_in_window"] >= r["window_len"] - 2 else "middle" for r in fn)
    print(f"    {dict(rel)}")
    per_ev = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["label"]:
            per_ev[(r["stem"], r["frame"] - r["pos_in_window"])][0] += 1
            per_ev[(r["stem"], r["frame"] - r["pos_in_window"])][1] += r["p"] < thr
    print("    missed frames per delivery:", {f"{s}@{a}": f"{m}/{n}" for (s, a), (n, m) in sorted(per_ev.items()) if m})
    delivery_report("stump crops", per_clip, thr, args.window, args.need, args.tol)

    if args.baseline:
        import cv2
        bmodel, (bh, bw), _ = load_model(args.baseline)
        btf = T.Compose([T.Resize((bh, bw)), T.ToTensor(), NORM])
        b_clip, b_rows = {}, []
        for stem in stems:
            anno = load_annotation(args.video_dir / f"{stem}.json")
            wins = [(e["start_frame"] - POS_LEAD, e["end_frame"]) for e in anno["temporal_events"]]
            cap = cv2.VideoCapture(str(anno["_video_path"]))  # as training_set.py decodes for exp21
            imgs, chunks = [], []
            while True:
                ok, f = cap.read()
                if ok:
                    imgs.append(btf(Image.fromarray(cv2.cvtColor(fixed_crop(f), cv2.COLOR_BGR2RGB))))
                if len(imgs) == 256 or (not ok and imgs):  # score as we decode; whole clips don't fit in memory
                    chunks.append(score(bmodel, imgs))
                    imgs = []
                if not ok:
                    break
            cap.release()
            probs = np.concatenate(chunks)
            fps = video_fps(anno["_video_path"])
            b_clip[stem] = (probs, wins, fps)
            for i, p in enumerate(probs):
                y = any(a <= i <= b for a, b in wins)
                b_rows.append({"stem": stem, "frame": i, "label": y, "p": float(p), "dist": dist_to_window(i, wins)})
        tp = sum(r["label"] and r["p"] >= thr for r in b_rows)
        fpb = [r for r in b_rows if not r["label"] and r["p"] >= thr]
        fnb = sum(r["label"] and r["p"] < thr for r in b_rows)
        print(f"\n[baseline {args.baseline.name}] per frame at {thr}: TP {tp}  FP {len(fpb)}  FN {fnb}  "
              f"precision {tp / max(1, tp + len(fpb)):.3f}  recall {tp / max(1, tp + fnb):.3f}")
        c = Counter(next(n for lo, hi, n in buckets if lo <= r["dist"] <= hi) for r in fpb)
        print("  false positives by distance:", "  ".join(f"{n}: {c.get(n, 0)}" for _, _, n in buckets))
        delivery_report(f"baseline {args.baseline.name}", b_clip, thr, args.window, args.need, args.tol)

    if args.out:
        json.dump(rows, open(args.out, "w"))


if __name__ == "__main__":
    main()
