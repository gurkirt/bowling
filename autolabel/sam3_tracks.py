"""sam3_tracks.py - Offline SAM 3 label engine for the bowling-action dataset.

Four passes over every annotated clip in videos/:

  bowler  Prompt SAM 3 with the hand-annotated bowler box at each event's
          middle_frame, propagate forward AND backward -> a per-frame bowler
          track covering the whole clip (run-up, release, follow-through).
  person  Text prompt "person" -> every person tracked with stable ids, so the
          non-striker / striker / keeper / fielders become labelled negatives
          instead of noise.
  stumps  Text prompt "cricket stumps" on a few sampled frames. The camera is
          near-static, so a per-video consensus box is enough for calibration.
  ball    Text prompt for the cricket ball (default "ball") on more densely
          sampled frames -- it moves fast, so a handful of stumps-style samples
          would mostly miss it; increase --ball_samples if recall is too low.

videos/*.json is never modified; every result lands in a sidecar under --out_dir.

Boxes are derived from the returned masks rather than read from the output dict,
so the only schema this depends on is "object id -> mask". Run --probe first to
dump the raw structure if a SAM 3 version changes it.

Typical use on the 5090 box:
    python autolabel/sam3_tracks.py --probe --limit 1          # dump output structure
    python autolabel/sam3_tracks.py --limit 5 --viz            # eyeball 5 clips
    python autolabel/sam3_tracks.py                            # full run
    python autolabel/sam3_tracks.py --report                   # stats over the sidecars

Frames are decoded at --target_fps (default 20, downsampled from source) and each
bowler-event window is capped at --max_window_frames, to bound the memory a single
SAM 3 session needs regardless of clip length or source frame rate.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pathlib
import shutil
import sys
import traceback
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in ORIGINAL frame pixels


# --------------------------------------------------------------------------
# annotations
# --------------------------------------------------------------------------

def load_annotation(json_path: pathlib.Path) -> Optional[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    video = anno.get("video")
    if not video:
        return None
    anno["_video_path"] = json_path.parent / video
    if not anno["_video_path"].exists():
        return None
    return anno


def bowler_prompts(anno: dict) -> List[Tuple[int, int, Box]]:
    """(event_index, prompt_frame, box) for every event that has a bowler box.

    The labeller stores one box keyed by the frame it was drawn on, which is the
    event's middle_frame, so match on that first and fall back to any boxed frame
    inside the event's range.
    """
    boxes = anno.get("bounding_boxes", {}) or {}
    out = []
    for i, ev in enumerate(anno.get("temporal_events", [])):
        start, end = ev["start_frame"], ev["end_frame"]
        mid = ev.get("middle_frame", (start + end) // 2)
        candidates = [mid] + [f for f in range(start, end + 1) if f != mid]
        for frame in candidates:
            for b in boxes.get(str(frame), []):
                if b.get("class", "bowler") == "bowler":
                    out.append((i, frame, (float(b["x1"]), float(b["y1"]),
                                           float(b["x2"]), float(b["y2"]))))
                    break
            else:
                continue
            break
    return out


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def box_to_norm_xywh(box: Box, w: int, h: int) -> Tuple[float, float, float, float]:
    """Pixel box -> normalized xywh, clamped into [0, 1]: annotated boxes can be a
    fraction of a pixel outside the frame (edge rounding), and SAM 3 asserts on range.
    """
    x1 = min(max(box[0], 0.0), w)
    y1 = min(max(box[1], 0.0), h)
    x2 = min(max(box[2], 0.0), w)
    y2 = min(max(box[3], 0.0), h)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    norm = (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)
    return tuple(min(max(v, 0.0), 1.0) for v in norm)


def remap_annotation_to_frames(anno: dict, kept: List[int]) -> dict:
    """Copy of anno with temporal_events/bounding_boxes keyed by ORIGINAL video frame
    numbers remapped to indices into the (fps-decimated) frames written by
    extract_frames, i.e. kept[new_index] == original_index. Used to drive SAM 3 on
    the decimated frame set; outputs are mapped back via kept[] before saving.
    """
    def nearest_new(orig_idx: int) -> int:
        pos = bisect.bisect_left(kept, orig_idx)
        cands = [p for p in (pos - 1, pos) if 0 <= p < len(kept)]
        return min(cands, key=lambda p: abs(kept[p] - orig_idx))

    remapped = dict(anno)
    remapped["temporal_events"] = [
        {**ev,
         "start_frame": nearest_new(ev["start_frame"]),
         "end_frame": nearest_new(ev["end_frame"]),
         "middle_frame": nearest_new(ev.get("middle_frame", (ev["start_frame"] + ev["end_frame"]) // 2))}
        for ev in anno.get("temporal_events", [])
    ]
    remapped["bounding_boxes"] = {
        str(nearest_new(int(f))): boxes for f, boxes in (anno.get("bounding_boxes") or {}).items()
    }
    return remapped


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def _detect_rotation_code(video: pathlib.Path) -> Optional[int]:
    """The cv2.rotate() code that turns a RAW (CAP_PROP_ORIENTATION_AUTO-disabled)
    decoded frame into the correctly-oriented one, determined by comparing pixel
    content against a CAP_PROP_ORIENTATION_AUTO-enabled reference decode of the same
    frame -- NOT by trusting a rotation-metadata sign convention (ffprobe's 'rotate'
    tag vs. 'side_data rotation' use opposite sign conventions, and one of them
    disagreed with the actual needed direction on a real file here). Returns None if
    no rotation is needed, or if it can't be determined.
    """
    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_prop is None:
        return None
    cap_ref = cv2.VideoCapture(str(video))
    cap_ref.set(orientation_prop, 1)
    ok_ref, ref = cap_ref.read()
    cap_ref.release()
    cap_raw = cv2.VideoCapture(str(video))
    cap_raw.set(orientation_prop, 0)
    ok_raw, raw = cap_raw.read()
    cap_raw.release()
    if not (ok_ref and ok_raw):
        return None
    rh, rw = ref.shape[:2]
    if raw.shape[:2] == (rh, rw):
        return None  # raw already matches the reference orientation
    ref_patch = cv2.resize(ref, (32, 32)).astype(np.int32)
    for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        cand = cv2.rotate(raw, code)
        if cand.shape[:2] != (rh, rw):
            continue
        cand_patch = cv2.resize(cand, (32, 32)).astype(np.int32)
        if np.abs(cand_patch - ref_patch).mean() < 20:
            return code
    return None


class SequentialSeekCapture:
    """cv2.VideoCapture whose set(CAP_PROP_POS_FRAMES, i) lands on the i-th frame of a
    sequential decode, which is how every training-set builder and the app number frames.

    OpenCV's own seek converts i to a timestamp with the average fps, so on variable-
    frame-rate clips it shows another frame (off by up to 20 in this dataset, see
    analysis/seek_offsets.py). Here a seek decodes forward to i instead, reopening the
    file to go backwards; once seeking starts, the last `cache` frames are kept so
    stepping back a few frames stays instant. Plain sequential reads pass straight through.
    """

    def __init__(self, opener: Callable[[], "cv2.VideoCapture"], cache: int = 64):
        self._opener = opener
        self._cap = opener()
        self._pos = 0          # index of the frame the next decode returns
        self._want = None      # pending seek target
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache
        self._seeking = False

    def _decode(self):
        ok, frame = self._cap.read()
        if ok:
            if self._seeking:
                self._cache[self._pos] = frame.copy()
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._pos += 1
        return ok, frame

    def read(self):
        want, self._want = self._want, None
        if want is None or want == self._pos:
            return self._decode()
        if want in self._cache:
            return True, self._cache[want].copy()
        if want < self._pos:
            self._cap.release()
            self._cap, self._pos = self._opener(), 0
        while self._pos < want:
            if not self._cap.grab():
                return False, None
            self._pos += 1
        return self._decode()

    def grab(self):
        ok, _ = self.read()
        return ok

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self._want if self._want is not None else self._pos
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._want, self._seeking = int(value), True
            return True
        return self._cap.set(prop_id, value)

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        return self._cap.release()


class _RotatingCapture:
    """Wraps a cv2.VideoCapture to rotate every frame read from it and report the
    (swapped) frame dimensions -- used when CAP_PROP_ORIENTATION_AUTO doesn't apply
    consistently across access patterns (see open_video); rotation is applied to
    every frame .read() returns, so it can't depend on sequential vs. seek reads.
    """

    def __init__(self, cap: "cv2.VideoCapture", rotate_code: int):
        self._cap = cap
        self._rotate_code = rotate_code
        self._swap_dims = rotate_code != cv2.ROTATE_180

    def read(self):
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.rotate(frame, self._rotate_code)
        return ok, frame

    def get(self, prop_id):
        if self._swap_dims:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)

    def release(self):
        return self._cap.release()


ROTATION_OVERRIDES_PATH = pathlib.Path("rotation_overrides.json")


def _rotation_override(video: pathlib.Path) -> Optional[int]:
    """Manual escape hatch: {"<video stem>": <degrees clockwise, 0/90/180/270>} in
    ./rotation_overrides.json, for the rare file where automatic detection picks the
    wrong direction -- _detect_rotation_code trusts CAP_PROP_ORIENTATION_AUTO's output
    as ground truth, which can itself be wrong (not just inconsistent) for a specific
    file, and nothing here can verify absolute direction without a human looking at it.
    """
    if not ROTATION_OVERRIDES_PATH.exists():
        return None
    try:
        return json.loads(ROTATION_OVERRIDES_PATH.read_text()).get(video.stem)
    except Exception:
        return None


def open_video(video: pathlib.Path) -> "cv2.VideoCapture":
    """cv2.VideoCapture with rotation applied so returned frames match the correctly
    oriented (portrait, for every clip in this dataset) frame every annotation was
    drawn on. Every place that reads raw frames from a video must go through this,
    or its content ends up rotated relative to boxes/annotations.

    CAP_PROP_ORIENTATION_AUTO gets the direction right when it applies, but was
    observed to apply inconsistently between sequential cap.read() and
    cap.set(CAP_PROP_POS_FRAMES, idx) seeks on some files (write_viz, sequential,
    looked right; a seek-based reviewer showed the same clip rotated wrong). So: use
    _detect_rotation_code to empirically verify (via pixel content, not a metadata
    sign convention) whether/how much rotation is needed, then apply it ourselves via
    _RotatingCapture on every read regardless of access pattern. rotation_overrides.json
    (see _rotation_override) takes priority for files where even that picks the wrong
    direction -- fill it in by eye after spotting a case via sam3_viz.py/review_bowler_iou.py.
    """
    def opener():
        c = cv2.VideoCapture(str(video))
        orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_prop is not None:
            c.set(orientation_prop, 0)  # rotation is applied ourselves below, deterministically
        return c

    cap = SequentialSeekCapture(opener)  # frame i == the i-th sequentially decoded frame, also on seeks

    override_deg = _rotation_override(video)
    if override_deg is not None:
        rot_code = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(override_deg % 360)
        return _RotatingCapture(cap, rot_code) if rot_code is not None else cap

    rot_code = _detect_rotation_code(video)
    if rot_code is not None:
        return _RotatingCapture(cap, rot_code)
    return cap


def extract_frames(video: pathlib.Path, dest: pathlib.Path, max_side: int,
                   target_fps: float = 0) -> Tuple[int, int, int, float, float, List[int]]:
    """Decode the clip to dest/%05d.jpg, downscaled so the long side <= max_side and
    subsampled so the effective rate is <= target_fps (0 = keep every source frame).

    Returns (n_frames, orig_w, orig_h, orig_fps, eff_fps, kept). kept[i] is the
    ORIGINAL video frame index that dest/{i:05d}.jpg came from, so callers can map
    tracked frame indices back to the numbering used in videos/*.json. Boxes are
    always reported back in ORIGINAL pixel coordinates, so the downscale is
    invisible downstream; the fps decimation is undone the same way via kept[].

    Requests auto-rotation per the container's rotation metadata (CAP_PROP_ORIENTATION_AUTO):
    without it, some OpenCV/FFmpeg builds hand back raw (un-rotated) sensor frames for
    portrait iPhone clips, e.g. reporting 1920x1080 for a video whose hand-drawn boxes
    were annotated on a 1080x1920 frame. Callers should cross-check the returned (w, h)
    against ground truth (see verify_frame_orientation) since this flag is best-effort
    and silently has no effect on OpenCV builds that don't support it.
    """
    cap = open_video(video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, round(orig_fps / target_fps)) if target_fps and orig_fps > target_fps else 1
    n, w, h, orig_idx = 0, 0, 0, 0
    kept: List[int] = []
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if orig_idx % stride == 0:
            if n == 0:
                h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(frame.shape[0], frame.shape[1]))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(round(frame.shape[1] * scale)),
                                           int(round(frame.shape[0] * scale))),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept.append(orig_idx)
            n += 1
        orig_idx += 1
    cap.release()
    eff_fps = orig_fps / stride if orig_fps else 0.0
    return n, w, h, orig_fps, eff_fps, kept


def verify_frame_orientation(anno: dict, w: int, h: int, tol: float = 0.1) -> Optional[str]:
    """None if (w, h) looks right, else a short problem description. Ground-truth
    check for the CAP_PROP_ORIENTATION_AUTO best-effort fix above: every source clip
    in this dataset is portrait-recorded, so width > height alone already means
    extract_frames is still handing back raw (un-rotated) sensor frames on this
    OpenCV build -- checked first since it needs no annotation. Where hand-drawn
    boxes exist, also cross-check they actually fit inside (w, h), with a generous
    tolerance (tol): a genuine rotation mismatch swaps w/h and is off by 50-80%, while
    a real clip's annotation can overshoot by a few percent from old video_labeler.py
    scale-factor rounding bugs or a slightly different source export resolution --
    neither is a rotation mismatch, and box_to_norm_xywh clamps it safely anyway, so
    it shouldn't skip the video.
    """
    if w > h:
        return f"expected a portrait clip (height > width) but decoded {w}x{h} (landscape)"
    boxes = [b for bs in (anno.get("bounding_boxes") or {}).values() for b in bs]
    if not boxes:
        return None
    max_x = max(b["x2"] for b in boxes)
    max_y = max(b["y2"] for b in boxes)
    if max_x <= w * (1 + tol) and max_y <= h * (1 + tol):
        return None
    if max_x <= h * (1 + tol) and max_y <= w * (1 + tol):
        return (f"hand-drawn boxes (max x={max_x:.0f}, y={max_y:.0f}) only fit a "
                f"{h}x{w} frame, not the decoded {w}x{h} -- frames look un-rotated")
    return (f"hand-drawn boxes (max x={max_x:.0f}, y={max_y:.0f}) don't fit the decoded "
            f"{w}x{h} frame even swapped -- not a simple rotation mismatch")


def slice_frame_dir(frames_dir: pathlib.Path, lo: int, hi: int,
                    reverse: bool, tag: str) -> pathlib.Path:
    """A sibling dir of symlinks covering frames [lo, hi], optionally reversed.

    SAM 3 propagates forward from the prompt to the end of the folder, so handing
    it a windowed and/or reversed folder is the version-proof way to control both
    the direction and the extent of propagation.
    """
    d = frames_dir.parent / f"{frames_dir.name}__{tag}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    order = range(hi, lo - 1, -1) if reverse else range(lo, hi + 1)
    for i, src in enumerate(order):
        os.symlink((frames_dir / f"{src:05d}.jpg").resolve(), d / f"{i:05d}.jpg")
    return d


def event_window(anno: dict, event_index: int, n_frames: int, margin: int,
                 max_frames: int = 0) -> Tuple[int, int]:
    """Frame range to propagate one delivery over.

    A clip with 18 deliveries must not have one bowler box propagated across all
    of them: the window is the event plus a margin, clamped to the midpoints
    between neighbouring events so consecutive deliveries never bleed together.
    max_frames additionally caps the total window size (memory per SAM 3 session),
    shrinking symmetrically around the event if the margin/neighbour clamp alone
    would still allow a larger window.
    """
    events = anno["temporal_events"]
    ev = events[event_index]
    lo_bound, hi_bound = 0, n_frames - 1
    if event_index > 0:
        lo_bound = (events[event_index - 1]["end_frame"] + ev["start_frame"]) // 2
    if event_index < len(events) - 1:
        hi_bound = (ev["end_frame"] + events[event_index + 1]["start_frame"]) // 2
    lo = max(0, lo_bound, ev["start_frame"] - margin)
    hi = min(n_frames - 1, hi_bound, ev["end_frame"] + margin)
    if max_frames and (hi - lo + 1) > max_frames:
        center = (ev["start_frame"] + ev["end_frame"]) // 2
        lo = max(lo, center - max_frames // 2)
        hi = min(hi, lo + max_frames - 1)
    return lo, max(lo, hi)


# --------------------------------------------------------------------------
# SAM 3 adapter
# --------------------------------------------------------------------------

class Sam3:
    """Thin adapter over the SAM 3 session API.

    Only two things are consumed from the model output: the set of object ids and
    a mask per object. Everything else (boxes, scores) is derived or optional, so
    a schema change upstream breaks one function instead of the pipeline.
    """

    def __init__(self, load_image_model: bool = True, max_objects: int = 20):
        from sam3.model_builder import build_sam3_video_predictor
        try:
            self.video = build_sam3_video_predictor(max_num_objects=max_objects)
        except TypeError:
            print(f"warning: this SAM 3 build's video predictor doesn't accept "
                  f"max_num_objects; continuing without a cap", file=sys.stderr)
            self.video = build_sam3_video_predictor()
        self.image = None
        if load_image_model:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.image = Sam3Processor(build_sam3_image_model())

    # -- video ------------------------------------------------------------

    def _start(self, frames_dir: pathlib.Path) -> str:
        r = self.video.handle_request(
            request=dict(type="start_session", resource_path=str(frames_dir)))
        return r["session_id"]

    def _close(self, session_id: str) -> None:
        try:
            self.video.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception:
            pass

    def track_box(self, frames_dir: pathlib.Path, frame_index: int,
                  box_norm: Tuple[float, float, float, float]) -> Dict[int, dict]:
        """Propagate a single normalized xywh box prompt. -> {frame: {obj_id: mask}}"""
        sid = self._start(frames_dir)
        try:
            self.video.handle_request(request=dict(
                type="add_prompt", session_id=sid, frame_index=frame_index,
                bounding_boxes=[list(box_norm)], bounding_box_labels=[1], obj_id=1))
            return self._propagate(sid)
        finally:
            self._close(sid)

    def track_text(self, frames_dir: pathlib.Path, frame_index: int, text: str) -> Dict[int, dict]:
        sid = self._start(frames_dir)
        try:
            self.video.handle_request(request=dict(
                type="add_prompt", session_id=sid, frame_index=frame_index, text=text))
            return self._propagate(sid)
        finally:
            self._close(sid)

    def _propagate(self, session_id: str) -> Dict[int, dict]:
        per_frame = {}
        for resp in self.video.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=session_id)):
            idx = resp.get("frame_index", resp.get("frame_idx"))
            per_frame[int(idx)] = resp["outputs"]
        return per_frame

    # -- image ------------------------------------------------------------

    def detect_image(self, image_path: pathlib.Path, text: str):
        from PIL import Image
        state = self.image.set_image(Image.open(image_path).convert("RGB"))
        return self.image.set_text_prompt(state=state, prompt=text)


# --------------------------------------------------------------------------
# output parsing (the only version-sensitive part)
# --------------------------------------------------------------------------

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if hasattr(x, "dtype") and str(x.dtype) in ("torch.bfloat16", "torch.float16"):
            x = x.float()  # numpy has no bfloat16 (and older numpy chokes on float16 -> object array)
    return np.asarray(x)


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[Box, Tuple[int, int]]]:
    """Tight box of a (possibly low-res, possibly logit) mask, in MASK coords."""
    m = np.squeeze(_to_numpy(mask))
    if m.ndim != 2 or m.size == 0:
        return None
    binary = m > 0.0 if m.dtype.kind == "f" else m.astype(bool)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)), m.shape


def parse_outputs(outputs) -> Dict[int, Tuple[Box, Tuple[int, int], float]]:
    """SAM 3 frame output -> {obj_id: (box_in_mask_coords, mask_hw, score)}.

    Handles the shapes seen across SAM 3 / 3.1 and the transformers port. Run
    --probe if this raises: it prints the real structure.
    """
    if outputs is None:
        return {}
    if not isinstance(outputs, dict):
        outputs = getattr(outputs, "__dict__", {}) or {}

    def first(*keys):
        """First key that is present and not None. `or` would break on arrays."""
        for k in keys:
            if outputs.get(k) is not None:
                return outputs[k]
        return None

    scores = first("obj_id_to_score", "scores", "out_probs")

    def score_for(oid, pos):
        if isinstance(scores, dict):
            return float(_to_numpy(scores.get(oid, float("nan"))))
        arr = _to_numpy(scores) if scores is not None else np.array([])
        return float(arr.flat[pos]) if arr.size > pos else float("nan")

    # Shape A: explicit id -> mask mapping.
    id_to_mask = first("obj_id_to_mask", "masks_by_obj")
    if isinstance(id_to_mask, dict):
        out = {}
        for pos, (oid, mask) in enumerate(id_to_mask.items()):
            got = mask_to_box(mask)
            if got:
                out[int(oid)] = (got[0], got[1], score_for(oid, pos))
        return out

    # Shape B: parallel object_ids + stacked masks.
    ids = first("object_ids", "obj_ids", "out_obj_ids")
    masks = first("pred_masks", "masks", "out_binary_masks")
    if ids is not None and masks is not None:
        ids = [int(i) for i in _to_numpy(ids).reshape(-1)]
        arr = _to_numpy(masks)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        out = {}
        for pos, oid in enumerate(ids):
            if pos >= arr.shape[0]:
                break
            got = mask_to_box(arr[pos])
            if got:
                out[oid] = (got[0], got[1], score_for(oid, pos))
        return out

    raise KeyError(f"Unrecognised SAM 3 output keys: {sorted(outputs.keys())} "
                   f"-- rerun with --probe and update parse_outputs().")


def rescale(box: Box, mask_hw: Tuple[int, int], orig_w: int, orig_h: int) -> Box:
    """Mask-resolution box -> original-frame pixels."""
    mh, mw = mask_hw
    sx, sy = orig_w / float(mw), orig_h / float(mh)
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------

def _collect_track(per_frame: Dict[int, dict], orig_w: int, orig_h: int,
                   lo: int, hi: int, reverse: bool) -> Dict[int, Dict[str, list]]:
    """{obj_id: {"boxes": {frame: box}, "scores": {frame: score}}}, in clip frames."""
    tracks: Dict[int, Dict[str, dict]] = {}
    for fidx, outputs in per_frame.items():
        real = (hi - fidx) if reverse else (lo + fidx)
        for oid, (box, mask_hw, score) in parse_outputs(outputs).items():
            t = tracks.setdefault(oid, {"boxes": {}, "scores": {}})
            t["boxes"][real] = [round(v, 1) for v in rescale(box, mask_hw, orig_w, orig_h)]
            if score == score:  # not NaN
                t["scores"][real] = round(score, 4)
    return tracks


def _stitch(fwd: dict, bwd: dict, join_frame: int, iou_thresh: float = 0.5) -> dict:
    """Merge a backward half-track set into a forward one.

    The two propagations are separate sessions, so their object ids are unrelated.
    Both halves do cover the prompt frame, though, so align ids by IoU there and
    carry the mapping across every backward frame. Forward wins on overlap.
    """
    def at(tracks, frame):
        return {oid: t["boxes"][frame] for oid, t in tracks.items() if frame in t["boxes"]}

    f_at, b_at = at(fwd, join_frame), at(bwd, join_frame)
    pairs = sorted(((iou(tuple(fb), tuple(bb)), fid, bid)
                    for fid, fb in f_at.items() for bid, bb in b_at.items()),
                   reverse=True)
    mapping, used_f, used_b = {}, set(), set()
    for v, fid, bid in pairs:
        if v < iou_thresh or fid in used_f or bid in used_b:
            continue
        mapping[bid] = fid
        used_f.add(fid); used_b.add(bid)

    merged = {oid: {"boxes": dict(t["boxes"]), "scores": dict(t["scores"])}
              for oid, t in fwd.items()}
    next_id = max(list(merged) + list(bwd) + [0]) + 1
    for bid, t in bwd.items():
        tgt = mapping.get(bid)
        if tgt is None:  # appears only before the prompt frame; keep as its own track
            tgt = next_id
            next_id += 1
            merged[tgt] = {"boxes": {}, "scores": {}}
        for frame, box in t["boxes"].items():
            merged[tgt]["boxes"].setdefault(frame, box)
        for frame, sc in t["scores"].items():
            merged[tgt]["scores"].setdefault(frame, sc)
    return merged


def pass_bowler(sam: Sam3, anno: dict, frames_dir: pathlib.Path,
                n_frames: int, w: int, h: int, margin: int, kept: List[int],
                max_window_frames: int = 0) -> List[dict]:
    """anno must already be remapped (remap_annotation_to_frames) to the decimated
    frame indices; kept[] maps those back to original video frame numbers for output.
    """
    results = []
    for event_index, prompt_frame, box in bowler_prompts(anno):
        if prompt_frame >= n_frames:
            continue
        lo, hi = event_window(anno, event_index, n_frames, margin, max_window_frames)
        prompt_frame = min(max(prompt_frame, lo), hi)
        # normalized xywh, per the notebook's abs_to_rel_coords(coord_type="box")
        box_norm = box_to_norm_xywh(box, w, h)

        fwd_dir = slice_frame_dir(frames_dir, lo, hi, False, f"b{event_index}f")
        bwd_dir = slice_frame_dir(frames_dir, lo, hi, True, f"b{event_index}r")
        try:
            fwd = _collect_track(sam.track_box(fwd_dir, prompt_frame - lo, box_norm),
                                 w, h, lo, hi, reverse=False)
            bwd = _collect_track(sam.track_box(bwd_dir, hi - prompt_frame, box_norm),
                                 w, h, lo, hi, reverse=True)
        finally:
            shutil.rmtree(fwd_dir, ignore_errors=True)
            shutil.rmtree(bwd_dir, ignore_errors=True)

        # The prompt seeds exactly one object, so the halves merge without matching.
        boxes, scores = {}, {}
        for half in (bwd, fwd):  # forward wins on the shared prompt frame
            for t in half.values():
                boxes.update(t["boxes"])
                scores.update(t["scores"])
        span = hi - lo + 1
        results.append({
            "event_index": event_index,
            "prompt_frame": kept[prompt_frame],
            "prompt_box": [round(v, 1) for v in box],
            "window": [kept[lo], kept[hi]],
            "n_frames_tracked": len(boxes),
            "coverage": round(len(boxes) / max(span, 1), 3),
            "boxes": {str(kept[k]): boxes[k] for k in sorted(boxes)},
            "scores": {str(kept[k]): scores[k] for k in sorted(scores)},
        })
    return results


def pass_person(sam: Sam3, frames_dir: pathlib.Path, prompt_frame: int,
                n_frames: int, w: int, h: int, text: str, kept: List[int],
                max_window_frames: int = 0) -> dict:
    """Text-prompted people, tracked in a window around the delivery frame (both
    directions), not the whole clip: handing SAM 3 the full frames folder made one
    session hold the entire (possibly very long) video regardless of how far
    propagation actually needed to go, which is what was causing OOMs on long clips.

    Seeded at the delivery frame, where every role is on screen, so propagation
    also runs backward to keep the run-up frames and their negatives. prompt_frame/
    n_frames are in the decimated frame space; kept[] maps output back to original
    video frame numbers.
    """
    if max_window_frames and max_window_frames > 0:
        half = max_window_frames // 2
        lo = max(0, prompt_frame - half)
        hi = min(n_frames - 1, prompt_frame + half)
    else:
        lo, hi = 0, n_frames - 1

    fwd_dir = slice_frame_dir(frames_dir, lo, hi, False, "p_fwd")
    bwd_dir = slice_frame_dir(frames_dir, lo, hi, True, "p_bwd")
    try:
        tracks = _collect_track(sam.track_text(fwd_dir, prompt_frame - lo, text),
                                w, h, lo, hi, reverse=False)
        if prompt_frame > lo:
            bwd = _collect_track(sam.track_text(bwd_dir, hi - prompt_frame, text),
                                 w, h, lo, hi, reverse=True)
            tracks = _stitch(tracks, bwd, prompt_frame)
    finally:
        shutil.rmtree(fwd_dir, ignore_errors=True)
        shutil.rmtree(bwd_dir, ignore_errors=True)
    return {str(oid): {"boxes": {str(kept[k]): v for k, v in sorted(t["boxes"].items())},
                       "scores": {str(kept[k]): v for k, v in sorted(t["scores"].items())},
                       "n_frames": len(t["boxes"])}
            for oid, t in tracks.items()}


def pass_image_scan(sam: Sam3, frames_dir: pathlib.Path, n_frames: int, w: int, h: int,
                    text: str, n_samples: int, kept: List[int], max_objects: int = 0) -> dict:
    """Per-frame image detection for a text prompt. n_samples caps how many frames are
    scanned, evenly spaced across the clip; n_samples <= 0 (or >= n_frames) scans every
    frame instead. Good for a near-static consensus box (stumps) on a sparse sample, or
    dense scanning for a fast-moving small object (ball) -- it is still per-frame
    detection, not tracking, so isolated frames can still miss it even when dense.
    max_objects caps how many (highest-score) detections are kept per frame (0 = no cap).
    """
    if sam.image is None or n_frames == 0:
        return {}
    if n_samples <= 0 or n_samples >= n_frames:
        idxs = list(range(n_frames))
    else:
        idxs = sorted({int(round(i * (n_frames - 1) / max(n_samples - 1, 1)))
                       for i in range(n_samples)})
    dets = []
    for i in idxs:
        try:
            out = sam.detect_image(frames_dir / f"{i:05d}.jpg", text)
        except Exception as e:
            dets.append({"frame": i, "error": str(e)})
            continue
        masks = out.get("masks")
        scores = _to_numpy(out.get("scores")).reshape(-1) if out.get("scores") is not None else np.array([])
        boxes = []
        arr = _to_numpy(masks) if masks is not None else np.zeros((0, 1, 1))
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        for k in range(arr.shape[0] if arr.ndim == 3 else 0):
            got = mask_to_box(arr[k])
            if got:
                boxes.append({"box": [round(v, 1) for v in rescale(got[0], got[1], w, h)],
                              "score": round(float(scores[k]), 4) if scores.size > k else None})
        if max_objects and len(boxes) > max_objects:
            boxes.sort(key=lambda b: b["score"] if b["score"] is not None else -1.0, reverse=True)
            boxes = boxes[:max_objects]
        dets.append({"frame": kept[i], "detections": boxes})
    return {"prompt": text, "samples": dets}


def interpolate_box(boxes: Dict[str, list], frame: int) -> Optional[list]:
    """Box at `frame`, linearly interpolated between the two nearest tracked frames
    straddling it. `boxes` is sparse (only frames actually decoded after --target_fps
    decimation have an entry), so an annotated frame number is rarely an exact key --
    without this, comparisons against the original hand-labelled frame miss almost
    every time purely because that frame was never decoded, not because tracking failed.
    Returns None if `frame` is outside the track's covered range (no extrapolation).
    """
    keys = sorted(int(k) for k in boxes)
    if not keys:
        return None
    pos = bisect.bisect_left(keys, frame)
    if pos < len(keys) and keys[pos] == frame:
        return boxes[str(frame)]
    if pos == 0 or pos == len(keys):
        return None
    lo, hi = keys[pos - 1], keys[pos]
    b_lo, b_hi = boxes[str(lo)], boxes[str(hi)]
    t = (frame - lo) / (hi - lo)
    return [b_lo[i] + t * (b_hi[i] - b_lo[i]) for i in range(4)]


def evaluate(anno: dict, persons: dict) -> dict:
    """Can the generic 'person' pass recover the hand-annotated bowler?

    This is the number that decides whether a geometric bowler-selection rule is
    viable at runtime, so it is computed for every clip.
    """
    rows = []
    for event_index, frame, gt in bowler_prompts(anno):
        best_iou, best_oid, n_people = 0.0, None, 0
        for oid, t in persons.items():
            box = interpolate_box(t["boxes"], frame)
            if not box:
                continue
            n_people += 1
            v = iou(tuple(box), gt)
            if v > best_iou:
                best_iou, best_oid = v, oid
        rows.append({"event_index": event_index, "frame": frame,
                     "n_people_on_frame": n_people,
                     "best_person_iou": round(best_iou, 3), "best_person_id": best_oid})
    return {"per_event": rows}


def match_bowler_to_person(bowler_tracks: List[dict], person_tracks: dict,
                           iou_thresh: float = 0.3, min_frames: int = 3) -> List[dict]:
    """For each bowler event, find which 'person' pass track id is the same bowler by
    averaging IoU over every frame both tracks have a box, so the generic person track
    can be labelled with the bowler role after SAM 3 is done, instead of only checking
    a single frame against the hand annotation (see evaluate()).
    """
    rows = []
    for ev in bowler_tracks:
        bboxes = ev["boxes"]
        best_oid, best_iou, best_n = None, 0.0, 0
        for oid, t in person_tracks.items():
            shared = bboxes.keys() & t["boxes"].keys()
            if len(shared) < min_frames:
                continue
            mean_iou = sum(iou(tuple(bboxes[f]), tuple(t["boxes"][f])) for f in shared) / len(shared)
            if mean_iou > best_iou:
                best_oid, best_iou, best_n = oid, mean_iou, len(shared)
        rows.append({
            "event_index": ev["event_index"],
            "bowler_person_id": best_oid if best_iou >= iou_thresh else None,
            "match_iou": round(best_iou, 3),
            "n_frames_compared": best_n,
        })
    return rows


# --------------------------------------------------------------------------
# viz
# --------------------------------------------------------------------------

COLORS = [(0, 220, 0), (255, 120, 0), (0, 160, 255), (200, 0, 200), (0, 220, 220), (160, 160, 0)]


def write_viz(video: pathlib.Path, sidecar: dict, out_path: pathlib.Path) -> None:
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 720 / max(w, h))
    ow, oh = int(w * scale), int(h * scale)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    bowler = {}
    for ev in sidecar.get("bowler_tracks", []):
        bowler.update(ev["boxes"])
    persons = sidecar.get("person_tracks", {})
    bowler_person_ids = {m["bowler_person_id"] for m in sidecar.get("bowler_person_match", [])
                         if m["bowler_person_id"] is not None}
    stumps = [d["box"] for s in sidecar.get("stumps", {}).get("samples", [])
              for d in s.get("detections", [])]
    ball = [d["box"] for s in sidecar.get("ball", {}).get("samples", [])
            for d in s.get("detections", [])]

    def draw(img, box, color, label):
        x1, y1, x2, y2 = [int(v * scale) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (ow, oh))
        for pos, (oid, t) in enumerate(sorted(persons.items())):
            box = interpolate_box(t["boxes"], i)
            if box:
                if oid in bowler_person_ids:
                    draw(frame, box, (0, 0, 255), f"p{oid} BOWLER")
                else:
                    draw(frame, box, COLORS[pos % len(COLORS)], f"p{oid}")
        bowler_box = interpolate_box(bowler, i)
        if bowler_box:
            draw(frame, bowler_box, (0, 0, 255), "BOWLER")
        for box in stumps:
            draw(frame, box, (255, 255, 255), "stumps")
        for box in ball:
            draw(frame, box, (0, 255, 255), "ball")
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def probe(sam: Sam3, anno: dict, frames_dir: pathlib.Path, n_frames: int) -> None:
    print("\n=== PROBE: raw SAM 3 output structure ===")
    prompts = bowler_prompts(anno)
    if prompts:
        _, frame, box = prompts[0]
        cap = open_video(anno["_video_path"])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        bn = (box[0] / w, box[1] / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h)
        pf = sam.track_box(frames_dir, frame, bn)
        _dump("box prompt", pf)
    pf = sam.track_text(frames_dir, 0, "person")
    _dump("text prompt 'person'", pf)
    if sam.image is not None:
        out = sam.detect_image(frames_dir / "00000.jpg", "cricket stumps")
        print("\n-- image predictor 'cricket stumps' --")
        print("  keys:", sorted(out.keys()) if isinstance(out, dict) else type(out))
        for k, v in (out.items() if isinstance(out, dict) else []):
            print(f"    {k}: {type(v).__name__} {getattr(v, 'shape', '')}")


def _dump(title: str, per_frame: Dict[int, dict]) -> None:
    print(f"\n-- {title}: {len(per_frame)} frames returned --")
    for fidx in sorted(per_frame)[:2]:
        out = per_frame[fidx]
        print(f"  frame {fidx}: type={type(out).__name__}")
        if isinstance(out, dict):
            for k, v in out.items():
                shape = getattr(v, "shape", None)
                if isinstance(v, dict):
                    print(f"    {k}: dict keys={list(v)[:5]}")
                    for kk, vv in list(v.items())[:1]:
                        print(f"      [{kk}]: {type(vv).__name__} {getattr(vv, 'shape', '')}")
                else:
                    print(f"    {k}: {type(v).__name__} {shape if shape is not None else repr(v)[:80]}")
        try:
            print("    parse_outputs ->", {k: (v[0], v[1], v[2]) for k, v in parse_outputs(out).items()})
        except Exception as e:
            print("    parse_outputs FAILED:", e)


def report(out_dir: pathlib.Path) -> None:
    sidecars = sorted(out_dir.glob("*.json"))
    if not sidecars:
        print(f"No sidecars in {out_dir}")
        return
    cov, n_people, ious, stump_hit, stump_scores, n_events = [], [], [], 0, [], 0
    ball_hit, ball_scores = 0, []
    match_ious, n_matched = [], 0
    for p in sidecars:
        d = json.load(open(p))
        for ev in d.get("bowler_tracks", []):
            cov.append(ev["coverage"]); n_events += 1
        n_people.append(len(d.get("person_tracks", {})))
        for row in d.get("eval", {}).get("per_event", []):
            ious.append(row["best_person_iou"])
        for row in d.get("bowler_person_match", []):
            match_ious.append(row["match_iou"])
            if row["bowler_person_id"] is not None:
                n_matched += 1
        dets = [x for s in d.get("stumps", {}).get("samples", []) for x in s.get("detections", [])]
        if dets:
            stump_hit += 1
            stump_scores += [x["score"] for x in dets if x.get("score") is not None]
        bdets = [x for s in d.get("ball", {}).get("samples", []) for x in s.get("detections", [])]
        if bdets:
            ball_hit += 1
            ball_scores += [x["score"] for x in bdets if x.get("score") is not None]

    def q(xs, p):
        xs = sorted(xs); return xs[int(p * (len(xs) - 1))] if xs else float("nan")

    print(f"\nsidecars={len(sidecars)}  events={n_events}")
    print("\n[bowler pass] fraction of clip frames with a tracked bowler box")
    print(f"  p10={q(cov,.1):.2f}  median={q(cov,.5):.2f}  p90={q(cov,.9):.2f}"
          f"   full-coverage events(>=0.95): {sum(c>=0.95 for c in cov)}/{len(cov)}")
    print("\n[person pass] tracks per clip")
    print(f"  median={q(n_people,.5):.0f}  p90={q(n_people,.9):.0f}  max={max(n_people) if n_people else 0}")
    print("\n[bowler recoverable from generic person detection] IoU vs hand annotation at prompt frame")
    print(f"  median={q(ious,.5):.2f}  p10={q(ious,.1):.2f}")
    for t in (0.3, 0.5, 0.7):
        print(f"  recall@IoU>{t}: {sum(v>t for v in ious)}/{len(ious)} = {sum(v>t for v in ious)/max(len(ious),1):.1%}")
    print("\n[bowler<->person match] whole-window IoU between bowler_tracks and the matched person track")
    print(f"  events matched: {n_matched}/{len(match_ious)} = {n_matched/max(len(match_ious),1):.1%}")
    if match_ious:
        print(f"  match IoU: p10={q(match_ious,.1):.2f} median={q(match_ious,.5):.2f}")
    print("\n[stumps] clips with >=1 detection: "
          f"{stump_hit}/{len(sidecars)} = {stump_hit/max(len(sidecars),1):.1%}")
    if stump_scores:
        print(f"  detection score: p10={q(stump_scores,.1):.2f} median={q(stump_scores,.5):.2f}")
    print("\n[ball] clips with >=1 detection: "
          f"{ball_hit}/{len(sidecars)} = {ball_hit/max(len(sidecars),1):.1%}")
    if ball_scores:
        print(f"  detection score: p10={q(ball_scores,.1):.2f} median={q(ball_scores,.5):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--frames_cache", type=pathlib.Path, default=pathlib.Path("sam3_frames"))
    ap.add_argument("--max_side", type=int, default=1536,
                    help="Downscale frames for SAM 3; boxes are still reported in original pixels.")
    ap.add_argument("--target_fps", type=float, default=30.0,
                    help="Subsample decoded frames to at most this rate before feeding SAM 3 "
                         "(0 = keep native fps). Tracked frames are still reported at the "
                         "original video's frame numbers.")
    ap.add_argument("--passes", default="person,stumps,ball",
                    help="'bowler' (box-prompt tracking) is opt-in only, not default: "
                         "the 'person' pass + evaluate()/match_bowler_to_person already "
                         "identify the bowler without needing a separate box-prompted session.")
    ap.add_argument("--person_text", default="person")
    ap.add_argument("--stump_text", default="cricket stumps")
    ap.add_argument("--stump_samples", type=int, default=5,
                    help="Frames scanned for stumps, evenly spaced (0 = every frame).")
    ap.add_argument("--ball_text", default="ball",
                    help="Text prompt for the cricket-ball image-detection pass.")
    ap.add_argument("--ball_samples", type=int, default=8,
                    help="Frames scanned for ball detection, evenly spaced (0 = every "
                         "frame; the ball moves fast, so dense scanning is recommended "
                         "when you have the GPU budget for it).")
    ap.add_argument("--bowler_margin", type=float, default=3.0,
                    help="Seconds of run-up/follow-through to propagate either side of an "
                         "event, clamped to the midpoint between adjacent deliveries.")
    ap.add_argument("--max_window_frames", type=int, default=120,
                    help="Cap on frames held in one SAM 3 video-predictor session (bowler "
                         "event window, or person-pass window around the prompt frame), "
                         "after margin/neighbour clamping (0 = no cap, i.e. whole clip for "
                         "person -- can OOM on long videos). Bounds SAM 3 memory use "
                         "regardless of clip length.")
    ap.add_argument("--person_prompt", default="middle", choices=["first", "middle"],
                    help="Frame to seed the text prompt on. 'middle' = the event's middle_frame, "
                         "where every role is on screen.")
    ap.add_argument("--max_objects", type=int, default=20,
                    help="Cap on tracked objects in the video predictor session, and on "
                         "(highest-score) detections kept per frame in the stumps/ball "
                         "image-detection passes.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split clips across N independent processes (e.g. one per GPU).")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="Which shard this process handles, in [0, num_shards). Pin the GPU "
                         "separately via CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--viz", action="store_true", help="Also write an annotated mp4 per clip.")
    ap.add_argument("--keep_frames", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--probe", action="store_true", help="Dump raw output structure and exit.")
    ap.add_argument("--report", action="store_true", help="Summarise existing sidecars and exit.")
    args = ap.parse_args()

    if args.report:
        report(args.out_dir)
        return
    passes = {p.strip() for p in args.passes.split(",") if p.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.frames_cache.mkdir(parents=True, exist_ok=True)

    annos = []
    for jp in sorted(args.video_dir.glob("*.json"), reverse=True):
        a = load_annotation(jp)
        if a and a.get("temporal_events"):
            annos.append(a)
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(f"--shard_index must be in [0, {args.num_shards})")
        annos = annos[args.shard_index::args.num_shards]  # disjoint by construction, safe to run in parallel
    if args.limit:
        annos = annos[:args.limit]
    print(f"{len(annos)} annotated clips (shard {args.shard_index}/{args.num_shards}); passes={sorted(passes)}")

    sam = Sam3(load_image_model=bool({"stumps", "ball"} & passes) or args.probe,
              max_objects=args.max_objects)

    for k, anno in enumerate(annos, 1):
        video = anno["_video_path"]
        sidecar_path = args.out_dir / f"{video.stem}.json"
        if sidecar_path.exists() and not args.overwrite and not args.probe:
            try:
                prev = json.load(open(sidecar_path, encoding="utf-8"))
                stale_orientation = prev.get("width", 0) > prev.get("height", 0)
            except Exception:
                stale_orientation = False
            if not stale_orientation:
                print(f"[{k}/{len(annos)}] skip {video.name} (exists)")
                continue
            print(f"[{k}/{len(annos)}] {video.name}: existing sidecar is landscape "
                  f"({prev.get('width')}x{prev.get('height')}) -- forcing reprocess")

        frames_dir = args.frames_cache / video.stem
        try:
            n_frames, w, h, fps, eff_fps, kept = extract_frames(
                video, frames_dir, args.max_side, args.target_fps)
            if n_frames == 0:
                print(f"[{k}/{len(annos)}] {video.name}: no frames, skipping")
                continue
            orientation_problem = verify_frame_orientation(anno, w, h)
            if orientation_problem:
                print(f"[{k}/{len(annos)}] {video.name}: SKIPPING, orientation mismatch: "
                      f"{orientation_problem}", file=sys.stderr)
                continue
            print(f"[{k}/{len(annos)}] {video.name}: {n_frames} frames {w}x{h} "
                  f"({fps:.1f}->{eff_fps:.1f} fps)")
            sam_anno = remap_annotation_to_frames(anno, kept)

            if args.probe:
                probe(sam, sam_anno, frames_dir, n_frames)
                return

            sidecar = {"video": video.name, "width": w, "height": h,
                       "total_frames": n_frames, "fps": round(fps, 3), "eff_fps": round(eff_fps, 3),
                       "max_side": args.max_side}

            if "bowler" in passes:
                margin = int(round(args.bowler_margin * (eff_fps or 30.0)))
                sidecar["bowler_tracks"] = pass_bowler(sam, sam_anno, frames_dir, n_frames, w, h,
                                                       margin, kept, args.max_window_frames)

            if "person" in passes:
                prompts = bowler_prompts(sam_anno)
                pf = prompts[0][1] if (args.person_prompt == "middle" and prompts) else 0
                pf = min(pf, n_frames - 1)
                sidecar["person_prompt_frame"] = kept[pf]
                sidecar["person_tracks"] = pass_person(sam, frames_dir, pf, n_frames,
                                                       w, h, args.person_text, kept,
                                                       args.max_window_frames)
                sidecar["eval"] = evaluate(anno, sidecar["person_tracks"])

            if "stumps" in passes:
                sidecar["stumps"] = pass_image_scan(sam, frames_dir, n_frames, w, h,
                                                    args.stump_text, args.stump_samples, kept,
                                                    args.max_objects)

            if "ball" in passes:
                sidecar["ball"] = pass_image_scan(sam, frames_dir, n_frames, w, h,
                                                  args.ball_text, args.ball_samples, kept,
                                                  args.max_objects)

            # bowler_person_match (box-prompt vs person-prompt) intentionally not computed:
            # the box-prompt bowler track is lower quality than the person track it'd be
            # matched against, so it isn't a trustworthy label source (see evaluate() instead).

            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=1)
            print(f"    -> {sidecar_path}")

            if args.viz:
                viz_dir = args.out_dir / "viz"
                viz_dir.mkdir(exist_ok=True)
                write_viz(video, sidecar, viz_dir / f"{video.stem}.mp4")

        except Exception:
            print(f"[{k}/{len(annos)}] {video.name}: FAILED", file=sys.stderr)
            traceback.print_exc()
        finally:
            if not args.keep_frames:
                shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
