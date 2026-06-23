#!/usr/bin/env python3
"""
compare_models.py — Run a video through both the PyTorch (fp32) model and
the exported Core ML (fp16) .mlpackage, then print a side-by-side summary
and save an interactive Altair comparison chart.

Usage:
    python compare_models.py <video> \\
        --pt         models/fastvit_sa12_exp07.pt \\
        --mlpackage  exports/fastvit_t12.mlpackage \\
        --output     compare_chart.html
"""

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import coremltools as ct
import altair as alt
import pandas as pd

from modellib import crop, read_start_end


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_pytorch_model(pt_path: pathlib.Path) -> Tuple[torch.nn.Module, int, int]:
    """Load a portable artifact or training checkpoint; return (model, H, W)."""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if "config" in ckpt:
        cfg = ckpt["config"]
        model_name  = cfg["model_name"]
        num_classes = cfg["num_classes"]
        h, w = cfg["input_height"], cfg["input_width"]
    elif "args" in ckpt:
        args = ckpt["args"]
        model_name  = args.model_name
        num_classes = args.num_classes
        h, w = args.input_height, args.input_width
    else:
        raise ValueError("Unrecognised checkpoint (no 'config' or 'args' key).")

    base = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()

    try:
        from timm.utils import reparameterize_model
        base = reparameterize_model(base).eval()
        print("  reparameterized for inference")
    except Exception:
        pass

    class _WithSoftmax(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softmax(self.m(x), dim=1)

    return _WithSoftmax(base).half().eval(), h, w


def load_coreml_model(mlpkg_path: pathlib.Path):
    """Load a Core ML .mlpackage; return (model, in_name, out_name)."""
    mlmodel  = ct.models.MLModel(str(mlpkg_path))
    spec     = mlmodel.get_spec()
    in_name  = spec.description.input[0].name
    out_name = spec.description.output[0].name
    return mlmodel, in_name, out_name


def make_transform(h: int, w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def predict_pytorch(
    model: torch.nn.Module,
    transform: transforms.Compose,
    bgr_frame: np.ndarray,
) -> Tuple[bool, float]:
    pil = Image.fromarray(crop(bgr_frame))
    tensor = transform(pil).unsqueeze(0).half()
    with torch.no_grad():
        probs = model(tensor)
    conf = probs[0, 1].item()
    return conf >= 0.5, conf


def predict_coreml(
    mlmodel,
    in_name: str,
    out_name: str,
    transform: transforms.Compose,
    bgr_frame: np.ndarray,
) -> Tuple[bool, float]:
    pil = Image.fromarray(crop(bgr_frame))
    arr = transform(pil).unsqueeze(0).numpy()   # (1, 3, H, W) float32
    preds = mlmodel.predict({in_name: arr})
    probs = np.asarray(preds[out_name]).flatten()
    conf  = float(probs[1])
    return conf >= 0.5, conf


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def compare_on_video(
    video_path: pathlib.Path,
    pt_path: pathlib.Path,
    mlpkg_path: pathlib.Path,
    output_chart: pathlib.Path,
) -> None:
    print(f"Loading PyTorch model (fp16): {pt_path}")
    pt_model, h, w = load_pytorch_model(pt_path)
    tf = make_transform(h, w)
    print(f"  input size : {h}x{w}")

    print(f"Loading CoreML model   : {mlpkg_path}")
    ml_model, in_name, out_name = load_coreml_model(mlpkg_path)
    # Both models share the same backbone so use the same transform.

    ann_path = video_path.with_suffix(".json")
    events: List[Tuple[int, int]] = []
    if ann_path.exists():
        events = read_start_end(video_path)
        print(f"Annotation             : {ann_path} ({len(events)} event(s))")
    else:
        print(f"No annotation found at {ann_path} — GT column will be absent.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video                  : {total} frames @ {fps:.1f} fps\n")

    rows = []
    idx  = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pt_pred, pt_conf = predict_pytorch(pt_model, tf, frame)
        ml_pred, ml_conf = predict_coreml(ml_model, in_name, out_name, tf, frame)

        gt: Optional[bool] = None
        if events:
            gt = any(s <= idx <= e for s, e in events)

        agree = pt_pred == ml_pred
        rows.append({
            "frame":   idx,
            "pt_conf": pt_conf,
            "ml_conf": ml_conf,
            "pt_pred": int(pt_pred),
            "ml_pred": int(ml_pred),
            "agree":   int(agree),
            "gt":      int(gt) if gt is not None else None,
        })

        if idx % 50 == 0:
            gt_str = f"  gt={int(gt)}" if gt is not None else ""
            print(
                f"  frame {idx:4d}/{total}"
                f"  pytorch={pt_conf:.3f}({'Y' if pt_pred else 'N'})"
                f"  coreml={ml_conf:.3f}({'Y' if ml_pred else 'N'})"
                f"  agree={'yes' if agree else 'NO'}{gt_str}"
            )
        idx += 1

    cap.release()

    # ----- Summary ----------------------------------------------------------
    n       = len(rows)
    n_agree = sum(r["agree"] for r in rows)
    diffs   = [abs(r["pt_conf"] - r["ml_conf"]) for r in rows]

    print(f"\n{'='*55}")
    print(f"Frames processed      : {n}")
    print(f"Agreement rate        : {n_agree}/{n}  ({100 * n_agree / n:.1f}%)")
    print(f"Max  |Δconfidence|    : {max(diffs):.4f}")
    print(f"Mean |Δconfidence|    : {sum(diffs) / len(diffs):.4f}")
    if events:
        pt_corr = sum(r["pt_pred"] == r["gt"] for r in rows if r["gt"] is not None)
        ml_corr = sum(r["ml_pred"] == r["gt"] for r in rows if r["gt"] is not None)
        print(f"PyTorch  accuracy     : {pt_corr}/{n}  ({100 * pt_corr / n:.1f}%)")
        print(f"CoreML   accuracy     : {ml_corr}/{n}  ({100 * ml_corr / n:.1f}%)")
    print(f"{'='*55}\n")

    # ----- Chart ------------------------------------------------------------
    df = pd.DataFrame(rows)

    conf_rows = []
    for _, r in df.iterrows():
        conf_rows.append({"frame": r["frame"], "confidence": r["pt_conf"], "model": "pytorch_fp16"})
        conf_rows.append({"frame": r["frame"], "confidence": r["ml_conf"], "model": "coreml_fp16"})
    conf_df = pd.DataFrame(conf_rows)

    line = (
        alt.Chart(conf_df)
        .mark_line(opacity=0.85)
        .encode(
            x=alt.X("frame:Q", title="Frame"),
            y=alt.Y("confidence:Q", title="Action confidence",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=["pytorch_fp16", "coreml_fp16"],
                    range=["steelblue", "tomato"],
                ),
            ),
            tooltip=["frame:Q", "confidence:Q", "model:N"],
        )
    )

    layers = [line]

    # Overlay thin green rules at every GT-positive frame.
    if events:
        gt_df = pd.DataFrame([{"frame": r["frame"]} for r in rows if r["gt"] == 1])
        if not gt_df.empty:
            gt_rules = (
                alt.Chart(gt_df)
                .mark_rule(color="green", strokeWidth=1, opacity=0.25)
                .encode(x="frame:Q")
            )
            layers.append(gt_rules)

    chart = (
        alt.layer(*layers)
        .properties(
            title=(
                f"{video_path.stem}  |  "
                f"agreement {100 * n_agree / n:.1f}%  |  "
                f"mean|Δconf| {sum(diffs) / len(diffs):.4f}"
            ),
            width=900,
            height=300,
        )
    )

    output_chart.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_chart))
    print(f"Chart saved to: {output_chart}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch fp32 vs Core ML fp16 model on a video."
    )
    parser.add_argument("video",       type=pathlib.Path,
                        help="Path to the input video file")
    parser.add_argument("--pt",        type=pathlib.Path, required=True,
                        help="PyTorch portable artifact (.pt) from export_pytorch.py")
    parser.add_argument("--mlpackage", type=pathlib.Path,
                        default=pathlib.Path("exports/fastvit_t12.mlpackage"),
                        help="Core ML .mlpackage (default: exports/fastvit_t12.mlpackage)")
    parser.add_argument("--output",    type=pathlib.Path,
                        default=pathlib.Path("compare_chart.html"),
                        help="Output HTML chart path (default: compare_chart.html)")
    args = parser.parse_args()

    for label, p in [("video", args.video), ("--pt", args.pt), ("--mlpackage", args.mlpackage)]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    compare_on_video(args.video, args.pt, args.mlpackage, args.output)


if __name__ == "__main__":
    main()
