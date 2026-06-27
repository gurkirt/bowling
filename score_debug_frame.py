#!/usr/bin/env python3
"""
score_debug_frame.py
Run debug frames through both PyTorch fp16 and CoreML fp16 models.

  debug_frame.png      – already 256×256 pre-cropped (no crop needed)
  debug_frame_full.png – raw 1080×1920 portrait frame (crop pipeline applied)

The cropped/resized version of the full frame is saved as
  debug_frame_full_cropped.png
so you can visually confirm it matches debug_frame.png.
"""
import pathlib
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import coremltools as ct

PT  = pathlib.Path("models/fastvit_sa12_exp07.pt")
ML  = pathlib.Path("exports/fastvit_t12.mlpackage")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Crop helper (mirrors modellib.crop exactly) ───────────────────────────────
def crop_frame(pil_img: Image.Image) -> Image.Image:
    """Remove top 32 %, centered square crop, return PIL RGB."""
    arr = np.array(pil_img)          # (H, W, 3)
    h, w = arr.shape[:2]
    top_offset = int(h * 0.32)
    new_h      = h - top_offset
    side       = min(new_h, w)
    cx         = w // 2
    cy         = top_offset + new_h // 2
    left       = max(cx - side // 2, 0)
    right      = left + side
    top        = max(cy - side // 2, top_offset)
    bottom     = top + side
    cropped    = arr[top:bottom, left:right]
    return Image.fromarray(cropped)

# ── Load & pre-process images ─────────────────────────────────────────────────
images = {}

# 1) Already-cropped 256×256
p = pathlib.Path("debug_frame.png")
if p.exists():
    img = Image.open(p).convert("RGB")
    print(f"[1] {p}  →  {img.size} (no crop needed)")
    images["debug_frame (256×256, pre-cropped)"] = img.resize((256, 256), Image.BILINEAR)

# 2) Full 1080×1920 portrait frame — apply crop then resize
p = pathlib.Path("debug_frame_full.png")
if p.exists():
    raw = Image.open(p).convert("RGB")
    print(f"[2] {p}  →  raw {raw.size}")
    cropped = crop_frame(raw)
    print(f"    after crop: {cropped.size}")
    resized = cropped.resize((256, 256), Image.BILINEAR)
    resized.save("debug_frame_full_cropped.png")
    print(f"    saved cropped+resized → debug_frame_full_cropped.png")
    images["debug_frame_full (cropped→256×256)"] = resized

if not images:
    raise FileNotFoundError("Neither debug_frame.png nor debug_frame_full.png found.")

tf = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ── Load PyTorch model ────────────────────────────────────────────────────────
print(f"\nLoading PyTorch model: {PT}")
ckpt = torch.load(str(PT), map_location="cpu", weights_only=False)
cfg  = ckpt.get("config") or vars(ckpt.get("args", object()))
mn   = cfg.get("model_name", "fastvit_sa12")
nc   = cfg.get("num_classes", 2)
h, w = cfg.get("input_height", 256), cfg.get("input_width", 256)
print(f"  backbone={mn}  classes={nc}  input={h}x{w}")

base = timm.create_model(mn, pretrained=False, num_classes=nc)
base.load_state_dict(ckpt["model_state_dict"])
try:
    from timm.utils import reparameterize_model
    base = reparameterize_model(base)
    print("  reparameterized for inference")
except Exception:
    pass
base = base.half().eval()

# ── Load CoreML model ─────────────────────────────────────────────────────────
print(f"\nLoading CoreML model: {ML}")
mlmodel  = ct.models.MLModel(str(ML))
spec     = mlmodel.get_spec()
in_name  = spec.description.input[0].name
out_name = spec.description.output[0].name
print(f"  input={in_name}  output={out_name}")

# ── Run inference on each image ───────────────────────────────────────────────
results = []
for label, img in images.items():
    t32 = tf(img).unsqueeze(0)                   # (1,3,256,256) float32
    t16 = t32.half()

    with torch.no_grad():
        probs = torch.softmax(base(t16), dim=1)[0]
    pt_no, pt_act = probs[0].item(), probs[1].item()

    preds    = mlmodel.predict({in_name: t32.numpy()})
    ml_probs = np.asarray(preds[out_name]).flatten()
    ml_no, ml_act = float(ml_probs[0]), float(ml_probs[1])

    results.append((label, pt_no, pt_act, ml_no, ml_act))

# ── Print summary table ───────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"{'Image / source':<38} {'model':<14} {'no_action':>10} {'action':>10}  decision")
print("-" * 80)
for label, pt_no, pt_act, ml_no, ml_act in results:
    print(f"{label:<38} {'PyTorch fp16':<14} {pt_no:>10.6f} {pt_act:>10.6f}  {'ACTION' if pt_act >= 0.5 else 'no_action'}")
    print(f"{'':<38} {'CoreML  fp16':<14} {ml_no:>10.6f} {ml_act:>10.6f}  {'ACTION' if ml_act >= 0.5 else 'no_action'}")
    diff = abs(pt_act - ml_act)
    print(f"{'':<38} {'|PT - ML|':<14} {abs(pt_no-ml_no):>10.6f} {diff:>10.6f}")
    print("-" * 80)
print("=" * 80)

