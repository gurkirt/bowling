# SAM 3 label engine

`autolabel/sam3_tracks.py` turns the 319 hand-drawn bowler boxes in `videos/*.json` into
per-frame tracks for every person in every clip, plus a stump probe. It is an
**offline data engine** — SAM 3 never ships in the app; it labels the data that
trains the small model that does.

## Setup (the 5090 box)

Blackwell needs a CUDA 12.8+ build of torch, which matches SAM 3's own instructions:

```bash
conda create -n sam3 python=3.12 && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .
pip install opencv-python numpy pillow
```

Checkpoints are gated: request access to `facebook/sam3` on Hugging Face and
`huggingface-cli login` before the first run.

Copy `videos/` and `autolabel/sam3_tracks.py` across; nothing else is needed (it has no
imports from this repo). Run it from the directory that holds `videos/`.

## Run order

```bash
# 1. Confirm the output schema on one clip before committing GPU hours.
python autolabel/sam3_tracks.py --probe --limit 1

# 2. Eyeball five clips. Writes annotated mp4s to sam3_out/viz/.
python autolabel/sam3_tracks.py --limit 5 --viz

# 3. Full run. Resumable — existing sidecars are skipped unless --overwrite.
python autolabel/sam3_tracks.py

# 4. Summary stats over the sidecars.
python autolabel/sam3_tracks.py --report
```

`--probe` matters: `parse_outputs()` is the one version-sensitive function, and
probe prints the real structure plus whether parsing succeeded. If SAM 3.1 or a
transformers port changes the keys, that function is the only thing to update.

## The three passes

| Pass | Prompt | Propagation |
|---|---|---|
| `bowler` | the annotated box at `middle_frame`, as normalized xywh | forward + backward, windowed to the delivery |
| `person` | text, `"person"` | forward + backward over the whole clip |
| `stumps` | text, `"cricket stumps"`, image model | 5 sampled frames — camera and stumps are static |

Propagation runs on symlinked frame folders rather than a direction flag, so
direction and extent are controlled by what SAM 3 is shown. The bowler window is
the event ±`--bowler_margin` seconds (default 2.5), clamped to the midpoints
between adjacent deliveries — without that, the 18-delivery clip would propagate
one bowler box across all 18. Windowing also cuts the run from ~299k to ~196k
frame-inferences.

Backward propagation is a separate session, so its object ids are unrelated to the
forward ones. The bowler pass seeds a single object and merges blindly; the person
pass aligns the halves by IoU on the shared prompt frame (`_stitch`).

Boxes are derived from the returned **masks**, so the pipeline depends only on
"object id → mask" and not on whether a given version returns boxes or in what
format. Everything is reported in original-resolution pixels regardless of
`--max_side`.

## Output

One sidecar per clip in `sam3_out/`; `videos/*.json` is never touched.

```jsonc
{
  "video": "IMG_8446.MOV", "width": 1080, "height": 1920, "total_frames": 146,
  "bowler_tracks": [{ "event_index": 0, "prompt_frame": 46, "window": [0, 126],
                      "coverage": 1.0, "boxes": {"0": [x1,y1,x2,y2], ...} }],
  "person_tracks": { "1": { "boxes": {...}, "n_frames": 146 } },
  "stumps": { "prompt": "cricket stumps", "samples": [...] },
  "eval": { "per_event": [{ "best_person_iou": 0.0, "n_people_on_frame": 0 }] }
}
```

## The number to read first

`--report` prints **recall@IoU>0.5** for "can a generic `person` prompt recover the
hand-annotated bowler". That decides the runtime design:

- **High** → a geometric bowler-selection rule over person tracks is viable, and
  the app can pick the bowler out of the crowd at inference time.
- **Low** → fall back to the corridor ROI from stump/keeper calibration, and use
  SAM 3 purely to build bowler-centred training crops.

Also worth checking before trusting the run: bowler `coverage` should sit near 1.0
(low values mean propagation is losing the bowler mid-run-up), and the stump
detection rate tells you whether `"cricket stumps"` is a concept SAM 3 actually
has. If it is weak, try `--stump_text wicket` or `--stump_text "wooden stumps"`
before concluding stump detection needs its own annotated model.
