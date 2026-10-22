# CricShorts — bowling delivery detection

A phone on a tripod stands where the umpire would, behind the bowler's-end stumps. A
small on-device classifier spots the moment the bowler delivers, and the
[CricReel](CricReel/README.md) iOS app cuts a clip of each ball for scoring and
highlights. This repo holds everything upstream of the app: labelling tools, a SAM 3
auto-labelling engine, training-data builders, training, analysis and export.

There are two model inputs:

- **Fixed crop** (`exp21`, deployed until s04): bottom 68% of the frame, centred square, 256×256.
- **Stump crops** (`s04`, being deployed): two crops anchored on the bowler's-end stumps,
  one each side, at a scale set by the stump height, 160×256 each, scored as a batch of 2.
  The apps ask the user to place a box on the stumps when the camera opens. See
  [Stump crops](#stump-crops).

## Repo layout

```
labeling/            interactive OpenCV labelling tools
  video_labeler.py     deliveries (start/end frames) + bowler box
  label_batter.py      striker / non-striker, picked from SAM3 person tracks
  label_stumps.py      bowler's-end stumps, with the two crops previewed
autolabel/           SAM 3 label engine (offline, GPU)
  sam3_tracks.py       person tracks, bowler track, stump and ball probes -> sam3_out/
  review_bowler_iou.py manual QC of SAM3 person tracks vs hand bowler boxes
  sam3_reval.py        recompute sidecar eval without rerunning SAM3
  sam3_viz.py          re-render overlay videos from sidecars
  run_sam3.sh          two-GPU sharded full run
labels/              progress of the review/labelling tools (resumable state)
analysis/            evaluation and comparison scripts (see Analysis)
docs/                design notes: sam3_labeling.md, export_coreml.md, plan.md (CricReel)

training_set.py      fixed-crop frames  -> training_set/frames/
stump_crops.py       stump-crop geometry (shared by labelling, data builder, app)
stump_training_set.py stump-crop frames -> training_set_stumps/frames/
dataset.py           on-the-fly dataset over either frames folder
train_classifier.py  timm classifier training (focal loss, EMA, folds / _val split)
modellib.py          inference helpers used by extract_action and analysis
extract_action.py    run a model over one video, cut 2 s clips, write a chart
export_pytorch.py    strip a checkpoint to a portable models/*.pt
export_executorch.py export to ExecuTorch .pte or Core ML .mlpackage
run_experiments.sh   current training launcher (stump-crop and fixed-crop runs)
run_ablation_experiments*.sh, QUICK_START_ABLATION.sh, run_train.sh   earlier rounds
experiments/         logs/<round>/<tag>.log and trainings/<round>/<tag>/ for every run

videos/              clips + one annotation JSON each (the labels live here)
videos-duplicates/   clips removed from videos/ because they duplicate another clip
CricReel/, criclips/ iOS apps
```

Run everything from the repo root; scripts in sub-folders find the root modules
themselves. Generated data (`training_set*/`, `sam3_out/`, `experiments/trainings/`, `*.MOV/.mp4`)
is git-ignored; `experiments/logs/` is small and can be committed.

## Setup

```bash
pip install -r requirements.txt        # torch, timm, opencv, tensorboard, ...
```

SAM 3 needs its own environment on a CUDA 12.8+ GPU box — see
[docs/sam3_labeling.md](docs/sam3_labeling.md). Export needs `coremltools` (and
`executorch` for `.pte`) on macOS — see [Export](#export).

## Annotation format

One `videos/<stem>.json` per clip. `temporal_events` are the deliveries;
`bounding_boxes` are keyed by frame number, and every box sits on the delivery's
`middle_frame` (the frame the bowler box was drawn on):

```jsonc
{
  "video": "IMG_8556.MOV",                // must match the file next to it
  "total_frames": 125,
  "temporal_events": [
    {"start_frame": 32, "end_frame": 42, "middle_frame": 37, "class": "bowling"}
  ],
  "bounding_boxes": {
    "37": [
      {"x1": 179, "y1": 1767, "x2": 755, "y2": 3267, "class": "bowler"},
      {"x1": 1013, "y1": 1817, "x2": 1105, "y2": 1998, "class": "striker", "sam3_person_id": "3"},
      {"x1": 1429, "y1": 1822, "x2": 1674, "y2": 2578, "class": "non_striker", "sam3_person_id": "0"},
      {"x1": 950, "y1": 2432, "x2": 1139, "y2": 2950, "class": "stumps"}
    ]
  }
}
```

Coordinates are pixels in the correctly rotated frame. Always read frames through
`autolabel.sam3_tracks.open_video`, which applies the rotation iPhone files need.
`sam3_person_id` links a box to the whole SAM3 track in `sam3_out/<stem>.json`.
Clips with no deliveries (e.g. app false triggers) have an empty `temporal_events`
and only a `stumps` box.

Validation clips end in `_val` (`IMG_9369_val.MOV`); `--val_split val_suffix` holds
exactly those out.

## Workflow

### 1. Label deliveries and the bowler

```bash
python labeling/video_labeler.py --video_dir videos
```

Pass 1 marks deliveries: arrows step a frame, `w`/`x` step 10, `s`/`e` set start/end,
`n` next delivery, `d` delete last, `l` list, `q` save. Pass 2 asks for a bowler box
on each delivery's middle frame: drag to draw, `u` undo, `c` clear, `q` save.

To fix the deliveries of a labelled video without losing its other boxes:

```bash
python labeling/video_labeler.py --relabel IMG_9398_val
```

It backs up the JSON to `labels/relabel_backup/`, opens with the saved deliveries
(`j` jumps to the next one, `r` removes the one nearest the current frame, then mark
the replacement with `s`/`e`/`n`), and on `q` asks for a new bowler box on each changed
delivery's middle frame. The stumps/striker/non-striker boxes move there with it.

### 2. Auto-label with SAM 3 (GPU box)

```bash
python autolabel/sam3_tracks.py --probe --limit 1    # check the output schema first
python autolabel/sam3_tracks.py                      # full run, resumable -> sam3_out/
python autolabel/sam3_tracks.py --report             # recall of the bowler from "person" tracks
python autolabel/review_bowler_iou.py                # QC low-IoU events -> labels/sam3_review.json
```

Tracks every person in every clip, prompts SAM 3 with the bowler box, and probes for
stumps and ball. Details and the output schema: [docs/sam3_labeling.md](docs/sam3_labeling.md).

### 3. Label striker and non-striker

```bash
python labeling/label_batter.py --role striker
python labeling/label_batter.py --role non_striker
```

Shows the bowler frame with every SAM3 person track numbered by size. Pick the role
with a click or `1`-`9` (`Tab` cycles), `Enter` accepts, `m` draws by hand, `x` not
visible, `s` skip, arrows step frames for context, `q` quit. The other role, once
labelled, is shown in magenta and excluded. Progress: `labels/batter_review.json`.

### 4. Label the bowler's-end stumps

```bash
python labeling/label_stumps.py
```

Proposes the box from an earlier delivery of the same clip, else SAM3's stump
detection, and draws both stump crops from it — the crop on the bowler's side turns
red if the bowler sticks out, so a bad stump box is obvious. `Enter` accept, `m`
draw (tight around the three stumps, base on the ground), `x` not visible, `o` bowler occluded at release (delivery left out of training), `s` skip,
`b` back to the previous item, `q` quit. Clips with no deliveries get one item on their
middle frame. Progress: `labels/stumps_review.json`. To fix a clip labelled in an earlier
session, pass `--videos <stem>`: every delivery comes back with its saved box.

### 5. Build training data

```bash
python stump_training_set.py --workers 16      # stump crops -> training_set_stumps/
python training_set.py                         # fixed crop  -> training_set/   (exp21 pipeline)
python training_set.py --fps_map               # labels/fps_map.json, for multi-frame striding
```

Both write `<out>/frames/<video>/frame_<i>_<true|false>.jpg`; frames from
`start_frame - 2` to `end_frame` are positives. `stump_training_set.py` writes two
folders per clip, `<stem>__L` and `<stem>__R` (mirrored), and positives only on the
bowler's side, so the other side of every delivery frame is a hard negative. It also
writes `training_set_stumps/meta.json` with the geometry and the calibration overlay.
Both clear a clip's old frames before rewriting it, but neither deletes folders of
clips that leave `videos/` — remove those by hand. Deliveries marked occluded in
`label_stumps.py` (`o`) are left out of both sets, with 2 s either side.

### 6. Train

```bash
./run_experiments.sh      # GPU 0: stump crops (s03), GPU 1: exp21's fixed crop (exp22)
```

Both use exp21's recipe. Set `ROUND` at the top of the script for a new round; each run
writes `experiments/logs/<round>/<tag>.log` and `experiments/trainings/<round>/<tag>/`.
Rounds so far: `round1` (exp1-4), `round2` (exp5-16), `round3` (exp17-21, exp21 is the
deployed model), `stumps` (s01-s02, before the seek fix), `round4_seekfix`.

`train_classifier.py` options worth knowing:

| Flag | Meaning |
|---|---|
| `--data_dir` | frames folder (`training_set/frames` or `training_set_stumps/frames`) |
| `--val_split val_suffix` | validate on `*_val` clips (used by exp21 and the stump runs) |
| `--fold`, `--num_folds` | otherwise, hold out one fold of clips |
| `--crop_margin 0.15` | frames were saved with this context margin; the model sees the centre crop |
| `--jitter_scale`, `--jitter_shift` | with a margin: random ±10% rescale and ±4% shift of the training crop, mimicking an imperfect stump calibration |
| `--no_hflip` | no random flip (stump crops are already mirrored to one side) |
| `--num_frames`, `--target_fps` | early-fused multi-frame input with fps-normalised stride |
| `--class_balance focal`, `--focal_alpha`, `--focal_gamma`, `--neg_skip_prob` | class imbalance |
| `--ema` | evaluate and export EMA weights |

The two crop folders of a clip always land in the same fold. Logs go to TensorBoard
under the output directory.

### 7. Evaluate on a video

```bash
python extract_action.py videos/IMG_9398_val.MOV models/fastvit_sa12_exp21.pt
```

Runs the fixed-crop model frame by frame, cuts a 2 s clip wherever 4 consecutive
frames are positive, and writes `chart.html` (prediction vs label per frame). See
also [Analysis](#analysis).

### 8. Export

```bash
python export_pytorch.py experiments/trainings/<round>/<tag>/.../best_model.pth --output models/<name>.pt
python export_executorch.py models/<name>.pt --backend mlpackage   # Core ML for CricReel
python export_executorch.py models/<name>.pt --compare             # .pte XNNPACK / CoreML / mlpackage parity
```

Stump-crop model (s04), as bundled in CriClips and CricReel:

```bash
python export_pytorch.py experiments/trainings/round4_seekfix/s04_fastvit_sa12_stumps_i01_256x160/*/best_model.pth \
    --output models/fastvit_sa12_s04.pt --stump_geometry 2.3,0.1,3.1,0.9
python export_executorch.py models/fastvit_sa12_s04.pt --backend mlpackage --batch 2   # on the Mac
cp -R models/fastvit_sa12_s04.mlpackage criclips/CriClips/
cp -R models/fastvit_sa12_s04.mlpackage CricReel/CricReel/
```

`--batch 2` fixes the input at [2, 3, 256, 160] (left crop, mirrored right crop); output
`probs` is [2, 2] and the app takes the higher action score. The crop geometry is in the
`.pt` (`stump_crops`) and the `.mlpackage` metadata; the apps hard-code it in
`StumpCalibration.swift`, which also holds the default stump box (median over the
`criclip_*` recordings).

`export_pytorch.py` keeps only config, EMA weights, normalisation and class names
(loadable with just `torch` + `timm`). `export_executorch.py` needs macOS with
`coremltools`; `.pte` targets also need `executorch` (Apple Silicon wheel; build from
source on Intel). [docs/export_coreml.md](docs/export_coreml.md) describes the Core ML
model's input/normalisation contract; its `export_coreml.py` commands predate
`export_executorch.py --backend mlpackage`, which replaces them.

## Stump crops

The bowler releases right beside the bowler's-end stumps: left of them over the wicket,
right of them round the wicket. Measured from the 337 hand-confirmed stump boxes:

- The bowler is always clearly on one side — 250 left, 64 right — never crossing the
  stump centre by more than 0.03 stump heights over a whole delivery.
- At release the bowler is 1.8–3.1 stump heights tall, so scaling by the stump height
  removes zoom differences between clips.
- One crop per side, 2.6 × 4.0 stump heights (`stump_crops.py`: 2.3 out, 0.3 past the
  stump centre, 3.1 above and 0.9 below the stump base), holds the bowler in every
  frame of 100% of deliveries (checked against SAM3 bowler tracks), and 96% even with
  the stump box ±10% in height or shifted 0.1 stump heights.
- The bowler fills a median 20% of the crop (5th percentile 15%) vs 17% (10%) in the
  fixed crop. The gain is consistency of scale and position, plus a natural hard
  negative from the other side.
- Two crops at 208×320 cost about the same as one 256×256. One classifier scores both
  as a batch of two (the right crop mirrored).

`python stump_crops.py` re-checks containment against the current labels.

**Calibration in the app.** The stumps are static, so they are fixed once per session:
the app overlays a stump outline and the user moves the tripod (keeping the phone
upright, checked with the accelerometer/gyro) until it covers the stumps; the two crops
are then fixed rectangles. People already frame clips this way, so the outline sits at
the medians in `training_set_stumps/meta.json`: stump centre 0.50 of frame width, base
0.79 of frame height, height 0.158 of frame height, width 0.38 of its height. Pixels of
a crop that fall outside the frame are black, in training and in the app.

## Analysis

| Script | What it does |
|---|---|
| `python stump_crops.py` | stump-crop containment and bowler share vs current labels |
| `python analysis/motion_gate_eval.py` | can frame differencing around the non-striker gate the classifier? (on trimmed clips: 60% of frames still run at 98% recall, so not adopted) |
| `python analysis/compare_models.py <video> ...` | PyTorch fp32 vs Core ML fp16 on one video, chart to `analysis/compare_chart.html` |
| `python analysis/score_debug_frame.py` | PyTorch vs Core ML on `analysis/debug_frame*.png` |
| `python analysis/debug.py` | classify a single image |
| `python analysis/analyze_ablation_results.py [progress]` | summarise round-1 runs in `experiments/` |

## Data snapshot (2026-10-21)

- 308 annotated clips in `videos/`; 364 bowler, 337 stumps, 316 striker, 234
  non-striker boxes. 8 `criclip_*` clips are app false triggers with no deliveries.
- `training_set_stumps/`: 297 training clips (101,320 crops, 3,444 positive) and 10
  `_val` clips (8,420 crops, 143 positive, 12 deliveries).
- `GullyCrickett_live_H2QenGyqSWU_cut.json` has no video file. `IMG_9398_val.MOV_00`
  (a cut of `IMG_9398_val`'s third delivery) was moved to `videos-duplicates/` so the
  delivery is not in both train and validation; exp21 was trained with it.
