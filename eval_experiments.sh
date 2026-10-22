#!/usr/bin/env bash
# Score finished runs on the *_val clips with analysis/val_errors.py and save the
# results next to each run's checkpoint:
#   experiments/trainings/<round>/<tag>/val_errors.txt   (report)
#   experiments/trainings/<round>/<tag>/val_errors.json  (per-crop scores)
# Every run is scored on the same _val clips and frames, so the per-frame F1 lines and
# the delivery reports compare directly. Input size and model come from the checkpoint.
#
# RUN:  ./eval_experiments.sh              (exp22, s03, s04 on GPU 1)
#       GPU=0 ./eval_experiments.sh s04    (any runs from evaluate() below)
export HF_HUB_OFFLINE=1

ROUND=round4_seekfix
T="experiments/trainings/$ROUND"

best() {
  # newest best_model.pth under a run tag
  ls -t "$T/$1"/*/best_model.pth 2>/dev/null | head -1
}

score() {
  # args: $1=tag $2=val_errors.py model flag (--checkpoint | --baseline) [$3=training set dir]
  local ck
  ck=$(best "$1")
  [ -n "$ck" ] || { echo "$1: no best_model.pth" >&2; return 1; }
  echo "$1: $ck"
  CUDA_VISIBLE_DEVICES=${GPU:-1} python analysis/val_errors.py \
    "$2" "$ck" --frames "${3:-training_set_stumps}/frames" \
    --out "$T/$1/val_errors.json" > "$T/$1/val_errors.txt" 2>&1
}

evaluate() {
  case "$1" in
    exp22) score exp22_fastvit_sa12_fixedcrop --baseline ;;  # full frames; --frames only lists the _val clips
    s03)   score s03_fastvit_sa12_stumps --checkpoint training_set_stumps ;;
    s04)   score s04_fastvit_sa12_stumps_i01_256x160 --checkpoint training_set_stumps_i01 ;;
    *)     echo "unknown run: $1" >&2; return 1 ;;
  esac
}

[ $# -gt 0 ] || set -- exp22 s03 s04
for name in "$@"; do
  evaluate "$name"
done
grep -H "per frame" "$T"/*/val_errors.txt
