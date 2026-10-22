#!/usr/bin/env bash
# Train the bowling classifier. Everything a run writes goes under experiments/:
#   experiments/logs/<round>/<tag>.log
#   experiments/trainings/<round>/<tag>/<run dir>/best_model.pth
#
# Both recipes are exp21's: focal 0.75/2.0, neg_skip 0.35, EMA 0.999, 30 epochs,
# val_suffix split (validate on the *_val clips). They differ only in the input:
#   train_fixed   exp21's crop: bottom 68% of the frame, centre square, 256x256
#                 (training_set.py -> training_set/frames)
#   train_stumps  the two stump-anchored side crops, 320x208, calibration jitter from
#                 the saved 0.15 margin, no random h-flip (right crops are already
#                 mirrored)  (stump_training_set.py -> training_set_stumps/frames)
#
# Build both sets from the same annotations first:
#   python training_set.py --video_dir videos --output_dir training_set
#   python stump_training_set.py --workers 16
#
# RUN:    ./run_experiments.sh
# WATCH:  tail -f experiments/logs/$ROUND/*.log
# STOP:   pkill -f train_classifier.py
export HF_HUB_OFFLINE=1

# round4_seekfix: labels re-indexed to decoded frames (see analysis/seek_offsets.txt),
# Gully clip included, occluded deliveries left out of both sets.
ROUND=round4_seekfix
mkdir -p "experiments/logs/$ROUND"

common_args() {
  echo --batch_size 64 --lr 1e-4 --epochs 30 --num_workers 8 \
       --class_balance focal --focal_alpha 0.75 --focal_gamma 2.0 --neg_skip_prob 0.35 \
       --val_split val_suffix --ema --ema_decay 0.999
}

train_fixed() {
  # args: $1=gpu $2=model $3=tag
  CUDA_VISIBLE_DEVICES=$1 python train_classifier.py $(common_args) \
    --data_dir training_set/frames \
    --model_name "$2" \
    --input_height 256 --input_width 256 \
    --output_dir "experiments/trainings/$ROUND/$3" \
    > "experiments/logs/$ROUND/$3.log" 2>&1
}

train_stumps() {
  # args: $1=gpu $2=model $3=tag
  CUDA_VISIBLE_DEVICES=$1 python train_classifier.py $(common_args) \
    --data_dir training_set_stumps/frames \
    --model_name "$2" \
    --input_height 320 --input_width 208 \
    --crop_margin 0.15 --jitter_scale 0.1 --jitter_shift 0.04 --no_hflip \
    --output_dir "experiments/trainings/$ROUND/$3" \
    > "experiments/logs/$ROUND/$3.log" 2>&1
}

# Earlier rounds (logs/trainings moved under experiments/):
#   stumps:  train_stumps 0 fastvit_sa12 s01_fastvit_sa12_stumps
#            train_stumps 1 mobilenetv4_conv_small s02_mnv4s_stumps
#   round3:  run_ablation_experiments3.sh (exp17-21; exp21 = train_fixed above)

train_stumps 0 fastvit_sa12 s03_fastvit_sa12_stumps &
train_fixed  1 fastvit_sa12 exp22_fastvit_sa12_fixedcrop &
wait
echo "$ROUND finished. Logs in experiments/logs/$ROUND/, models in experiments/trainings/$ROUND/."
