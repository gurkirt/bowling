#!/bin/bash
# Ablation round 2: 16 experiments, fold 3, run 2 at a time (one per GPU: 0 and 1).
# Hardware: 2x RTX 2080 (11 GB VRAM each) -> one experiment per GPU at a time.
#
# What we learned from round 1 (see ablation_logs/exp1..4):
#   - medium > small (mainly higher recall)
#   - neg_skip_prob=0.5 (keep ~50% of negatives) beat 0.8 every time
#   - focal alpha=0.90 did not help over 0.75
#   - val F1/acc bounced a lot epoch-to-epoch  -> now using EMA (smoother, default on)
#
# This round explores three axes:
#   A) architecture (8 mobile/iPhone-friendly backbones that fit 11 GB at 256x256)
#      All are reparameterizable / mobile-optimized and export cleanly to
#      CoreML/ExecuTorch: Apple MobileOne, Apple FastViT, Google MobileNetV4.
#   B) "more negative frames": lower neg_skip_prob (0.25, 0.0) on the best backbones
#   C) focal alpha/gamma and EMA decay fine-tuning on mobilenetv4_conv_medium
#
# NOTE on pretrained weights: timm downloads pretrained weights from HF the first
# time each NEW architecture is used. If the training node is offline, pre-cache
# the weights first (run once on a machine with internet), e.g.:
#   python -c "import timm; [timm.create_model(m, pretrained=True) for m in \
#     ['mobilenetv4_conv_large','mobilenetv4_hybrid_medium','mobileone_s1', \
#      'mobileone_s2','mobileone_s4','fastvit_t12','fastvit_sa12','fastvit_sa24']]"
# then export HF_HUB_OFFLINE=1 below.

export HF_HUB_OFFLINE=1

mkdir -p ablation_logs2

train_model() {
  # args: $1=gpu $2=model $3=alpha $4=gamma $5=neg_skip $6=batch $7=ema_decay $8=tag
  CUDA_VISIBLE_DEVICES=$1 python train_classifier.py \
    --model_name "$2" \
    --batch_size "$6" \
    --lr 1e-4 \
    --epochs 30 \
    --fold 3 \
    --num_workers 8 \
    --input_width 256 \
    --input_height 256 \
    --class_balance focal \
    --focal_alpha "$3" \
    --focal_gamma "$4" \
    --neg_skip_prob "$5" \
    --ema \
    --ema_decay "$7" \
    --output_dir "trainings2/$8" \
    > "ablation_logs2/$8.log" 2>&1
}

# ---------------------------------------------------------------------------
# RUN:      ./run_ablation_experiments2.sh        (runs all 8 waves, 2 GPUs each)
# WATCH:    tail -f ablation_logs2/*.log
# STOP:     pkill -f train_classifier.py
# Config:   train_model  gpu  model  alpha gamma neg_skip batch ema_decay  tag
# Batch sizes fit 11 GB at 256x256 (no AMP); shrink if you OOM.
# ---------------------------------------------------------------------------

# --- Group A: mobile/iPhone-friendly architectures (MobileNetV4, MobileOne, FastViT) ---
echo "=== Wave 1 ==="
train_model 0 mobilenetv4_conv_large    0.75 2.0 0.5 48 0.999 exp01_mnv4large_ns50   &
train_model 1 mobilenetv4_hybrid_medium 0.75 2.0 0.5 48 0.999 exp02_mnv4hybridM_ns50 &
wait

echo "=== Wave 2 ==="
train_model 0 mobileone_s1              0.75 2.0 0.5 64 0.999 exp03_mobileone_s1     &
train_model 1 mobileone_s2              0.75 2.0 0.5 64 0.999 exp04_mobileone_s2     &
wait

echo "=== Wave 3 ==="
train_model 0 mobileone_s4              0.75 2.0 0.5 48 0.999 exp05_mobileone_s4     &
train_model 1 fastvit_t12               0.75 2.0 0.5 64 0.999 exp06_fastvit_t12      &
wait

echo "=== Wave 4 ==="
train_model 0 fastvit_sa12              0.75 2.0 0.5 64 0.999 exp07_fastvit_sa12     &
train_model 1 fastvit_sa24              0.75 2.0 0.5 48 0.999 exp08_fastvit_sa24     &
wait

# --- Group B: more negative frames (lower neg_skip) on medium + large ---
echo "=== Wave 5 ==="
train_model 0 mobilenetv4_conv_medium   0.75 2.0 0.25 64 0.999 exp09_medium_ns25     &
train_model 1 mobilenetv4_conv_medium   0.75 2.0 0.0  64 0.999 exp10_medium_ns00     &
wait

echo "=== Wave 6 ==="
train_model 0 mobilenetv4_conv_large    0.75 2.0 0.25 48 0.999 exp11_large_ns25      &
train_model 1 mobilenetv4_conv_large    0.75 2.0 0.0  48 0.999 exp12_large_ns00      &
wait

# --- Group C: focal/EMA fine-tuning on the round-1 winner (medium, ns=0.5) ---
echo "=== Wave 7 ==="
train_model 0 mobilenetv4_conv_medium   0.85 2.0 0.5 64 0.999  exp13_medium_a85      &
train_model 1 mobilenetv4_conv_medium   0.65 2.0 0.5 64 0.999  exp14_medium_a65      &
wait

echo "=== Wave 8 ==="
train_model 0 mobilenetv4_conv_medium   0.75 1.0 0.5 64 0.999  exp15_medium_g1       &
train_model 1 mobilenetv4_conv_medium   0.75 2.0 0.5 64 0.9995 exp16_medium_ema9995  &
wait

echo "All 16 experiments finished. Logs in ablation_logs2/, models in trainings2/."


