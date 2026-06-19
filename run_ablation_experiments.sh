#!/bin/bash
# Ablation: 4 experiments, fold 3, 2 per GPU (GPU 0 and GPU 1).
# Focus: high recall of action frames + good accuracy.
# Axes ablated: negative sampling rate (neg_skip_prob), focal alpha, model size.
export HF_HUB_OFFLINE=1
train_model() {
  # args: $1=gpu  $2=model  $3=alpha  $4=gamma  $5=neg_skip  $6=tag
  CUDA_VISIBLE_DEVICES=$1 python train_classifier.py \
    --model_name "$2" \
    --batch_size 64 \
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
    --output_dir "trainings/$6" \
    > "ablation_logs/$6.log" 2>&1 
}

mkdir -p ablation_logs

#            gpu  model                     alpha gamma neg_skip tag
train_model   0   mobilenetv4_conv_small    0.75  2.0   0.5      exp1_small_a75_g2_ns50 &
train_model   1   mobilenetv4_conv_small    0.75  2.0   0.8      exp2_small_a75_g2_ns80 &
train_model   0   mobilenetv4_conv_medium   0.75  2.0   0.5      exp3_medium_a75_g2_ns50 &
train_model   1   mobilenetv4_conv_medium   0.90  2.0   0.8      exp4_medium_a90_g2_ns80 &

wait
echo "All experiments finished. Logs in ablation_logs/, models in trainings/."
