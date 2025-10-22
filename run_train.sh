
function train_model() {
    
CUDA_VISIBLE_DEVICES=$1 python train_classifier.py \
  --model_name $5 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 30 \
  --fold $2 \
  --input_width 256 \
  --input_height 256 \
  --class_balance 'focal' \
  --focal_alpha $3 \
  --focal_gamma $4 \

}

device=0
max_device=3
# for alpha in 0.25 0.5 0.75; do
#   for gamma in 0.75, 0.5; do
#     train_model $((device % max_device)) 3 $alpha $gamma &
#     device=$((device + 1))
#   done
# done
fold=3
alpha=0.75
gamma=1.0
for model in 'mobilenetv4_conv_small' 'mobilenetv4_conv_medium'; do
  train_model $((device % max_device)) $fold $alpha $gamma $model &
  device=$((device + 1))
done

# train_model 0 1 &
# train_model 1 2 &
# train_model 2 3 &
# train_model 0 4 &
# train_model 1 5 &
# train_model 2 6 &
# train_model 0 7 &
# train_model 1 8 &
# train_model 2 9 &
# train_model 0 0 &