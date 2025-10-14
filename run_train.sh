
function train_model() {
    
CUDA_VISIBLE_DEVICES=$1 python train_classifier.py \
  --model_name mobilenetv3_small_075 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 30 \
  --fold $2 \
  --input_width 256 \
  --input_height 256 

}

train_model 1 1 & 
train_model 2 2 &
train_model 3 3 &
train_model 4 4 &
train_model 5 5 &
train_model 0 6 &
train_model 1 7 &
train_model 2 8 &
train_model 3 9 &
train_model 4 0