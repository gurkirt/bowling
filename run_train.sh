
function train_model() {
    
CUDA_VISIBLE_DEVICES=$1 python train_classifier.py \
  --model_name mobilenetv3_small_075 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 30 \
  --fold $2 \
  --input_width 256 \
  --input_height 256 \
  --class_balance 'focal' 

}

train_model 0 1  
# train_model 1 2 &
# train_model 2 3 &
# train_model 0 4 &
# train_model 1 5 &
# train_model 2 6 &
# train_model 0 7 &
# train_model 1 8 &
# train_model 2 9 &
# train_model 4 0