CUDA_VISIBLE_DEVICES=0 python autolabel/sam3_tracks.py --num_shards 2 --shard_index 0 --viz --target_fps 20 &
CUDA_VISIBLE_DEVICES=2 python autolabel/sam3_tracks.py --num_shards 2 --shard_index 1 --viz --target_fps 20 &
wait