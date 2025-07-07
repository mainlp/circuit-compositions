save_path="../model_checkpoints"
cache_dir="../../data/cached_probabilities" 
export PYTHONPATH="../"

python ../cache_base_model_output.py \
    --cache_dir=$cache_dir \
    --model_dir=$save_path

