save_path="../model_checkpoints"
cache_dir="../../data/cached_probabilities" 
export PYTHONPATH="../"

python ../comp_rep/pruning/calculate_mean_ablation_values.py \
    --model_path=$save_path \
    --n=50000 \
    --run_all

