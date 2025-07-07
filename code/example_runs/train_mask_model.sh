save_path="../model_checkpoints"
wandb_path="../"
cache_dir="../../data/cached_probabilities"  # Path to the cached probabilities of the base model
export PYTHONPATH="../"


# The hyperparameters are subtask-dependent and needs to be modified accordingly. Please refer to the paper for the exact configurations.

for subtask in "copy" "echo" "repeat" "reverse" "shift" "swap_first_lat" "append" "prepend" "remove_first" "remove_second" do
	python ../mask_train.py \
		--save_path=$save_path \
		--wandb_path=$wandb_path \
		--cache_dir=$cache_dir \
		--epochs=500 \
		--eval \
		--pruning_method="continuous" \
		--faithfulness_freq=20 \
		--subtask=$subtask \
		--max_temp=200 \
		--mask_lambda=1e-4 \
		--mask_initial_value=0.05 \
		--lr=1e-4 \
		--ablation_value="mean"
done
