save_path="../model_checkpoints"
wandb_path="../"
export PYTHONPATH="../"

python ../train.py \
	--lr 7e-5 \
	--hidden_size 512 \
	--layers 6 \
	--epochs 25 \
	--dropout 0.2 \
	--eval \
	--save_path=$save_path \
	--wandb_path=$wandb_path
