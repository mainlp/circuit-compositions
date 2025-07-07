DATA_SAMPLES=20000
TRAIN_RATIO=0.8

for function in "copy" "reverse" "shift" "echo" "swap_first_last" "repeat" "append" "prepend" "remove_first" "remove_second"
do
	echo Generating data for function: $function
	python ../code/comp_rep/data_prep/create_function_data.py --nr_samples $DATA_SAMPLES --subtask $function --train_ratio $TRAIN_RATIO
done
