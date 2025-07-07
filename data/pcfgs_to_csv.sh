# Example use: pcfgs_to_csv.sh PCFGS/pcfgset/train.src PCFGS/pcfgset/train.tgt train.csv
src_file=$1
tgt_file=$2
save_path=$3

paste -d ";" $1 $2 > $save_path
