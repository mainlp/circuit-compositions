chmod u+x ./download_pcfg.sh
./download_pcfg.sh

echo "Transforming data to a common format"
mkdir base_tasks
./pcfgs_to_csv.sh PCFGS/pcfgset/train.src PCFGS/pcfgset/train.tgt base_tasks/train.csv
./pcfgs_to_csv.sh PCFGS/pcfgset/dev.src PCFGS/pcfgset/dev.tgt base_tasks/dev.csv
./pcfgs_to_csv.sh PCFGS/pcfgset/test.src PCFGS/pcfgset/test.tgt base_tasks/test.csv
echo "Finished..."



