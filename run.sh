#!/usr/bin/env bash

dataset=$1
version=$2

echo "Dataset: "$dataset""
echo "Version: "$version""

bash prepare.sh $dataset $version >> /dev/null

cd ./src

# train
python main_optimize.py --dataset=$dataset --ver=$version >> opt_"$version".log

# inference
python main_optimize.py --dataset=$dataset --mode=test --ver=$version >> /dev/null

# evaluate
cd ../evaluate

python prepare.py $dataset $version >> /dev/null
python eval.py $dataset $version >> ../output/"$dataset"-"$version".txt