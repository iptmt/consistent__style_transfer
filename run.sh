#!/usr/bin/env bash

dataset=$1
version=$2

echo "Dataset: "$dataset""
echo "Version: "$version""

bash prepare.sh $dataset $version

cd ./src

# train
nohup python main_optimize.py --dataset=$dataset --model_version=$version > opt_"$version".log 2>&1 &

# inference
nohup python main_optimize.py --dataset=$dataset --mode=test --model_version=$version > /dev/null 2>&1 &

# evaluate
cd ../evaluate

nohup python prepare.py $dataset $version > /dev/null 2>&1 &
nohup python eval.py $dataset $version >> ../output/"$dataset"-"$version".txt &