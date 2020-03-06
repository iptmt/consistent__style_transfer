#!/usr/bin/env bash

dataset=$1
version=$2
cuda_id=$3

echo "Dataset: "$dataset""
echo "Version: "$version""
echo "CUDA ID: "$cuda_id""

bash prepare.sh $dataset $version >> /dev/null

cd ./src

# train
CUDA_VISIBLE_DEVICES=$cuda_id nohup python main_optimize.py --dataset=$dataset --ver=$version > opt_"$version".log 2>&1 &

# inference
CUDA_VISIBLE_DEVICES=$cuda_id nohup python main_optimize.py --dataset=$dataset --mode=test --ver=$version > /dev/null 2>&1 &

# evaluate
cd ../evaluate

nohup python prepare.py $dataset $version > /dev/null 2>&1 &
nohup python eval.py $dataset $version >> ../output/"$dataset"-"$version".txt &