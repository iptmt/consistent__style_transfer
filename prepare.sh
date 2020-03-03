#!/bin/bash

dataset=$1
# basedir=/code/qwh/model_agnostic_ST # default .
basedir=.

# prepare working directories
for dir in "$basedir"/data "$basedir"/dump "$basedir"/output "$basedir"/log
do
    if [ ! -f "$dir" ]; then
        mkdir $dir
    fi
    if [ ! -f "$dir"/"$dataset" ]; then
        mkdir "$dir"/"$dataset"
    fi
done

# prepare evaluate directories
if [ ! -f "$basedir"/evaluate/eval_dump ]; then
    mkdir "$basedir"/evaluate/eval_dump
fi

if [ ! -f "$basedir"/evaluate/eval_tmp ]; then
    mkdir "$basedir"/evaluate/eval_tmp
fi

if [ ! -f "$basedir"/evaluate/eval_dump/adv_models ]; then
    mkdir "$basedir"/evaluate/eval_dump/adv_models
fi