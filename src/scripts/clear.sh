#!/bin/bash
echo "dataset: $1"
echo "version: $2"

rm -rf ../../dump/"$1"/optimize-"$2"
rm -rf ../../log/"$1"/optimize-"$2"
rm -rf ../../evaluate/eval_dump/adv_models/unigram_lr_"$2"_"$1".bin
rm -rf ../../output/"$1"-"$2"
