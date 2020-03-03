#!/usr/bin/env bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/user/miniconda/envs/py36/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/user/miniconda/envs/env/etc/profile.d/conda.sh" ]; then
        . "/home/user/miniconda/envs/env/etc/profile.d/conda.sh"
    else
        export PATH="/home/user/miniconda/envs/env/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate env

dataset=$1
version=$2

echo "Dataset: "$dataset""
echo "Version: "$version""

# prepare
bash /code/qwh/model_agnostic_ST/prepare.sh $dataset

# train
python /code/qwh/model_agnostic_ST/src/main_pretrain.py --dataset=$dataset --model_version=$version
# python /code/qwh/model_agnostic_ST/src/main_warmup.py --dataset=$dataset --model_version=$version
# python /code/qwh/model_agnostic_ST/src/main_optimize.py --dataset=$dataset --model_version=$version

# inference
# python /code/qwh/model_agnostic_ST/src/main_optimize.py --dataset=$dataset --mode=test --model_version=$version