ds=$1

echo $ds

python main_warmup.py --dataset=$ds --ver=0

python main_optimize.py --dataset=$ds --ver=full
python main_optimize.py --dataset=$ds --ver=full --mode=test
python main_optimize.py --dataset=$ds --ver=wo_s --w_s=0
python main_optimize.py --dataset=$ds --ver=wo_s --w_s=0 --mode=test
python main_optimize.py --dataset=$ds --ver=wo_c --w_c=0
python main_optimize.py --dataset=$ds --ver=wo_c --w_c=0 --mode=test
python main_optimize.py --dataset=$ds --ver=wo_adv --w_adv=0
python main_optimize.py --dataset=$ds --ver=wo_adv --w_adv=0 --mode=test
python main_optimize.py --dataset=$ds --ver=wo_bt --w_bt=0
python main_optimize.py --dataset=$ds --ver=wo_bt --w_bt=0 --mode=test
python main_optimize.py --dataset=$ds --ver=wo_allc --w_c=0 --w_bt=0
python main_optimize.py --dataset=$ds --ver=wo_allc --w_c=0 --w_bt=0 --mode=test

cd ../evaluate
python prepare.py $ds full
python prepare.py $ds wo_s
python prepare.py $ds wo_c
python prepare.py $ds wo_adv
python prepare.py $ds wo_bt
python prepare.py $ds wo_allc
