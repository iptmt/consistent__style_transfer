import argparse
import torch

# base_dir = "/code/qwh/model_agnostic_ST"
base_dir = ".."

def fetch_args():
    parser = argparse.ArgumentParser(description="Parameters")

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, default="train") # "train" or "test"
    parser.add_argument('--ver', type=str, required=True)

    # args for file system
    parser.add_argument('--data_dir', type=str, default=f"{base_dir}/data")
    parser.add_argument('--dump_dir', type=str, default=f"{base_dir}/dump")
    parser.add_argument('--log_dir', type=str, default=f"{base_dir}/log")
    parser.add_argument('--out_dir', type=str, default=f"{base_dir}/output")

    # args for model setting
    parser.add_argument('--n_class', type=int, default=2, help="number of styles")
    parser.add_argument('--p_drop', type=float, default=0.1, help="dropout rate")

    parser.add_argument('--w_s', type=float, default=0.03, help="weight of CP")
    parser.add_argument('--w_c', type=float, default=0.5, help="weight of NT")

    parser.add_argument('--tau', type=float, default=0.01, help="annealling temperature")
    parser.add_argument('--gap', type=float, default=0., help="annealling temperature")
    parser.add_argument('--denoise', type=str, default=False, help="denoise the output")

    parser.add_argument('--device', type=str, default="0", help="device id")
    parser.add_argument('--restore_version', type=int, default=-1, help="version for restore trainer and it's state")
    
    args = parser.parse_args()

    if args.dataset == "yelp":
        args.max_len = 18
    elif args.dataset == "shen":
        args.max_len = 18
    elif args.dataset == "book":
        args.max_len = 30
    else:
        raise ValueError

    return args