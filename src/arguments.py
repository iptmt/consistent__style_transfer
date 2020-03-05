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
    parser.add_argument('--d_embed', type=int, default=128, help="embedding dim for seq2seq")
    parser.add_argument('--d_enc_hidden', type=int, default=256, help="hidden dim for encoder")
    parser.add_argument('--d_dec_hidden', type=int, default=512, help="hidden dim for decoder")
    parser.add_argument('--n_enc_layer', type=int, default=1, help="number of layers of encoder")
    parser.add_argument('--n_dec_layer', type=int, default=1, help="number of layers of decoder")
    parser.add_argument('--n_class', type=int, default=2, help="number of styles")
    parser.add_argument('--p_drop', type=float, default=0.1, help="dropout rate")

    parser.add_argument('--alpha', type=float, default=0.05, help="weight of CP")
    parser.add_argument('--beta', type=float, default=0.1, help="weight of NT")
    parser.add_argument('--gamma', type=float, default=0.1, help="weight of STI")

    parser.add_argument('--tau', type=float, default=0.1, help="annealling temperature")
    parser.add_argument('--gap', type=float, default=0., help="annealling temperature")

    # args for training options
    # parser.add_argument('--batch_size', type=int, default=200, help="batch size of sentences for each iteration")
    # parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")

    parser.add_argument('--device', type=str, default="0", help="on all GPUs")
    parser.add_argument('--restore_version', type=int, default=-1, help="version for restore trainer and it's state")
    
    args = parser.parse_args()

    if args.dataset == "yelp":
        args.max_len = 18
        args.n_samples = 441651
    elif args.dataset == "gyafc":
        args.max_len = 28
        args.n_samples = 103934
    else:
        raise ValueError

    return args