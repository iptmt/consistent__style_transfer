import os
import sys
import time
import warnings
import fasttext
import numpy as np

from auto.style_lexicon import generate_lexicon
from auto.content_preserve import mask_style_words, train_word2vec_model
from auto.utils import load_dataset, merge_datasets, save_lines, save_model, load_model
from auto.nt_classifier import get_unigram_lr_model # unigram logistic regression

warnings.filterwarnings("ignore")

dataset = sys.argv[1]
if len(sys.argv) > 2:
    model_name = sys.argv[2]
else:
    model_name = None

# base_dir = "/code/qwh/model_agnostic_ST"
# eval_dir = f"{base_dir}/evaluate"
base_dir = "../"
eval_dir = "./"

data_dir = f"{base_dir}/data/{dataset}/"
out_dir = f"{base_dir}/output/{dataset}-{model_name}/"

start_time = time.time()


# CLASSIFIER
if not os.path.exists(f"{eval_dir}/eval_dump/model_{dataset}.bin"):
    print("\n<-TRAINING STYLE CLASSIFIER WITH FASTTEXT->")
    ## prepare data and dev files for training classifier (fasttext)
    w_file_tr = open(f"{eval_dir}/eval_tmp/{dataset}.train", "w+", encoding="utf-8")
    w_file_de = open(f"{eval_dir}/eval_tmp/{dataset}.dev", "w+", encoding="utf-8")
    for name in os.listdir(data_dir):
        if "train" in name:
            label = name.split(".")[-1]
            with open(data_dir + name, 'r', encoding="utf-8") as f:
                for line in f:
                    w_file_tr.write(f"__label__{label}\t{line.strip()}\n")
        if "dev" in name:
            label = name.split(".")[-1]
            with open(data_dir + name, 'r', encoding="utf-8") as f:
                for line in f:
                    w_file_de.write(f"__label__{label}\t{line.strip()}\n")
    w_file_tr.close()
    w_file_de.close()

    ## train classifier by fasttext with default parameters
    model = fasttext.train_supervised(f"{eval_dir}/eval_tmp/{dataset}.train")
    ## evaluation
    N, p, r = model.test(f"{eval_dir}/eval_tmp/{dataset}.dev")
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    ## saving
    model.save_model(f"{eval_dir}/eval_dump/model_{dataset}.bin")


# LEXICON & WORD2VEC
if not os.path.exists(f"{eval_dir}/eval_dump/lexicon_{dataset}.json") or \
   not os.path.exists(f"{eval_dir}/eval_dump/vectorizer_{dataset}.bin") or \
   not os.path.exists(f"{eval_dir}/eval_dump/mask_w2v_{dataset}.bin"):
    print("\n<-GENERATING LEXICON & TRAINING WORD2VEC->")
    ## generate lexicon
    lexicon, vectorizer = generate_lexicon(data_dir + "style.train.0", data_dir + "style.train.1", 
                                           f"{eval_dir}/eval_dump/lexicon_{dataset}.json", 
                                           f"{eval_dir}/eval_dump/vectorizer_{dataset}.bin")
    ## train word vectors
    texts = []
    for name in os.listdir(data_dir):
        if "train" in name or "dev" in name:
            texts += load_dataset(data_dir + name)
    all_texts_style_masked = mask_style_words(texts, lexicon)
    print("Training masked version word2vec model...")
    train_word2vec_model(all_texts_style_masked, f"{eval_dir}/eval_dump/mask_w2v_{dataset}.bin")


# NATURALNESS
if model_name and not os.path.exists(f"{eval_dir}/eval_dump/adv_models/unigram_lr_{model_name}_{dataset}.bin"):
    print("\n<-TRAINING ADVESARIAL CLASSIFIER->")
    ## aggregate true/fake sentences
    tsf_sents, ori_sents = [], []
    for name in os.listdir(out_dir):
        if "train" in name:
            tsf_sents += load_dataset(out_dir + name)
    save_lines(tsf_sents, f"{eval_dir}/eval_tmp/{dataset}-{model_name}.train.tsf")
    for name in os.listdir(data_dir):
        if "train" in name:
            ori_sents += load_dataset(data_dir+ name)
    save_lines(ori_sents, f"{eval_dir}/eval_tmp/{dataset}-{model_name}.train.ori")

    vectorizer = load_model(f"{eval_dir}/eval_dump/vectorizer_{dataset}.bin")
    lr_model = get_unigram_lr_model(f"{eval_dir}/eval_tmp/{dataset}-{model_name}.train.tsf", f"{eval_dir}/eval_tmp/{dataset}-{model_name}.train.ori", vectorizer)
    save_model(lr_model, f"{eval_dir}/eval_dump/adv_models/unigram_lr_{model_name}_{dataset}.bin")

time_cost = time.time() - start_time
print("\nTime cost: %.2fs" % time_cost)
print("\nDone!")