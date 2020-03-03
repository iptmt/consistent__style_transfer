import os
import copy
import torch
import random

import numpy as np

from torch.nn.functional import one_hot, pad


def path_cat(*parts):
    return os.path.join(*parts)

def pth_tensor(tensor, dtype):
    return torch.tensor(tensor, dtype=dtype)

def add_borders(tokens_list, start=None, end=None):
    if start is not None and end is not None: 
        return [[start] + tokens + [end] for tokens in tokens_list]
    elif start is None and end is not None:
        return [tokens + [end] for tokens in tokens_list]
    elif start is not None and end is None:
        return [[start] + tokens for tokens in tokens_list]

def align(sentences, pad_value, max_len=None):
    if max_len is None:
        max_len = max([len(sent) for sent in sentences])
    lengths = [len(sent[:max_len]) for sent in sentences]
    sentences = [sent[:max_len] + [pad_value] * (max_len - len(sent)) for sent in sentences]
    return sentences, lengths, max_len

# def convert_to_str(tensor, vocab):
#     ids = tensor.cpu().numpy().tolist()
#     tokens = vocab.IdsToSent(ids, remove_special=False)
#     return tokens

# # cal BLEU
# def cal_bleu(seq1, seq2):
#     return nltk.translate.bleu_score.sentence_bleu([seq1], [seq2])

def transfer_noise(sentences, p):
    word_bag, sentences_noise, lens = [], [], []
    for s in sentences:
        s_noise = []
        ind = (np.random.uniform(size=(len(s))) < p)
        lens.append(len(s))
        for idx, v in enumerate(ind):
            if v:
                word_bag.append(s[idx])
            else:
                s_noise.append(s[idx])
        sentences_noise.append(s_noise)
    lens = np.array(lens, dtype=np.float)
    p = lens / lens.sum()
    indexes = list(range(len(p)))
    choices = np.random.choice(indexes, size=(len(word_bag),), p=p)
    for idx, w in enumerate(word_bag):
        # select template index
        index = choices[idx]
        # select position
        pos = random.randint(0, len(sentences_noise[index]))
        sentences_noise[index].insert(pos, w)
    return sentences_noise