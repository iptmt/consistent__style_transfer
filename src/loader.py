import os
import torch

from data_util import *
from wmd import WMDdistance
from vocab import PAD_ID, BOS_ID, EOS_ID, BPETokenizer
from torch.utils.data import Dataset, RandomSampler, DataLoader


class StyleDataset(Dataset):
    def __init__(self, files, vocab, max_len, load_func):
        super().__init__()
        self.files = files
        self.vocab = vocab
        self.max_len = max_len
        self.load_func = load_func # callable
        self.samples = self.__load()
    
    def __load(self):
        samples = []
        for file in self.files:
            samples += self.load_func(file, self.truncate)
        return samples
    
    def truncate(self, sentence):
        return self.vocab.encode(sentence)[: self.max_len]
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)

def load_s2l(file_name, parse_func):
    assert os.path.exists(file_name)
    label = int(file_name.split(".")[-1])
    with open(file_name, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    sentences = list(filter(lambda s: s, sentences))
    return [(parse_func(s), label) for s in sentences]

#===============================================================================#
#                      functions of parsing samples
#===============================================================================#

def collate_pretrain(vocab, w2v):
    def collate_func(batch_samples):
        sentences, labels = zip(*batch_samples)

        noised_sentences_1 = transfer_noise(sentences, p=0.15)
        noised_sentences_2 = transfer_noise(sentences, p=0.15)
        noised_sentences_3 = rand_perm(sentences, p=0.15)

        aligned_sentences, _, _ = align(sentences, PAD_ID)
        aligned_noised_sentences_1, _, _ = align(noised_sentences_1, PAD_ID)
        aligned_noised_sentences_2, _, _ = align(noised_sentences_2, PAD_ID)

        aligned_noised_sentences_3, _, _ = align(noised_sentences_3, PAD_ID)

        c_label = w2v.cal_wmd_label(noised_sentences_1, noised_sentences_2, vocab)

        return (
            pth_tensor(aligned_sentences, torch.long),
            pth_tensor(aligned_noised_sentences_1, torch.long),
            pth_tensor(aligned_noised_sentences_2, torch.long),
            pth_tensor(aligned_noised_sentences_3, torch.long),
            pth_tensor(labels, torch.long),
            pth_tensor(c_label, torch.float)
        )
    return collate_func

def collate_warmup(batch_samples):
    sentences, labels = zip(*batch_samples)
    noised_sentences = transfer_noise(sentences, p=0.1)

    aligned_sentences, _, _ = align(sentences, PAD_ID)
    aligned_noised_sentences, _, _ = align(noised_sentences, PAD_ID)
    return (
        pth_tensor(aligned_noised_sentences, torch.long),
        pth_tensor(aligned_sentences, torch.long),
        pth_tensor(labels, torch.long)
    )

def collate_optimize(batch_samples):
    sentences, labels = zip(*batch_samples)
    aligned_sentences, _, _ = align(sentences, PAD_ID)
    return (
        pth_tensor(aligned_sentences, torch.long),
        pth_tensor(labels, torch.long)
    )