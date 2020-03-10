import numpy as np
from vocab import BPETokenizer
from wmd import WMDdistance
from data_util import transfer_noise


base_dir = "../data/gyafc/"
lengths = []
words = set()
length = dict()

for d in ("style.train.0",):
    with open(base_dir + d, 'r', encoding='utf-8') as f:
        for l in f:
            ws = l.split()
            lens = len(ws)
            if lens in length:
                length[lens] += 1
            else:
                length[lens] = 1
            lengths.append(lens)
            for w in ws:
                if w not in words:
                    words.add(w)

print(length)
print("-" * 100)
print(len(words))
print("-" * 100)
print(sum(lengths) / len(lengths))