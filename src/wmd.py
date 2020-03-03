import os
import random
import gensim
from gensim.models.word2vec import Word2Vec

import logging
# this logger is really nosing me
logging.getLogger("gensim").setLevel(logging.WARNING)


class WMDdistance:
    def __init__(self, file_lists, tokenizer, lazy=False):
        if not lazy:
            corpus = []
            for file in file_lists:
                corpus += self._load_file(file)
            random.shuffle(corpus)
            sentences = [self.tokenize(tokenizer, s) for s in corpus]
            self.model = Word2Vec(sentences, iter=10)
        else:
            self.model = None

    def tokenize(self, tokenizer, text):
        return tokenizer.ids_to_tokens(tokenizer.encode(text))
    
    def _load_file(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def cal_wmd(self, x1, x2):
        return self.model.wv.wmdistance(x1, x2)
    
    def cal_wmd_label(self, xs1, xs2, tokenizer):
        label = []
        for x1, x2 in zip(xs1, xs2):
            label.append(self.cal_wmd(tokenizer.ids_to_tokens(x1), tokenizer.ids_to_tokens(x2)))
        return label

    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        wmd = cls(None, None, lazy=True)
        wmd.model = Word2Vec.load(path)
        wmd.model.wv.init_sims(replace=True)
        return wmd

    
if __name__ == "__main__":
    # usage: python wmd.py [dataset]
    import sys
    from vocab import BPETokenizer

    assert len(sys.argv) == 2
    dataset = sys.argv[1]

    files = [f"../data/{dataset}/style.train.0", f"../data/{dataset}/style.train.1"]
    tkz = BPETokenizer.load(f"../dump/{dataset}/{dataset}-vocab.json", 
                            f"../dump/{dataset}/{dataset}-merges.txt")
    
    dump_path = f"../dump/{dataset}/{dataset}-w2v.bin"
    if not os.path.exists(dump_path):
        wmd = WMDdistance(files, tkz)
        wmd.save(dump_path)
    
    wmd = WMDdistance.load(dump_path)