import os
import pickle
from tokenizers import CharBPETokenizer

PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

class BPETokenizer:
    def __init__(self, text_list, vocab_size, lazy=False):
        if not lazy:
            self.tokenizer = CharBPETokenizer()
            self.tokenizer.train(text_list, vocab_size=vocab_size,
                                 min_frequency=2, special_tokens=[PAD, BOS, EOS, "<unk>"])
            self.tokenizer.add_special_tokens([PAD, BOS, EOS])
        else:
            self.tokenizer = None

    def tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(t) for t in tokens]    
    
    def ids_to_tokens(self, ids):
        return [self.tokenizer.id_to_token(i) for i in ids]

    def encode(self, text):
        encodes = self.tokenizer.encode(text)
        return encodes.ids
 
    def decode(self, ids, skip_special=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def save(self, path, file_name):
        self.tokenizer.save(path, file_name)

    @classmethod
    def load(cls, vocab, merges):
        tkz = cls(None, None, lazy=True)
        tkz.tokenizer = CharBPETokenizer(vocab, merges)
        tkz.tokenizer.add_special_tokens([PAD, BOS, EOS])
        return tkz
 
    def __len__(self):
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    # usage: python vocab.py [dataset] [vocab_size]
    import sys

    assert len(sys.argv) == 3

    dataset = sys.argv[1]
    vocab_size = int(sys.argv[2])

    files = [f"../data/{dataset}/style.train.0", f"../data/{dataset}/style.train.1"]

    if not os.path.exists(f"../dump/{dataset}/{dataset}-vocab.json") or not os.path.exists(f"../dump/{dataset}/{dataset}-merges.txt"):
        tkz = BPETokenizer(files, vocab_size)
        tkz.save(path=f"../dump/{dataset}", file_name=f"{dataset}")

    tkz = BPETokenizer.load(f"../dump/{dataset}/{dataset}-vocab.json", f"../dump/{dataset}/{dataset}-merges.txt")