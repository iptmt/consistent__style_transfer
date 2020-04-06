data_dir = "../../data"
for ds in ("yelp", "book"):
    vocab, sent_lens = set(), list()
    for sty in ("train", "dev", "test"):
        for s in ("0", "1"):
            cnt = 0
            with open(f"{data_dir}/{ds}/style.{sty}.{s}", 'r', encoding='utf-8') as f:
                for line in f:
                    cnt += 1
                    tokens = line.strip().split()
                    sent_lens.append(len(tokens))
                    for t in tokens:
                        vocab.add(t)
            print(f"{ds}, {sty}, {s}: {cnt}")
    print(f"{ds}, vocabulary size: {len(vocab)}")
    print(f"{ds}, average length: {sum(sent_lens)/len(sent_lens)}")
    print("=" * 100)