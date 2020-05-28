import sys 
sys.path.append("../..")
sys.path.append("../../evaluate")

import csv
import random
from collections import defaultdict
from evaluate.auto.style_lexicon import load_lexicon
from evaluate.auto.content_preserve import mask_style_words


topics = ["yelp", "book"]
models = ["drg", "unmt", "full"]

files = ["style.test.0", "style.test.1"]

def read(file):
    with open(file, "r") as f:
        return [line.strip() for line in f]

pairs = {}

for t in topics:
    for m in models:
        # 定义文件路径
        tsf_dir = f"../../output/{t}-{m}"
        if t == "book" and m == "drg":
            ori_dir = tsf_dir
        else:
            ori_dir = f"../../data/{t}"
        # 读取文件，形成句子对，收集数据
        for fn in files:
            ori = read(f"{ori_dir}/{fn}")
            tsf = read(f"{tsf_dir}/{fn}.tsf")
            pairs[m + "-" + t + "-" + fn.split(".")[-1]] = dict(list(zip(ori, tsf)))

# 确认索引
y0 =random.sample(list(pairs["drg-yelp-0"].keys()), 50)
y1 =random.sample(list(pairs["drg-yelp-1"].keys()), 50)
b0 =random.sample(list(pairs["drg-book-0"].keys()), 50)
b1 =random.sample(list(pairs["drg-book-1"].keys()), 50)

# 对各个系统，导出相应的结果
yelp_indexes = list(y0) + list(y1)
book_indexes = list(b0) + list(b1)
yelp_lex = load_lexicon("../../evaluate/eval_dump/lexicon_yelp.json")
book_lex = load_lexicon("../../evaluate/eval_dump/lexicon_book.json")
yelp_mask = mask_style_words(yelp_indexes, yelp_lex)
book_mask = mask_style_words(book_indexes, book_lex)
yelp_mask_pairs = dict(list(zip(yelp_indexes, yelp_mask)))
book_mask_pairs = dict(list(zip(book_indexes, book_mask)))
yelp_pairs, book_pairs = defaultdict(list),defaultdict(list) 
for m in models:
    dic_m = dict()
    for k in pairs:
        if m in k:
            dic_m.update(pairs[k])
    yelp_res = [dic_m[idx] for idx in yelp_indexes]
    yelp_res_mask = mask_style_words(yelp_res, yelp_lex)
    yelp_paris_m = list(zip(yelp_indexes, list(zip(yelp_res, yelp_res_mask))))
    for yelp_ori, yelp_out in yelp_paris_m:
        yelp_pairs[yelp_ori].append(yelp_out)

    book_res = [dic_m[idx] for idx in book_indexes]
    book_res_mask = mask_style_words(book_res, book_lex)
    book_pairs_m = list(zip(book_indexes, list(zip(book_res, book_res_mask))))
    for book_ori, book_out in book_pairs_m:
        book_pairs[book_ori].append(book_out)

# 写入csv文件
with open('yelp.csv', 'w', newline='') as csvfile:
    fields = ["id", "origin", "transfer", "STI", "CP", "NT"]
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for idx, ori in enumerate(yelp_pairs):
        system_pairs = yelp_pairs[ori]
        ori_mask = yelp_mask_pairs[ori]
        for idx_, (s, sm) in enumerate(system_pairs):
            content = {
                "id": idx + 1, 
                "origin": ori + "\n" + ori_mask if idx_ == 0 else "",
                "transfer": s + "\n" + sm,
                "STI": "", "CP": "", "NT": "" 
            }


# with open('amazon.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)