import sys
import fasttext
from auto.utils import load_dataset, load_model

from auto.transfer_intensity import calculate_STIs
from auto.style_lexicon import load_lexicon
from auto.content_preserve import mask_style_words, load_word2vec_model, calculate_wmd_scores
from auto.naturalness import UnigramBasedClassifier, generate_judgments, aggerate_judgments


dataset = sys.argv[1]
model_name = sys.argv[2]

# base_dir = "/code/qwh/model_agnostic_ST"
# eval_dir = f"{base_dir}/evaluate"
base_dir = "../"
eval_dir = "./"

data_dir = f"{base_dir}/data/{dataset}/"
out_dir = f"{base_dir}/output/{dataset}-{model_name}/"

ds_ori_0, ds_ori_1 = load_dataset(data_dir + "style.test.0"), load_dataset(data_dir + "style.test.1")
origin = ds_ori_0 + ds_ori_1

ds_tsf_0, ds_tsf_1 = load_dataset(out_dir + "style.test.0.tsf"), load_dataset(out_dir + "style.test.1.tsf")
transfer = ds_tsf_0 + ds_tsf_1

mean = lambda seq: sum(seq)/len(seq)

# calculate STI
labels = [1] * len(ds_tsf_0) + [0] * len(ds_tsf_1)
sti_model = fasttext.load_model(f"{eval_dir}/eval_dump/model_{dataset}.bin")
SITs = calculate_STIs(origin, transfer, labels, sti_model)
print("STI (higher is better): %.4f" % mean(SITs))

# calculate CP
lexicon = load_lexicon(f"{eval_dir}/eval_dump/lexicon_{dataset}.json")
w2v = load_word2vec_model(f"{eval_dir}/eval_dump/mask_w2v_{dataset}.bin")
masked_origin = mask_style_words(origin, lexicon)
masked_transfer = mask_style_words(transfer, lexicon)
wmd_scores = calculate_wmd_scores(masked_origin, masked_transfer, w2v)
print("CP (lower is better): %.4f" % mean(wmd_scores))

# calculate NT
vectorizer = load_model(f"{eval_dir}/eval_dump/vectorizer_{dataset}.bin")
adv_classifier = UnigramBasedClassifier(f"{eval_dir}/eval_dump/adv_models/unigram_lr_{model_name}_{dataset}.bin", vectorizer)
input_scores = adv_classifier.score(origin)
output_scores = adv_classifier.score(transfer)
judgments = generate_judgments(input_scores, output_scores)
nt_score = aggerate_judgments(judgments)
print("NT (higher is better): %.4f" % nt_score)