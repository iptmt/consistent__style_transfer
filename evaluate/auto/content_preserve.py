import sys 
sys.path.append("..")

from gensim.models.word2vec import Word2Vec
from auto.style_lexicon import load_lexicon
from auto.tokenizer import tokenize
from auto.utils import calculate_correlations, get_val_as_str, load_dataset, load_turk_scores, merge_datasets
import numpy as np


CUSTOM_STYLE = 'MASK'

def mask_style_words(texts, lexicon):
    edited_texts = []
    
    for text in texts:
        tokens = tokenize(text)
        edited_tokens = []
        
        for token in tokens:
            if token.lower() in lexicon:
                edited_tokens.append(CUSTOM_STYLE)
            else:
                edited_tokens.append(token)
            
        edited_texts.append(' '.join(edited_tokens))

    return edited_texts

## MODELS / SCORING OF WMD
def train_word2vec_model(texts, path):
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(tokenize(text))
    model = Word2Vec(tokenized_texts)
    model.save(path)

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    model.wv.init_sims(replace=True) # normalize vectors
    return model

def calculate_wmd_scores(references, candidates, wmd_model):
    wmd_scores = []

    for i in range(len(references)):
        wmd = wmd_model.wv.wmdistance(tokenize(references[i]), tokenize(candidates[i]))
        wmd_scores.append(wmd)

    return wmd_scores

if __name__ == "__main__":
    # # load data to train models used for WMD calculations
    lexicon = load_lexicon("../eval_dump/lexicon_yelp.json")
    w2v = load_word2vec_model("../eval_dump/mask_w2v_yelp.bin")

    # # load texts under different style modification settings
    input_neg_texts = load_dataset('../../data/yelp/style.test.0')
    input_pos_texts = load_dataset('../../data/yelp/style.test.1')
    input_texts = merge_datasets(input_neg_texts, input_pos_texts)
    inputs_with_style_masked = mask_style_words(input_texts, lexicon)

    output_neg_texts = load_dataset('../../output/style.test.0.tsf')
    output_pos_texts = load_dataset('../../output/style.test.1.tsf')
    output_texts = merge_datasets(output_neg_texts, output_pos_texts)
    output_with_style_masked = mask_style_words(output_texts, lexicon)

    scores = calculate_wmd_scores(inputs_with_style_masked, output_with_style_masked, w2v)
    print(sum(scores) / len(scores))