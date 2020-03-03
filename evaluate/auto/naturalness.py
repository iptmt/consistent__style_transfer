import sys 
sys.path.append("..")

from collections import Counter
# from keras.models import load_model as load_keras_model
# from keras.preprocessing.sequence import pad_sequences
from auto.tokenizer import RE_PATTERN
from auto.utils import get_val_as_str, invert_dict, load_dataset, load_model, load_turk_scores, merge_datasets
import numpy as np
import pandas as pd
import re

MAX_SEQ_LEN = 30 # for neural classifier 

## DATA PREP
def convert_to_indices(text, vocab):
    # tokenize input text
    tokens = re.compile(RE_PATTERN).split(text)    
    non_empty_tokens = list(filter(lambda token: token, tokens))
    
    indices = []
    
    # collect indices of tokens in vocabulary
    for token in non_empty_tokens:
        if token in vocab:
            index = vocab[token]
        else:
            index = vocab['CUSTOM_UNKNOWN']
            
        indices.append(index)
    
    return indices

# def format_inputs(texts, vocab):
#     # prepare texts for use in neural classifier
#     texts_as_indices = []
#     for text in texts:
#         texts_as_indices.append(convert_to_indices(text, vocab))
#     return pad_sequences(texts_as_indices, maxlen=MAX_SEQ_LEN, padding='post', truncating='post', value=0.)

class NaturalnessClassifier: 
    pass

class UnigramBasedClassifier(NaturalnessClassifier):
    def __init__(self, model_path, text_vectorizer):
        self.classifier = load_model(model_path)
        self.text_vectorizer = text_vectorizer
        
    def score(self, texts):
        vectorized_texts = self.text_vectorizer.transform(texts)
        distribution = self.classifier.predict_proba(vectorized_texts)
        scores = distribution[:,1] # column 1 represents probability of being 'natural'
        return scores

# class NeuralBasedClassifier(NaturalnessClassifier):
#     def __init__(self, model_path):
#         self.classifier = load_keras_model(model_path)

#     def score(self, texts, vocab):
#         inps = format_inputs(texts, vocab)
#         distribution = self.classifier.predict(inps)
#         scores = distribution.squeeze()
#         return scores

def generate_judgments(input_text_scores, output_text_scores):
    judgments = []
    
    for i in range(len(input_text_scores)):
        input_text_score = input_text_scores[i]
        output_text_score = output_text_scores[i]
        
        if input_text_score != output_text_score:
            # represent input text being scored as more natural as 1, otherwise 0
            val = int(input_text_score > output_text_score)
        else:
            val = None
        judgments.append(val)
        
    return judgments

def aggerate_judgments(judgments):
    relative_judgments = {"Success": 0, "Fail": 0, "Equal": 0}
    for judgment in judgments:
        if judgment is None:
            relative_judgments["Equal"] += 1
        elif judgment == 0:
            relative_judgments["Success"] += 1
        elif judgment == 1:
            relative_judgments["Fail"] += 1
        else:
            print(judgment)
            raise ValueError
    score = (relative_judgments["Success"] - relative_judgments["Fail"]) / len(judgments)
    return relative_judgments, score

# def format_relative_judgments(judgments):
#     judgments_map = {'A': 1, 'B': 0, None: None}
#     return list(map(lambda judgment: judgments_map[judgments], judgments))