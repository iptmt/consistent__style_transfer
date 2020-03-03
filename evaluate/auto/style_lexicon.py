import sys 
sys.path.append("..")

from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from auto.tokenizer import tokenize
from auto.utils import invert_dict, load_json, load_train_set, save_json, save_model
import numpy as np
import warnings


def fit_vectorizer(inp):
    vectorizer = CountVectorizer(binary=True, tokenizer=tokenize)
    vectorizer.fit(inp)
    return vectorizer

def train(regularization_type, C, n_jobs, vec_x_train, y_train):
    lr = LogisticRegression(penalty=regularization_type, C=C, n_jobs=n_jobs)
    lr.fit(vec_x_train, y_train)
    return lr

def extract_nonzero_weights(model):
    all_weights = model.coef_
    
    feature_numbers_for_nonzero_weights = []
    nonzero_weights = []
    
    for style_number, weights in enumerate(all_weights):
        feature_numbers = np.where(abs(weights) > 0.0)[0]
        feature_numbers_for_nonzero_weights.append(feature_numbers)
        nonzero_weights.append(weights[feature_numbers])
    
    return np.array(nonzero_weights), np.array(feature_numbers_for_nonzero_weights)

def select_feature_numbers(weights, number_of_standard_deviations=2):
    standard_deviation = np.std(weights)
    mean = np.mean(weights)
    left_bound = mean - number_of_standard_deviations * standard_deviation
    right_bound = mean + number_of_standard_deviations * standard_deviation
    feature_numbers = np.where((weights < left_bound) | (weights > right_bound))[0]
    return feature_numbers

def extract_ranked_features(nonzero_weights, feature_numbers, inverse_vocabulary, weighted_feature_numbers):
    dictionary = {} 

    for index, feature_number in enumerate(weighted_feature_numbers):
        feature = inverse_vocabulary[feature_number]
        weight = nonzero_weights[feature_numbers[index]]
        dictionary[feature] = weight

    return sorted(dictionary.items(), key=itemgetter(1))

def collect_style_features_and_weights(weights, styles, inverse_vocabulary, feature_numbers, number_of_standard_deviations=2):
    style_features_and_weights = {}
    
    for style_number, style_weights in enumerate(weights):
        style = styles[style_number]
        selected_feature_numbers = select_feature_numbers(style_weights, number_of_standard_deviations) 
        weighted_feature_numbers = feature_numbers[style_number][selected_feature_numbers]        
        ranked_features = extract_ranked_features(style_weights, selected_feature_numbers, inverse_vocabulary, weighted_feature_numbers)    
        style_features_and_weights[style] = ranked_features
        
    return style_features_and_weights


def generate_lexicon(negative_path, positive_path, lexicon_path, vectorizer_path):
    warnings.filterwarnings("ignore")
    ## STEP 1.
    ## LOAD AND PREPARE DATA
    print("Loading training dataset...")
    x_tr, y_tr = load_train_set(negative_path, positive_path) 

    print("Fitting vectorizer...")
    vectorizer = fit_vectorizer(x_tr)
    save_model(vectorizer, vectorizer_path)
    inverse_vocabulary = invert_dict(vectorizer.vocabulary_)

    ## STEP 2.
    regularization_type = 'l1'
    C = 3
    n_jobs = 4
    vec_x_tr = vectorizer.transform(x_tr)
    print("Training logistic regression model...")
    lr_model = train(regularization_type, C, n_jobs, vec_x_tr, y_tr)

    ## STEP 3.
    ## EXTRACT STYLE FEATURES AND WEIGHTS 
    print("Generating lexicon...")
    styles = {0: "binary sentiment"}
    style_features_and_weights_path = lexicon_path
    nonzero_weights, feature_numbers = extract_nonzero_weights(lr_model) 
    style_features_and_weights = collect_style_features_and_weights(nonzero_weights, styles, inverse_vocabulary, feature_numbers)
    save_json(style_features_and_weights, style_features_and_weights_path)

    return set(map(lambda x: x[0], style_features_and_weights["binary sentiment"])), vectorizer


def load_lexicon(lexicon_path):
    # collect style words from existing set of style features and weights
    style_features_and_weights = load_json(lexicon_path)
    return set(map(lambda x: x[0], style_features_and_weights["binary sentiment"]))