from scipy.stats import linregress
from sklearn.externals import joblib
import json
import math
import random
import numpy as np
import pandas


## I/O / LOADING
def merge_datasets(dataset1, dataset2):
    x = []
    x.extend(dataset1)
    x.extend(dataset2)
    return x

def compile_binary_dataset(negative_samples, positive_samples):
    x = merge_datasets(negative_samples, positive_samples)
    y = np.concatenate([np.zeros(len(negative_samples)), np.ones(len(positive_samples))])
    return x, y

def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

def load_test_set(n_path, p_path):
    neg_yelp = load_dataset(n_path)
    pos_yelp = load_dataset(p_path)
    yelp_x_test, yelp_y_test = compile_binary_dataset(neg_yelp, pos_yelp)
    return yelp_x_test, yelp_y_test

def load_train_set(n_path, p_path, limits=200000):
    neg_yelp = load_dataset(n_path)
    pos_yelp = load_dataset(p_path)
    # shuffle
    random.shuffle(neg_yelp)
    random.shuffle(pos_yelp)
    # truncate
    neg_yelp = neg_yelp[:limits]
    pos_yelp = pos_yelp[:limits]
    yelp_x_train, yelp_y_train = compile_binary_dataset(neg_yelp, pos_yelp)
    return yelp_x_train, yelp_y_train

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_model(path):
    return joblib.load(path)

def load_turk_scores(aspect, model, param, param_val, npy_file=True):
    filetype = 'npy' if npy_file else 'npz'
    return np.load(f'../evaluations/human/{aspect}/{model}_{param}_{param_val}.{filetype}')

def save_json(data, path):
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f)

def save_model(model, path):
    joblib.dump(model, path)

def save_lines(data, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for line in data:
            f.write(line.strip() + "\n")


## CORRELATION TESTING
def calculate_std_err_of_r(r, n):
    # find standard error of correlation coefficient (based on jstor.org/stable/2277400)
    return (1-r**2)/math.sqrt(n)

def get_margin_of_error(std_err):
    # represent one standard deviation 
    # can be used with respect to mean of data to find confidence interval
    return 1.96 * std_err

def calculate_correlations(metrics_dict, turk_scores):
    correlation_dict = {}
    number_of_samples = len(turk_scores)
    
    for metric in metrics_dict:
        automated_scores = metrics_dict[metric]
        _, _, pearson_corr, pearson_p_val, _ = linregress(automated_scores, turk_scores)
        std_error_of_r = calculate_std_err_of_r(pearson_corr, number_of_samples) 
        sample_based_margin_of_error = get_margin_of_error(std_error_of_r)
        assert pearson_p_val < 0.05

        correlation_dict[metric] = {
            'r-val': pearson_corr,
            'error_bound': sample_based_margin_of_error
        }

    return pandas.DataFrame(data=correlation_dict).transpose()


## MISCELLANEOUS
def get_val_as_str(val):
    return str(val).replace('.', '_')

def invert_dict(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))