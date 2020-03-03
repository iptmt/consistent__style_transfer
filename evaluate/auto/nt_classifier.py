import sys 
sys.path.append("..")

from sklearn.linear_model import LogisticRegression
from auto.utils import load_train_set


def train_unigram_lr(regularization_type, C, n_jobs, vec_x_train, y_train):
    lr = LogisticRegression(penalty=regularization_type, C=C, n_jobs=n_jobs)
    lr.fit(vec_x_train, y_train)
    return lr

def train_neural_lr():
    pass


def get_unigram_lr_model(n_path, p_path, vectorizer):
    x_tr, y_tr = load_train_set(n_path, p_path)
    regularization_type = 'l1'
    C = 3
    n_jobs = 4
    vec_x_tr = vectorizer.transform(x_tr)
    print("Training logistic regression model...")
    lr_model = train_unigram_lr(regularization_type, C, n_jobs, vec_x_tr, y_tr)
    return lr_model