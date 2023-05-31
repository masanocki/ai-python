import numpy as np
import pandas as pd
from nbc import DiscreteNaiveBayes
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def read_spambase_data(filepath):
    D = pd.read_csv(filepath, header=None, sep=',').to_numpy()
    y = D[:, -1].astype('int8') # crack consumption
    X = D[:, :-1]
    return X, y

def train_test_split(X, y, train_ratio=0.75, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    indexes = np.random.permutation(m)
    X = X[indexes]
    y = y[indexes]
    index = int(np.round(train_ratio * m))
    X_train = X[:index]
    y_train = y[:index]
    X_test = X[index:]
    y_test = y[index:]
    return X_train, y_train, X_test, y_test

def discretize_spambase_data(X, bins=5, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip(((X - mins_ref) / (maxes_ref - mins_ref) * bins).astype(np.int8), 0, bins-1)
    return X_d, mins_ref, maxes_ref

def niebezpieczna_sytuacja_log(X, y):
    X = np.tile(X, 30)
    bins = 60
    n = X.shape[1]
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75, seed=0)   
    X_d_train, mins_ref, maxes_ref = discretize_spambase_data(X_train, bins=bins)
    X_d_test, _, _ = discretize_spambase_data(X_test, bins=bins, mins_ref=mins_ref, maxes_ref=maxes_ref)
    
    domain_sizes = bins * np.ones(n, dtype=np.int8)
    
    clfs = [DiscreteNaiveBayes(domain_sizes, laplace=True, logs=True),
            CategoricalNB(min_categories=domain_sizes), GaussianNB(),
            KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(),
            MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64, 32), random_state=1, solver='adam')
    ]
    
    for clf in clfs:
        print('---')
        print(f'{clf.__class__.__name__}', end=', ')
        clf.fit(X_d_train, y_train)
        print(f'ACC: {clf.score(X_d_test, y_test)}')

def niebezpieczna_sytuacja_bez_log(X, y):
    X = np.tile(X, 30)
    bins = 60
    n = X.shape[1]
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75, seed=0)   
    X_d_train, mins_ref, maxes_ref = discretize_spambase_data(X_train, bins=bins)
    X_d_test, _, _ = discretize_spambase_data(X_test, bins=bins, mins_ref=mins_ref, maxes_ref=maxes_ref)
    
    domain_sizes = bins * np.ones(n, dtype=np.int8)
    
    clfs = [DiscreteNaiveBayes(domain_sizes, laplace=True, logs=False),
            CategoricalNB(min_categories=domain_sizes), GaussianNB(),
            KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(),
            MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64, 32), random_state=1, solver='adam')
    ]
    
    for clf in clfs:
        print('---')
        print(f'{clf.__class__.__name__}', end=', ')
        clf.fit(X_d_train, y_train)
        print(f'ACC: {clf.score(X_d_test, y_test)}')

def bezpieczna_sytuacja(X, y):
    bins = 60
    n = X.shape[1]
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75, seed=0)   
    X_d_train, mins_ref, maxes_ref = discretize_spambase_data(X_train, bins=bins)
    X_d_test, _, _ = discretize_spambase_data(X_test, bins=bins, mins_ref=mins_ref, maxes_ref=maxes_ref)
       
    domain_sizes = bins * np.ones(n, dtype=np.int8)
    
    clfs = [DiscreteNaiveBayes(domain_sizes, laplace=True, logs=False),
            CategoricalNB(min_categories=domain_sizes), GaussianNB(),
            KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(),
            MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64, 32), random_state=1, solver='adam')
    ]
    
    for clf in clfs:
        print('---')
        print(f'{clf.__class__.__name__}', end=', ')
        clf.fit(X_d_train, y_train)
        print(f'ACC: {clf.score(X_d_test, y_test)}')

if __name__ == '__main__':
    X, y = read_spambase_data('spambase.data')

    # bezpieczna_sytuacja(X, y)
    # niebezpieczna_sytuacja_bez_log(X, y)
    niebezpieczna_sytuacja_log(X, y)