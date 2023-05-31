from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SimplePerceptron(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=1.0, ws=False):
        self.learning_rate_ = learning_rate
        self.w_ = None
        self.k_ = 0 # licznik krokow
        self.class_labels_ = None
        self.ws_ = ws


    def fit(self, X, y):
        self.class_labels_ = np.unique(y) # zakladamy obecnosc dokladnie 2 klas i np. self.class_labels_[0] --> -1
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.class_labels_[0]] = -1
        XX = np.c_[np.ones(m), X]
        self.w_ = np.zeros(n + 1)

        while True:
            E = [] # lista indeksow zle sklasyfikowanych punktow danych
            for i in range(m):
                s = self.w_.dot(XX[i])
                f = 1 if s > 0.0 else -1
                if f != yy[i]:
                    E.append(i)
                    break
            if len(E) == 0 or self.k_ == 4000:
                return
            i = np.random.choice(E)
            self.w_ = self.w_ + self.learning_rate_ * yy[i] * XX[i]
            self.k_ += 1

    def predict(self, X):
        sums = self.decision_fun(X)
        m = X.shape[0]
        responses = np.zeros(m, dtype=np.int8)
        responses[sums > 0.0] = self.class_labels_[1]
        responses[sums <= 0.0] = self.class_labels_[0]
        if self.ws_:
            return sums
        else:
            return responses

    def decision_fun(self, X):
        m = X.shape[0]
        XX = np.c_[np.ones(m), X]
        return self.w_.dot(XX.T)

