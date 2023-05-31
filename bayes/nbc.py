from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DiscreteNaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, domain_sizes, laplace=False, logs=False):
        self.class_labels_ = None
        self.PY_ = None  # a priori probabilities of classes (1D vector)
        self.P_ = None  # 3D array with all conditional probabilities P(X_j = v | Y = y)
        self.domain_sizes_ = domain_sizes
        self.laplace_ = laplace
        self.logs_ = logs

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        K = self.class_labels_.size # no. of classes
        yy = np.zeros(m, dtype=np.int8)
        for i, label in enumerate(self.class_labels_):
            yy[y == label] = i # mapping labels to: 0, 1, 2, ...

        self.PY_ = np.zeros(K)
        for k in range(K):
            self.PY_[k] = np.sum(yy == k) / m # np.mean(yy == k)

        self.P_ = np.zeros((K, n), dtype=np.object)
        for k in range(K):
            for j in range(n):
                self.P_[k, j] = np.zeros(self.domain_sizes_[j])

        for i in range(m):
            x = X[i]
            for j in range(n):
                self.P_[yy[i], j][x[j]] += 1

        # w przypadku bezpiecznym: przechowywanie od razu logarytmów prawdopodobienstw
        # w przypadku niebezpiecznym: przechowywanie prawdopodobieństw
        for k in range(K):
            if self.laplace_:
                for j in range(n):
                    if self.logs_:
                        self.P_[k, j] = np.log((self.P_[k, j] + 1) / (np.sum(self.P_[k, j]) + self.domain_sizes_[j]))
                    else:
                        self.P_[k, j] = (self.P_[k, j] + 1) / (np.sum(self.P_[k, j]) + self.domain_sizes_[j])
            else:
                for j in range(n):
                    if self.logs_:
                        self.P_[k, j] = np.log(self.P_[k, j] / np.sum(self.P_[k, j]))
                    else:
                        self.P_[k, j] /= np.sum(self.P_[k, j])



    def predict(self, X):
        return self.class_labels_[np.argmax(self.predict_proba(X), axis=1)]

    # w przypadku bezpiecznym: sumowanie
    # w przypadku niebezpiecznym: iloczyn
    def predict_proba(self, X):
        # Formula: argmax_y \prod_{j=1}^n P(X_j = x_j | Y = y) * P(Y = y)
        m, n = X.shape
        K = self.class_labels_.size
        probas = np.zeros((m, K))
        for i in range(m):
            for k in range(K):
                probas[i, k] = self.PY_[k]
                for j in range(n):
                    if self.logs_:
                        probas[i, k] += self.P_[k, j][X[i, j]]
                    else:
                        probas[i, k] *= self.P_[k, j][X[i, j]]
        return probas

