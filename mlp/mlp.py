import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MultiLayerPerceptron(BaseEstimator, RegressorMixin):

    def __init__(self, K=16, T=10**4, learning_rate=0.01, batch_size=1, alg='backpropagation', seed=0):
        self.K_ = K # liczba neuronow
        self.T_ = T # liczba krokow uczenia
        self.learning_rate_ = learning_rate
        self.seed_ = seed
        self.V_ = None # macierz z wagami w warstwie ukrytej: K x (n + 1)
        self.W_ = None # wektor (kolumnowy) z wagami w warstwie wyjsciowej: (K + 1) x 1
        self.batch_size_ = batch_size
        self.alg_ = alg

    def sigmoid(self, s):
        return 1.0 / (1.0 + np.exp(-s))

    def sigmoid_d(self, phi):
        return phi * (1.0 - phi)

    def relu(self, s):
        s[s < 0] = 0.0
        return s

    def relu_d(self, s):
        return (s > 0.0) * 1.0

    def activation(self, s):
        # return self.sigmoid(s)
        return self.relu(s)

    def activation_d(self, s, phi):
        # return self.sigmoid_d(phi)
        return self.relu_d(s)

    def fit(self, X, y):
        np.random.seed(self.seed_)
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        self.V_ = (np.random.rand(self.K_, n + 1) * 2.0 - 1.0) * 1e-2
        self.W_ = (np.random.rand(self.K_ + 1, 1) * 2.0 - 1.0) * 1e-2
        for _ in range(self.T_):
            dW = np.zeros(self.W_.shape)
            dV = np.zeros(self.V_.shape)
            indexes = np.random.choice(m, self.batch_size_)
            X_batch = X[indexes]
            s = self.V_.dot(X_batch.T)
            phi = self.activation(s)
            one_phi = np.r_[np.ones((1, self.batch_size_)), phi]
            y_MLP = self.W_.T.dot(one_phi)
            loss = y_MLP[0] - y[indexes]
            dW = np.sum(loss * one_phi, axis=1)
            dW = np.array([dW]).T
            dV = (loss * self.W_[1:] * self.activation_d(s, phi)).dot(X_batch)
            if self.alg_ == 'backpropagation':
                self.W_ = self.W_ - self.learning_rate_ * dW
                self.V_ = self.V_ - self.learning_rate_ * dV
            elif self.alg_ == 'momentum':
                momentum = 0.9 # wspolczynnik rozpedu
                # wektory do informacji o poprzednich aktualizacjach wag
                mW = np.zeros(self.W_.shape)
                mV = np.zeros(self.V_.shape)
                mW = momentum * mW + (1 - momentum) * dW
                self.W_ = self.W_ - self.learning_rate_ * mW
                mV = momentum * mV + (1 - momentum) * dV
                self.V_ = self.V_ - self.learning_rate_ * mV         
        if self.alg_ == 'rprop':
            eta_0 = 0.05
            eta_min = 10**-3.0
            eta_max = 50.0
            eta_increase = 1.2
            eta_decrease = 0.5
            d_W = np.zeros((self.W_.shape[0], self.W_.shape[1])) + eta_0
            d_V = np.zeros((self.V_.shape[0], self.V_.shape[1])) + eta_0
            xxx, yyy = self.W_.shape
            temp_dW = 0.0
            for i in range(xxx):
                for j in range(yyy):
                    if temp_dW * dW[i][j] > 0.0:
                        d_W[i][j] = np.minimum(d_W[i][j] * eta_increase, eta_max)
                        self.W_[i][j] = self.W_[i][j] - d_W[i][j] * np.sign(dW[i][j])
                    elif temp_dW * dW[i][j] < 0.0:
                        d_W[i][j] = np.maximum(d_W[i][j] * eta_decrease, eta_min)
                        self.W_[i][j] = self.W_[i][j] - d_W[i][j] * np.sign(dW[i][j])
                    else:
                        self.W_[i][j] = self.W_[i][j] - d_W[i][j] * np.sign(dW[i][j])
                    temp_dW = dW[i][j]
            xxx, yyy = self.V_.shape
            temp_dV = 0.0
            for i in range(xxx):
                for j in range(yyy):
                    if temp_dV * dV[i][j] > 0.0:
                        d_V[i][j] = np.minimum(d_V[i][j] * eta_increase, eta_max)
                        self.V_[i][j] = self.V_[i][j] - d_V[i][j] * np.sign(dV[i][j])
                    elif temp_dV * dV[i][j] < 0.0:
                        d_V[i][j] = np.maximum(d_V[i][j] * eta_decrease, eta_min)
                        self.V_[i][j] = self.V_[i][j] - d_V[i][j] * np.sign(dV[i][j])
                    else:
                        self.V_[i][j] = self.V_[i][j] - d_V[i][j] * np.sign(dV[i][j])
                    temp_dV = dV[i][j]
                    
                
    def predict(self, X):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        s = self.V_.dot(X.T)  # K x m
        phi = self.activation(s)  # K x m
        one_phi = np.r_[np.ones((1, m)), phi]  # (K + 1) x m
        y_MLP = self.W_.T.dot(one_phi)  # (1 X (K + 1)).dot((K + 1) x m)
        return y_MLP[0]
