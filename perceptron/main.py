import numpy as np
from matplotlib import pyplot as plt
from perceptron import SimplePerceptron
np.random.seed(1)

def generate_data(m):
    X_1 = np.random.uniform(low=-1, high=1, size=(m, 1))
    X_2 = np.random.uniform(low=0, high=2*np.pi, size=(m, 1))
    X = np.c_[X_2, X_1]
    y = np.array([-1 if np.abs(np.sin(val[0])) > np.abs(val[1]) else 1 for val in X])
    return X, y

def normalize_data(X):
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    for j in range(X.shape[0]):
        X[j, 0] = -1 + (X[j, 0] - x_min) * (1 - (-1)) / (x_max - x_min)
    return X

def get_centres(X, m=90):
    return X[np.random.randint(0, X.shape[0], m)]

def upper_dimension(X, c):
    sigma = 0.2
    z = np.zeros((X.shape[0], c.shape[0]))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i, j] = np.exp(-((X[i, 0] - c[j, 0])**2 + (X[i, 1] - c[j, 1])**2)/(2*sigma**2))
    return z
    

if __name__ == '__main__':
    X, y = generate_data(m=2000)
    X = normalize_data(X)
    clf = SimplePerceptron(learning_rate=1.0)
    c = get_centres(X)
    clf.fit(upper_dimension(X, c), y)
    
    # meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(upper_dimension(np.c_[xx.ravel(), yy.ravel()], c))
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=1, cmap=plt.cm.get_cmap('Pastel1'))
    plt.contour(xx, yy, Z, levels=0, colors = ['black'], linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=5)
    plt.scatter(c[:, 0], c[:, 1], c='black')   
    
    clf = SimplePerceptron(learning_rate=1.0, ws=True)
    clf.fit(upper_dimension(X, c), y)
    Z = clf.predict(upper_dimension(np.c_[xx.ravel(), yy.ravel()], c))
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, Z, cmap='coolwarm')
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, levels=10, cmap='coolwarm')
    plt.contour(xx, yy, Z, levels=10, colors = ['black'], linewidths=0.7)
    plt.show()
    
