import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mlp import MultiLayerPerceptron
import time

# plt.get_backend()
# Option 1
# QT backend
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# # Option 2
# # TkAgg backend
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

# # Option 3
# # WX backend
# manager = plt.get_current_fig_manager()
# manager.frame.Maximize(True)

# alg:
# backpropagation
# momentum - (uczenie z rozpedem (momentum + EMA))
# rprop
# adam

def fake_data(m):
    np.random.seed(0)
    X = np.random.rand(m, 2) * np.pi
    y = np.cos(X[:, 0] * X[:, 1]) * np.cos(2 * X[:, 0]) + np.random.randn(m) * 0.1
    return X, y

def test_all(X, y):
    algs = ['backpropagation', 'momentum', 'rprop']
    Ts = [10**4, 10**5, 10**6, 2*10**4, 2*10**5, 2*10**6]
    batch_sizes = [100, 10, 1, 100, 10, 1]
    
    with open('results.txt', 'w') as file:
        for algo in algs:
            countX = 0
            countY = 0
            fig, axs = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': '3d'})
            fig.set_size_inches(32, 18)
            print(f'alg: {algo}')
            file.write(f'alg: {algo}\n')
            for i in range(6):
                nn = MultiLayerPerceptron(K=64, T=Ts[i], learning_rate=0.01, batch_size=batch_sizes[i], alg=algo, seed=1)
                t1 = time.time()
                nn.fit(X, y)
                t2 = time.time()
                predictions = nn.predict(X)
                mse = np.mean(0.5 * (y - predictions)**2)
                score = nn.score(X, y)
                print(f'T: {Ts[i]}, batch_size: {batch_sizes[i]} --> FIT DONE: [{t2-t1} s], MSE: {mse}, ACC: {score}')
                file.write(f'T: {Ts[i]}, batch_size: {batch_sizes[i]} --> FIT DONE: [{t2-t1} s], MSE: {mse}, ACC: {score}\n')
                
                # wykresy
                
                # XDDDDD NIE WIEM JAK TO INACZEJ ZROBIC NIZ TAK
                sbtitle = '[T:' + str(Ts[i]) + ', batch_size:' + str(batch_sizes[i]) + ']' + ' MSE:' + str(mse) + ' ACC:' + str(score)
                if i%2 == 0:
                    if i == 0 or i == 1:
                        plt.figtext(0.3, 0.9, sbtitle)
                    elif i == 2 or i == 3:
                        plt.figtext(0.3, 0.62, sbtitle)
                    else:
                        plt.figtext(0.3, 0.35, sbtitle)
                else:
                    if i == 0 or i == 1:
                        plt.figtext(0.71, 0.9, sbtitle)
                    elif i == 2 or i == 3:
                        plt.figtext(0.71, 0.62, sbtitle)
                    else:
                        plt.figtext(0.71, 0.35, sbtitle)
                fig.suptitle(algo)
                steps = 20
                X1, X2 = np.meshgrid(np.linspace(0.0, np.pi, steps), np.linspace(0.0, np.pi, steps))
                X12 = np.array([X1.ravel(), X2.ravel()]).T
                y_ref = np.cos(X12[:, 0] * X12[:, 1]) * np.cos(2 * X12[:, 0])
                Y_ref = np.reshape(y_ref, (steps, steps))
                axs[countX][countY].plot_surface(X1, X2, Y_ref, cmap=cm.get_cmap("Spectral"))
                axs[countX][countY].scatter(X[:, 0], X[:, 1], y)
                axs[countX][countY].set(yticklabels=[])
                axs[countX][countY].set(xticklabels=[])
                countY = countY + 1
                y_pred = nn.predict(X12)
                Y_pred = np.reshape(y_pred, (steps, steps))
                axs[countX][countY].plot_surface(X1, X2, Y_pred, cmap=cm.get_cmap("Spectral"))
                axs[countX][countY].set(yticklabels=[])
                axs[countX][countY].set(xticklabels=[])
                countY = countY + 1
                if countY >= 4:
                    countX = countX + 1
                    countY = 0
            filename = algo+'.png'
            plt.savefig(filename,  bbox_inches='tight')
    plt.show()
    
def test_single(X, y, t, b_size, algo):
    nn = MultiLayerPerceptron(K=64, T=t, learning_rate=0.01, batch_size=b_size, alg=algo, seed=1)

    t1 = time.time()
    nn.fit(X, y)
    t2 = time.time()
    print(f'FIT DONE. [TIME: {t2 - t1} s]')

    predictions = nn.predict(X)
    mse = np.mean(0.5 * (y - predictions)**2)
    score = nn.score(X, y)
    print(f'T: {t}, batch_size: {b_size} --> FIT DONE: [{t2-t1} s], MSE: {mse}, ACC: {score}')

    # wykresy
    steps = 20
    X1, X2 = np.meshgrid(np.linspace(0.0, np.pi, steps), np.linspace(0.0, np.pi, steps))
    X12 = np.array([X1.ravel(), X2.ravel()]).T
    y_ref = np.cos(X12[:, 0] * X12[:, 1]) * np.cos(2 * X12[:, 0])
    Y_ref = np.reshape(y_ref, (steps, steps))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X1, X2, Y_ref, cmap=cm.get_cmap("Spectral"))
    ax.scatter(X[:, 0], X[:, 1], y)
    y_pred = nn.predict(X12)
    Y_pred = np.reshape(y_pred, (steps, steps))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap=cm.get_cmap("Spectral"))
    plt.show()
            
            
if __name__ == '__main__':
    X, y = fake_data(1000)
    # test_all(X, y)
    test_single(X=X, y=y, t=10**4, b_size=100, algo='rprop')
    