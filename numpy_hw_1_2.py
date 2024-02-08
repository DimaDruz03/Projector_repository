import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
    
np.random.seed(0)
X, y = skds.make_blobs(200, centers=2, cluster_std=0.9)
X[0] += 1.5

def heavyside(x):
    return np.where(x > 0, 1, 0)

def predict(w, x):
    return heavyside(np.dot(x, w))

def plot_decision_boundary(pred_func, close_fig=True):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    flat = np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(np.hstack((np.ones((flat.shape[0], 1)), flat)))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap=plt.cm.Spectral)
    if close_fig:
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
    else: plt.show()

def train_and_visualize():
    global prev_error, error, prev_weights, weights

    prediction = predict(weights, np.hstack((np.ones((X.shape[0], 1)), X)))
    diff = y - prediction
    prev_error = error
    error = np.mean(np.abs(diff))
    plot_decision_boundary(lambda x: predict(weights, x))

    prev_weights = weights
    weights = weights + LR*np.dot(np.vstack((np.ones((1, X.T.shape[1])), X.T)), diff)


success = False
LR = 0.01
patience = 50

prev_error = 1.0
error = prev_error
min_error = error
prev_weights = 2*np.random.random((3,))-1
weights = prev_weights
best_weights = weights

while not success:
    while prev_error > error :
        train_and_visualize()
        if error < min_error or error == 0:
            min_error = error
            best_weights = prev_weights
            if error == 0: break

    success = True
    if min_error > 0:
        for i in range(patience):
            train_and_visualize()
            if error < min_error or error == 0:
                min_error = error
                best_weights = prev_weights
                if error > 0: success = False
                break
        
print(f"Min error = {min_error}; Best weights: w_bias = {best_weights[0]}; w1 = {best_weights[1]}; w2 = {best_weights[2]}")
plot_decision_boundary(lambda x: predict(best_weights, x), False)
