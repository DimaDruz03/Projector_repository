import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
    
np.random.seed(0)
X, y = skds.make_blobs(200, centers=2, cluster_std=0.9)
X[0] += 1.5

def activation_func(x):
    res = np.where(x > 0, 1, 0)
    return res

def perc(w, x):
    res = activation_func(np.dot(x, w))
    return res

def plot_decision_boundary(pred_func, close_fig=True):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap=plt.cm.Spectral)
    if close_fig:
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
    else: plt.show()

def train_and_visualize(close_fig=True):
    global prev_error, error, prev_weights, weights
    prediction = perc(weights, X)
    diff = y - prediction
    prev_error = error
    error = np.mean(np.abs(diff))
    plot_decision_boundary(lambda x: perc(weights, x), close_fig)

    prev_weights = weights
    weights = weights + LR*np.dot(X.T, diff)

success = False
LR = 1.1
patience = 10
min_errors = {}
 
while(True):
    LR -= 0.1
    if LR < 0.1 : break
    success = False
    prev_error = 2.0
    error = 1.0
    min_error = error
    prev_weights = 2*np.random.random((2,))-1
    weights = prev_weights
    best_weights = weights

    while not success:
        while prev_error > error:
            train_and_visualize()
            if error < min_error or error == 0:
                min_error = error
                best_weights = prev_weights
    
        success = True

        if min_error > 0:
            for i in range(patience):
                train_and_visualize()
                if error < min_error:
                    min_error = error
                    best_weights = prev_weights
                    success = False
                    break

        if success == True :
            print(f"Min error = {min_error} for LR = {np.round(LR, 2)}.")
            min_errors[f"LR={np.round(LR, 2)}"] = min_error

min_lr = 1.0
for i in range(1, 11, 1):
    lr = i / 10
    if min_errors[f"LR={np.round(lr, 2)}"] < min_errors[f"LR={np.round(min_lr, 2)}"]: min_lr = lr

print(f"The best LR={min_lr}. min error = {min_errors[f"LR={min_lr}"]}")
weights = best_weights
train_and_visualize(False)
