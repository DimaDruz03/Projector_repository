import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
    
np.random.seed(0)
X, y = skds.make_blobs(200, centers=2, cluster_std=0.9)
X[0] += 1.5

def activation_func(x):
    return np.where(x > 0, 1, 0)

def perc(w, x):
    return activation_func(np.dot(x, w) + bias)

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

def train_and_visualize():
    global prev_error, error, prev_weights, weights

    prediction = perc(weights, X)
    diff = y - prediction
    prev_error = error
    error = np.mean(np.abs(diff))

    prev_weights = weights
    weights = weights + LR*np.dot(X.T, diff)


success = False
LR = 0.1

bias_min = -20
bias_max = 20
bias_step = 1
bias = bias_min - bias_step

min_errors = {}
min_errors[f"bias={bias}"] = 1.0
patience = 20

best_weights_dict = {}

while bias < bias_max and min_errors[f"bias={round(bias, 2)}"] > 0:
    bias += bias_step

    success = False
    prev_error = 2.0
    error = 1.0
    min_error = error
    np.random.seed(10000000)
    prev_weights = tuple(2*np.random.random((2,))-1)
    weights = prev_weights
    best_weights = weights

    while not success:
        while prev_error > error :
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
            round_bias = round(bias, 2)
            min_errors[f"bias={round_bias}"] = min_error
            best_weights_dict[f"bias={round_bias}"] = best_weights
            print(f"Min error = {min_error} for bias = {round_bias}; Best weights: w1 = {best_weights[0]}; w2 = {best_weights[1]};")

key = ""
min = 1
for i in min_errors:
    if min_errors[i] < min:
        key = i
        min = min_errors[i]
        
bias = float(key[5:])
weights = best_weights_dict[key]
print(f"The best {key}; min error = {min}; Best weights: w1 = {weights[0]}; w2 = {weights[1]};")
plot_decision_boundary(lambda x: perc(weights, x), False)
