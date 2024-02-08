import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds

class_amount = 3
features_amount = 2
output_amount = 1
np.random.seed(0)
X, y = skds.make_blobs(200)

objects = []
labels = []
indices = []
for t in range(class_amount):
    i = np.array([])
    l = np.array([])
    for idx in range(len(y)):
        if y[idx] == t:
            i = np.append(i, idx)
            l = np.append(l, y[idx])
    indices.append(i)
    labels.append(l)

for il in indices:
    array = np.empty((0,2))
    for idx in range(len(il)):
        array = np.vstack((array, X[int(il[idx])]))
    objects.append(array)

def softmax(z):
    res = np.array([])
    for vector in z:
        exp_z = np.exp(vector)
        res = np.append(res, exp_z / np.sum(exp_z))

    return res

def predict(w, x):
    hi_outputs = np.dot(w, x.T)

    return np.array([np.argmax(i) for i in softmax(hi_outputs.T).reshape((200,3))])

def train_and_visualize():
    global prev_error, error, prev_weight_matrix, weight_matrix

    prediction = predict(weight_matrix, X)

    pred_list = []
    for il in indices:
        p_list =  np.array([])
        for idx in il:
            p_list = np.append(p_list, prediction[int(idx)])
        pred_list.append(p_list)

    diff_list = [labels[0] - pred_list[0], labels[1] - pred_list[1], labels[2] - pred_list[2]]
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111, facecolor='black')
    ax.scatter(X[:, 0], X[:, 1], s=10, c=prediction, cmap=plt.cm.Spectral)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

    prev_weight_matrix = np.copy(weight_matrix)

    for idx1 in range(class_amount):
        for idx2 in range(features_amount):
            weight_matrix[idx1, idx2] = weight_matrix[idx1, idx2] + LR * np.dot(objects[idx1].T[idx2], diff_list[idx1])

success = False
LR = 7
patience = 20

while True:
    success = False
    prev_error = 2.0
    error = 1.0
    min_error = error
    prev_weight_matrix = 2*np.random.random((class_amount, features_amount))
    weight_matrix = prev_weight_matrix
    best_weight_matrix = weight_matrix

    while not success:
        while prev_error > error :
            train_and_visualize()
            if error < min_error or error == 0:
                min_error = error
                best_weight_matrix = prev_weight_matrix
    
        success = True

        if min_error > 0:
            for i in range(patience):
                train_and_visualize()
                if error < min_error:
                    min_error = error
                    best_weight_matrix = prev_weight_matrix
                    success = False
                    break

