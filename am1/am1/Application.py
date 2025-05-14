import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


#exercice reconnaitre des images de chien et chat


def init(X):
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (w, b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def gradient(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def artificial_neuron(X_train, y_train, x_test, y_test, learning_rate=0.1, n_iter=1000):
    W, b = init(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)


        if i %10 == 0:
            train_loss.append(log_loss(A, y_train))
            y_prediction = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train,y_prediction))

            A_test = model(X_test, W, b)

            test_loss.append(log_loss(A_test, y_test))
            y_prediction = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test,y_prediction))

        dW, db = gradient(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.show()

    return (W, b)


def predict(X, W, b):
    A = model(X, W, b)
    return np.round(A, 0)


def load_data():
    train_dataset = h5py.File('trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels

    return X_train, y_train, X_test, y_test

x_train, y_train, x_test, y_test = load_data()



#par rapport aux fonctions x_train est en 3d ce qui passe pas pour
#le mettre en paramètre donc il faut l'applatir

#c'est comme faire x_train.shape[0], x_train.shape[1] * x_train.shape[2] on divise par la valeur max pour normaliser
X_reshape = x_train.reshape(x_train.shape[0], -1) / x_train.max()

X_test = x_test.reshape(x_test.shape[0], -1) / x_test.max()

W, b = artificial_neuron(X_reshape, y_train, x_test, y_test, learning_rate=0.01)
