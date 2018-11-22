#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
import random
from math import sin
import math

plotFolder = "graphs/"

def getY(x):
    return sin(x) + 0.5*sin(3*x) + random.gauss(0, 0.1)

def generateDataset(size):
    x = []
    y = []

    for _ in range(size):
        newX = random.uniform(-4,4)
        x.append([newX])
        y.append(getY(newX))

    x = np.array(x)
    x.reshape(-1, 1)

    return (x, y)

def generateYDataset(x, size):
    y = []

    for _ in range(size):
        y.append(getY(x))

    return y

def resError(x):
    N = 2000

    y = generateYDataset(x, N)
    return np.var(y)

def bias(x, classifier):
    N = 2000
    expect = np.mean(generateYDataset(x, N))

    yHat = classifier.predict(x)

    return (expect - yHat[0])**2


if __name__ == "__main__":
    # print(generateDataset(1000,0))
    random.seed(0)

    
    dataset = generateDataset(1000)
    # X = np.reshape(dataset[0], (-1,1))
    X = dataset[0]
    Y = dataset[1]


    lr = LinearRegression()
    lr.fit(X, Y)

    x = np.linspace(-4,4,100)
    
    # Residual error
    # y = []
    # for i in x:
    #     y.append(resError(i))

    # plt.figure()
    # plt.plot(x, y, label='Empirical results')

    # y = []
    # for i in x:
    #     y.append(0.01)
    # plt.plot(x, y, label='Theoritical results')
    # plt.legend()

    # plt.savefig(plotFolder + "resErrorX0.eps")

    # Bias - Linear Regressor
    plt.figure()
    y = []
    for i in x:
        y.append(lr.predict(i))
    plt.plot(x, y, label="Linear Regression Trained Model")
    plt.plot(X, Y, "bo", markersize=1, label="Dataset")
    plt.legend()

    plt.savefig(plotFolder + "linearRegressorX0.eps")

    plt.figure()

    y = []
    for i in x:
        y.append(bias(i, lr))

    plt.plot(x, y, label="Empirical results")

    y = []
    for i in x:
        y.append((sin(i) + 0.5*sin(3*i) - lr.predict(i))**2)
    plt.plot(x, y, label="Theoritical results")

    plt.legend()
    plt.savefig(plotFolder + "BiasLinearRegressorX0.eps")

