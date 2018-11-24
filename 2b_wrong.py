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

def bias(x, classifiers):
    N = 2000
    expect = np.mean(generateYDataset(x, N))

    yHats = []
    for classifier in classifiers:
        yHats.append(classifier.predict(x))

    return (expect - np.mean(yHats))**2




if __name__ == "__main__":
    # print(generateDataset(1000,0))
    random.seed(0)
    trainingSize = 1000
    numberOfLS = 20
    
    X = []
    Y = []
    lRegressions = []
    for i in range(numberOfLS):
        dataset = generateDataset(trainingSize)
        X.append(dataset[0])
        Y.append(dataset[1])

        lr = LinearRegression()
        lr.fit(dataset[0], dataset[1])
        lRegressions.append(lr)

    # oneDimX = []
    # for item in X:
    #     oneDimX.append(item[0])

    # pol = np.polyfit(oneDimX, Y, 5)

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
    # y = []
    # for i in x:
    #     y.append(bias(i, lRegressions))
    # plt.plot(x, y, label="Linear Regression Trained Model", color="red")
    # plt.plot(X[0], Y[0], "bo", markersize=1, label="Dataset")
    # plt.legend()

    # plt.savefig(plotFolder + "linearRegressorX0.eps")

    # plt.figure()

    y = []
    for i in x:
        y.append(bias(i, lRegressions))

    plt.plot(x, y, label="Squared bias")
    plt.show()

    plt.legend()
    plt.savefig(plotFolder + "BiasLinearRegressorX0.eps")


    # Bias - Non-Linear Regressor
    # plt.figure()
    # y = []
    # y = np.polyval(pol, x)
    # plt.plot(x, y, label="Non-Linear Regression Trained Model", color="red")
    # plt.plot(X, Y, "bo", markersize=1, label="Dataset")
    # plt.legend(loc = 'upper left')

    # plt.savefig(plotFolder + "nonLinearRegressorX0.eps")

    # plt.figure()

    # y = []
    # for i in x:
    #     y.append(bias(i, pol))

    # plt.plot(x, y, label="Empirical results")

    # plt.legend()
    # plt.savefig(plotFolder + "BiasNonLinearRegressorX0.eps")

