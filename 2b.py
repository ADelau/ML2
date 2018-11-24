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
from sklearn.neighbors import KNeighborsRegressor

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

def bias(x, datasetX0, classifiers):
    expect = np.mean(datasetX0)

    yHats = []
    for classifier in classifiers:
        yHats.append(classifier.predict(x))

    return (expect - np.mean(yHats))**2

def variance(x, classifiers):

    yHats = []
    for classifier in classifiers:
        yHats.append(classifier.predict(x))

    return np.var(yHats)


if __name__ == "__main__":
    random.seed(0)
    trainingSize = 1000
    numberOfLS = 20
    nNeighbors = 5
    
    X = []
    Y = []
    lRegressions = []
    nonLRegression = []
    for i in range(numberOfLS):
        dataset = generateDataset(trainingSize)
        X.append(dataset[0])
        Y.append(dataset[1])

        lr = LinearRegression()
        lr.fit(dataset[0], dataset[1])
        lRegressions.append(lr)

        knn = KNeighborsRegressor(nNeighbors)
        knn.fit(dataset[0], dataset[1])
        nonLRegression.append(knn)

    x = np.linspace(-4,4,100)
    datasetsX0 = []
    for i in x:
        datasetsX0.append(generateYDataset(i, trainingSize))

    # oneDimX = []
    # for item in X:
    #     oneDimX.append(item[0])

    # pol = np.polyfit(oneDimX, Y, 5)

    
    # Residual error
    y = []
    for data in datasetsX0:
        y.append(np.var(data))

    plt.figure()
    plt.plot(x, y, label='Empirical results')

    y = []
    for i in x:
        y.append(0.01)
    plt.plot(x, y, label='Theoritical results')
    plt.legend()
    plt.savefig(plotFolder + "resErrorX0.eps")

    # Bias - Linear Regressor
    plt.figure()
    y = []
    for i in x:
        y.append(lRegressions[0].predict(i))
    plt.plot(x, y, label="Linear Regression Trained Model", color="red")
    plt.plot(X[0], Y[0], "bo", markersize=1, label="Dataset")
    plt.savefig(plotFolder + "linearRegressorX0.eps")

    plt.figure()
    y = []
    for i in range(len(x)):
        y.append(bias(x[i], datasetsX0[i], lRegressions))

    plt.plot(x, y, label="Squared bias")

    plt.legend()
    plt.savefig(plotFolder + "BiasLinearRegressorX0.eps")


    # Bias - Non-Linear Regressor
    plt.figure()
    y = []
    for i in x:
        y.append(nonLRegression[0].predict(i))
    plt.plot(x, y, label="Non-Linear Regression Trained Model", color="red")
    plt.plot(X[0], Y[0], "bo", markersize=1, label="Dataset")
    plt.legend(loc = 'upper left')

    plt.savefig(plotFolder + "nonLinearRegressorX0.eps")
    plt.figure()

    y = []
    for i in range(len(x)):
        y.append(bias(x[i], datasetsX0[i], nonLRegression))

    plt.plot(x, y, label="Squared bias")

    plt.legend()

    plt.savefig(plotFolder + "BiasNonLinearRegressorX0.eps")

    # Variance - Linear Regressor
    plt.figure()

    y = []
    for i in x:
        y.append(variance(i, lRegressions))

    plt.plot(x, y, label="Variance")
    plt.legend()

    plt.savefig(plotFolder + "varianceLinearRegressorX0.eps")

    # Variance - Non-Linear Regressor
    plt.figure()

    y = []
    for i in x:
        y.append(variance(i, nonLRegression))

    plt.plot(x, y, label="Variance")
    plt.legend()

    plt.savefig(plotFolder + "varianceNonLinearRegressorX0.eps")




