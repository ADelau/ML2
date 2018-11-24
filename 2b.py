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

def get_y(x):
    """
    Get f(x) for the given x

    Arguments:
    ----------
    - `x`: the value of x.

    Return:
    -------
    - f(x)
    """

    return sin(x) + 0.5*sin(3*x) + random.gauss(0, 0.1)

def generate_dataset(size):
    """
    Generate a dataset

    Arguments:
    ----------
    - `size`: the size of the dataset.

    Return:
    -------
    - A tuple (x,y) representing the dataset.
    """

    x = []
    y = []

    for _ in range(size):
        newX = random.uniform(-4,4)
        x.append([newX])
        y.append(get_y(newX))

    x = np.array(x)
    x.reshape(-1, 1)

    return (x, y)

def generate_y_dataset(x, size):
    """
    Generate a dataset for a given x

    Arguments:
    ----------
    - `x`: the value of x.
    - `size`: the size of the dataset.

    Return:
    -------
    - A list of y representing the dataset.
    """

    y = []

    for _ in range(size):
        y.append(get_y(x))

    return y

def bias(x, datasetX0, regressors):
    """
    Compute the squared bias given a x, a dataset of y for this x
    and a list of trained regressors

    Arguments:
    ----------
    - `x`: the value of x.
    - `datasetX0`: a dataset of y for this x.
    - `regressors`: a list of trained regressors.

    Return:
    -------
    - The squared bias.
    """

    expect = np.mean(datasetX0)

    yHats = []
    for regressor in regressors:
        yHats.append(regressor.predict(x))

    return (expect - np.mean(yHats))**2

def variance(x, regressors):
    """ 
    Compute the estimation variance given a x
    and a list of trained regressors

    Arguments:
    ----------
    - `x`: the value of x.
    - `regressors`: a list of trained regressors.

    Return:
    -------
    - The estimation variance.
    """

    yHats = []
    for regressor in regressors:
        yHats.append(regressor.predict(x))

    return np.var(yHats)


if __name__ == "__main__":
    random.seed(0)
    trainingSize = 1000
    numberOfLS = 20
    nNeighbors = 5
    
    # Generate datasets and train linear and non-linear regressor in them
    X = []
    Y = []

    # Array of linear regressors
    lRegressions = []

     #Array of non-linear regressors
    nonLRegression = []
    for i in range(numberOfLS):
        # Generate a dataset
        dataset = generate_dataset(trainingSize)
        X.append(dataset[0])
        Y.append(dataset[1])

        # Create and train a linear regressor on the dataset
        lr = LinearRegression()
        lr.fit(dataset[0], dataset[1])
        lRegressions.append(lr)

        # Create and train a non-linear regressor on the dataset
        knn = KNeighborsRegressor(nNeighbors)
        knn.fit(dataset[0], dataset[1])
        nonLRegression.append(knn)

    # Generate datasets for each x
    x = np.linspace(-4,4,100)
    datasetsX0 = []
    for i in x:
        datasetsX0.append(generate_y_dataset(i, trainingSize))


    
    # Residual error
    y = []

    # Compute residual error for each x
    for data in datasetsX0:
        y.append(np.var(data))

    plt.figure()
    plt.plot(x, y, label='Empirical results')

    y = []
    # Plot the theoritical result for the residual error
    for i in x:
        y.append(0.01)
    plt.plot(x, y, label='Theoritical results')
    plt.legend()
    plt.savefig(plotFolder + "resErrorX0.eps")

    # Bias - Linear Regressor
    plt.figure()
    y = []
    # Plot a linear regressor model
    for i in x:
        y.append(lRegressions[0].predict(i))
    plt.plot(x, y, label="Linear Regression Trained Model", color="red")
    plt.plot(X[0], Y[0], "bo", markersize=1, label="Dataset")
    plt.savefig(plotFolder + "linearRegressorX0.eps")

    plt.figure()
    y = []
    # Compute squared bias for each x
    for i in range(len(x)):
        y.append(bias(x[i], datasetsX0[i], lRegressions))

    plt.plot(x, y, label="Squared bias")

    plt.legend()
    plt.savefig(plotFolder + "BiasLinearRegressorX0.eps")


    # Bias - Non-Linear Regressor
    plt.figure()
    y = []

    # Plot a non-linear regressor model
    for i in x:
        y.append(nonLRegression[0].predict(i))
    plt.plot(x, y, label="Non-Linear Regression Trained Model", color="red")
    plt.plot(X[0], Y[0], "bo", markersize=1, label="Dataset")
    plt.legend(loc = 'upper left')

    plt.savefig(plotFolder + "nonLinearRegressorX0.eps")
    plt.figure()

    y = []
    # Compute squared bias for each x
    for i in range(len(x)):
        y.append(bias(x[i], datasetsX0[i], nonLRegression))

    plt.plot(x, y, label="Squared bias")

    plt.legend()

    plt.savefig(plotFolder + "BiasNonLinearRegressorX0.eps")

    # Variance - Linear Regressor
    plt.figure()

    y = []

    # Compute the variance for each x
    for i in x:
        y.append(variance(i, lRegressions))

    plt.plot(x, y, label="Variance")
    plt.legend()

    plt.savefig(plotFolder + "varianceLinearRegressorX0.eps")

    # Variance - Non-Linear Regressor
    plt.figure()

    y = []
    # Compute the variance for each x
    for i in x:
        y.append(variance(i, nonLRegression))

    plt.plot(x, y, label="Variance")
    plt.legend()

    plt.savefig(plotFolder + "varianceNonLinearRegressorX0.eps")




