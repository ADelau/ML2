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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

NB_LS = 10
SIZE_EY = 10000
SIZE_TS = 1000

PLOT_FOLDER = "graphs/"

REFERENCE_SIZE_LS = 50
REFERENCE_STD = 1
LINEAR_REFERENCE_COMPLEXITY = 5
NONLINEAR_REFERENCE_COMPLEXITY = 5

SIZES_LS = [x*5 for x in range(1, 21)]
STDS = [x/10 for x in range(21)]
LINEAR_COMPLEXITY = [x for x in range(1, 11)]
NONLINEAR_COMPLEXITY = [x for x in range(1, 11)]
NB_X = 100
X_TO_COMPUTE = np.linspace(-4, 4, NB_X)


def get_y(x, noiseSTD):
    return sin(x) + 0.5*sin(3*x) + random.gauss(0, noiseSTD)

def generate_y_dataset(x, size, noiseSTD):
    """
    Generate a dataset for a given x

    Arguments:
    ----------
    - `x`: the value of x.
    - `size`: the size of the dataset.
    - 'noiseSTD': the standard deviation of epsilon

    Return:
    -------
    - A list of y representing the dataset.
    """

    y = []

    for _ in range(size):
        y.append(get_y(x, noiseSTD))

    return y

def generate_dataset(size, noiseSTD):
	 """
    Generate a dataset for a given x

    Arguments:
    ----------
    - `size`: the size of the dataset.
    - 'noiseSTD': the standard deviation of epsilon

    Return:
    -------
    - 'x': A list of y representing the dataset.
    """
	x = []
	y = []

	for _ in range(size):
	    newX = random.uniform(-4,4)
	    x.append([newX])
	    y.append(get_y(newX, noiseSTD))

	x = np.array(x)
	x.reshape(-1, 1)
	y = np.array(y)
	y.reshape(-1, 1)

	return (x, y)

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


def get_linear_fitted_regressor(x, y, poly):
	linearRegressor = LinearRegression()
	xLinear = poly.fit_transform(x)
	linearRegressor.fit(xLinear, y)

	return linearRegressor

def get_nonlinear_fitted_regressor(x, y, complexity):
	nonLinearRegressor = KNeighborsRegressor(complexity)
	nonLinearRegressor.fit(x, y)

	return nonLinearRegressor

def mean_res_error(noiseSTD):
	resErrors = []
	for x in X_TO_COMPUTE:
		y = generate_y_dataset(x, SIZE_EY, noiseSTD)
		resErrors.append(np.var(y))

	resErrors = np.array(resErrors)
	return np.mean(resErrors)

def mean_squared_bias(sizeLS, noiseSTD, linearComplexity, nonLinearComplexity):
	poly = PolynomialFeatures(degree = linearComplexity)

	linearXSquaredBias = []
	nonlinearXSquaredBias = []
	for x in X_TO_COMPUTE:
		linearRegressors = []
		nonlinearRegressors = []
		datasetX0 = generate_y_dataset(x, SIZE_EY, noiseSTD)
		for i in range(NB_LS):
			trainX, trainY = generate_dataset(sizeLS, noiseSTD)
			linearRegressors.append(get_linear_fitted_regressor(trainX, trainY, poly))
			nonlinearRegressors.append(get_nonlinear_fitted_regressor(trainX, trainY, nonLinearComplexity))
		
		linearXSquaredBias.append(bias(poly.fit_transform(x), datasetX0, linearRegressors))
		nonlinearXSquaredBias.append(bias(x, datasetX0, nonlinearRegressors))

	linearXSquaredBias = np.array(linearXSquaredBias)
	nonlinearXSquaredBias = np.array(nonlinearXSquaredBias)
	linearSquaredBias = np.mean(linearXSquaredBias)
	nonlinearSquaredBias = np.mean(nonlinearXSquaredBias)

	return linearSquaredBias, nonlinearSquaredBias


def mean_variance(sizeLS, noiseSTD, linearComplexity, nonLinearComplexity):
	poly = PolynomialFeatures(degree = linearComplexity)

	linearXVariance = []
	nonlinearXVariance = []
	for x in X_TO_COMPUTE:
		linearRegressors = []
		nonlinearRegressors = []
		for i in range(NB_LS):
			trainX, trainY = generate_dataset(sizeLS, noiseSTD)
			linearRegressors.append(get_linear_fitted_regressor(trainX, trainY, poly))
			nonlinearRegressors.append(get_nonlinear_fitted_regressor(trainX, trainY, nonLinearComplexity))
		
		linearXVariance.append(variance(poly.fit_transform(x), linearRegressors))
		nonlinearXVariance.append(variance(x, nonlinearRegressors))

	linearXVariance = np.array(linearXVariance)
	nonlinearXVariance = np.array(nonlinearXVariance)
	linearVariance = np.mean(linearXVariance)
	nonlinearVariance = np.mean(nonlinearXVariance)

	return linearVariance, nonlinearVariance
	

#make size vary
def plot_size():
	
	resError = [mean_res_error(REFERENCE_STD)]*len(SIZES_LS)
	linearSquaredBias = []
	nonLinearSquaredBias = []
	linearVariance = []
	nonLinearVariance = []
	
	for sizeLS in SIZES_LS:
		print(sizeLS)
		tmpLinearSquaredBias, tmpNonLinearSquaredBias = mean_squared_bias(sizeLS, REFERENCE_STD, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = mean_variance(sizeLS, REFERENCE_STD, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.title("evolution w.r.t size for polynomial regression")
	plt.plot(SIZES_LS, resError, "b-", label = "Residual error")
	plt.plot(SIZES_LS, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(SIZES_LS, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("size")
	plt.savefig(PLOT_FOLDER + "linearSize")

	plt.figure()
	plt.title("evolution w.r.t size for knn regression")
	plt.plot(SIZES_LS, resError, "b-", label = "Residual error")
	plt.plot(SIZES_LS, nonLinearSquaredBias, "r-", label = "Squared bias")
	plt.plot(SIZES_LS, nonLinearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("size")
	plt.savefig(PLOT_FOLDER + "nonLinearSize")
	

#make noise std vary
def plot_std():
	resError = []
	linearSquaredBias = []
	nonLinearSquaredBias = []
	linearVariance = []
	nonLinearVariance = []
	
	for std in STDS:
		print(std)
		resError.append(mean_res_error(std))

		tmpLinearSquaredBias, tmpNonLinearSquaredBias = mean_squared_bias(REFERENCE_SIZE_LS, std, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = mean_variance(REFERENCE_SIZE_LS, std, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.title("evolution w.r.t noise std for polynomial regression")
	plt.plot(STDS, resError, "b-", label = "Residual error")
	plt.plot(STDS, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(STDS, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("noise std")
	plt.savefig(PLOT_FOLDER + "linearSTD")

	plt.figure()
	plt.title("evolution w.r.t noise std for knn regression")
	plt.plot(STDS, resError, "b-", label = "Residual error")
	plt.plot(STDS, nonLinearSquaredBias, "r-", label = "Squared bias")
	plt.plot(STDS, nonLinearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("noise std")
	plt.savefig(PLOT_FOLDER + "nonLinearSTD")

#make complexity vary
def plot_complexity():
	resError = [mean_res_error(REFERENCE_STD)]*min(len(LINEAR_COMPLEXITY), len(NONLINEAR_COMPLEXITY))
	linearSquaredBias = []
	nonLinearSquaredBias = []
	linearVariance = []
	nonLinearVariance = []
	
	for i in range(min(len(LINEAR_COMPLEXITY), len(NONLINEAR_COMPLEXITY))):
		print(i)
		tmpLinearSquaredBias, tmpNonLinearSquaredBias = mean_squared_bias(REFERENCE_SIZE_LS, REFERENCE_STD, LINEAR_COMPLEXITY[i], NONLINEAR_COMPLEXITY[i])
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = mean_variance(REFERENCE_SIZE_LS, REFERENCE_STD, LINEAR_COMPLEXITY[i], NONLINEAR_COMPLEXITY[i])
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.title("evolution w.r.t polynom degree for polynomial regression")
	plt.plot(LINEAR_COMPLEXITY, resError, "b-", label = "Residual error")
	plt.plot(LINEAR_COMPLEXITY, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(LINEAR_COMPLEXITY, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("polynom degree")
	plt.savefig(PLOT_FOLDER + "linearComplexity")

	plt.figure()
	plt.title("evolution w.r.t number of neighbours for knn regression")
	plt.plot(NONLINEAR_COMPLEXITY, resError, "b-", label = "Residual error")
	plt.plot(NONLINEAR_COMPLEXITY, nonLinearSquaredBias, "r-", label = "Squared bias")
	plt.plot(NONLINEAR_COMPLEXITY, nonLinearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("number of neighbours")
	plt.savefig(PLOT_FOLDER + "nonLinearComplexity")


if __name__ == "__main__":
	random.seed(11)
	print("size \n")
	plot_size()
	print("std \n")
	plot_std()
	print("complexity \n")
	plot_complexity()

