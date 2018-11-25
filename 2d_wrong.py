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

REFERENCE_SIZE_LS = 1000
REFERENCE_STD = 0.1
LINEAR_REFERENCE_COMPLEXITY = 1
NONLINEAR_REFERENCE_COMPLEXITY = 1

SIZES_LS = [x*100 for x in range(1, 100)]
STDS = [x/20 for x in range(100)]
LINEAR_COMPLEXITY = [x for x in range(1, 10)]
NONLINEAR_COMPLEXITY = [x for x in range(1, 10)]

def get_y(x, noiseSTD):
    return sin(x) + 0.5*sin(3*x) + random.gauss(0, noiseSTD)

def generate_dataset(size, noiseSTD):
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

def get_fitted_regressors(x, y, poly, nonLinearComplexity):
	linearRegressor = LinearRegression()
	xLinear = poly.fit_transform(x)
	linearRegressor.fit(xLinear, y)

	nonLinearRegressor = KNeighborsRegressor(nonLinearComplexity)
	nonLinearRegressor.fit(x, y)

	return linearRegressor, nonLinearRegressor

def res_error(noiseSTD):
	x, y = generate_dataset(SIZE_EY, noiseSTD)
	return np.var(y)

def squared_bias(sizeLS, noiseSTD, linearComplexity, nonLinearComplexity):
	poly = PolynomialFeatures(degree = linearComplexity)
	x, y = generate_dataset(SIZE_EY, noiseSTD)
	Ey = np.mean(y)
	linearYPred = []
	nonLinearYPred = []

	for i in range(NB_LS):
		trainX, trainY = generate_dataset(sizeLS, noiseSTD)
		linearRegressor, nonLinearRegressor = get_fitted_regressors(trainX, trainY, poly, nonLinearComplexity)

		testX, testY = generate_dataset(SIZE_TS, noiseSTD)
		linearTestX = poly.fit_transform(testX)
		linearYPred.extend(linearRegressor.predict(linearTestX))
		nonLinearYPred.extend(nonLinearRegressor.predict(testX))

	linearYPred = np.array(linearYPred)
	nonLinearYPred = np.array(nonLinearYPred)
	linearEYPred = np.mean(linearYPred)
	nonLinearEYPred = np.mean(nonLinearYPred)

	linearSquaredBias = (Ey - linearEYPred)**2
	nonLinearSquaredBias = (Ey - nonLinearEYPred)**2

	return linearSquaredBias, nonLinearSquaredBias


def variance(sizeLS, noiseSTD, linearComplexity, nonLinearComplexity):
	poly = PolynomialFeatures(degree = linearComplexity)
	linearYPred = []
	nonLinearYPred = []

	for i in range(NB_LS):
		trainX, trainY = generate_dataset(sizeLS, noiseSTD)
		linearRegressor, nonLinearRegressor = get_fitted_regressors(trainX, trainY, poly, nonLinearComplexity)

		testX, testY = generate_dataset(SIZE_TS, noiseSTD)
		linearTestX = poly.fit_transform(testX)
		linearYPred.append(np.mean(linearRegressor.predict(linearTestX)))
		nonLinearYPred.append(np.mean(nonLinearRegressor.predict(testX)))

	linearYPred = np.array(linearYPred)
	nonLinearYPred = np.array(nonLinearYPred)
	linearVariance = np.var(linearYPred)
	nonLinearVariance = np.var(nonLinearYPred)

	return linearVariance, nonLinearVariance
	

#make size vary
def plot_size():
	
	resError = [res_error(REFERENCE_STD)]*len(SIZES_LS)
	linearSquaredBias = []
	nonLinearSquaredBias = []
	linearVariance = []
	nonLinearVariance = []
	
	for sizeLS in SIZES_LS:
		tmpLinearSquaredBias, tmpNonLinearSquaredBias = squared_bias(sizeLS, REFERENCE_STD, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = variance(sizeLS, REFERENCE_STD, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.plot(SIZES_LS, resError, "b-", label = "Residual error")
	plt.plot(SIZES_LS, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(SIZES_LS, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("size")
	plt.savefig(PLOT_FOLDER + "linearSize")

	plt.figure()
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
		resError.append(res_error(std))

		tmpLinearSquaredBias, tmpNonLinearSquaredBias = squared_bias(REFERENCE_SIZE_LS, std, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = variance(REFERENCE_SIZE_LS, std, LINEAR_REFERENCE_COMPLEXITY, NONLINEAR_REFERENCE_COMPLEXITY)
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.plot(STDS, resError, "b-", label = "Residual error")
	plt.plot(STDS, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(STDS, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("noise std")
	plt.savefig(PLOT_FOLDER + "linearSTD")

	plt.figure()
	plt.plot(STDS, resError, "b-", label = "Residual error")
	plt.plot(STDS, nonLinearSquaredBias, "r-", label = "Squared bias")
	plt.plot(STDS, nonLinearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("noise std")
	plt.savefig(PLOT_FOLDER + "nonLinearSTD")

#make complexity vary
def plot_complexity():
	resError = [res_error(REFERENCE_STD)]*min(len(LINEAR_COMPLEXITY), len(NONLINEAR_COMPLEXITY))
	linearSquaredBias = []
	nonLinearSquaredBias = []
	linearVariance = []
	nonLinearVariance = []
	
	for i in range(min(len(LINEAR_COMPLEXITY), len(NONLINEAR_COMPLEXITY))):
		tmpLinearSquaredBias, tmpNonLinearSquaredBias = squared_bias(REFERENCE_SIZE_LS, REFERENCE_STD, LINEAR_COMPLEXITY[i], NONLINEAR_COMPLEXITY[i])
		linearSquaredBias.append(tmpLinearSquaredBias)
		nonLinearSquaredBias.append(tmpNonLinearSquaredBias)

		tmpLinearVariance, tmpNonLinearVariance = variance(REFERENCE_SIZE_LS, REFERENCE_STD, LINEAR_COMPLEXITY[i], NONLINEAR_COMPLEXITY[i])
		linearVariance.append(tmpLinearVariance)
		nonLinearVariance.append(tmpNonLinearVariance)

	plt.figure()
	plt.plot(LINEAR_COMPLEXITY, resError, "b-", label = "Residual error")
	plt.plot(LINEAR_COMPLEXITY, linearSquaredBias, "r-", label = "Squared bias")
	plt.plot(LINEAR_COMPLEXITY, linearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("complexity")
	plt.savefig(PLOT_FOLDER + "linearComplexity")

	plt.figure()
	plt.plot(NONLINEAR_COMPLEXITY, resError, "b-", label = "Residual error")
	plt.plot(NONLINEAR_COMPLEXITY, nonLinearSquaredBias, "r-", label = "Squared bias")
	plt.plot(NONLINEAR_COMPLEXITY, nonLinearVariance, "k-", label = "Variance")
	plt.legend()
	plt.xlabel("complexity")
	plt.savefig(PLOT_FOLDER + "nonLinearComplexity")


if __name__ == "__main__":
	random.seed(11)
	plot_size()
	plot_std()
	plot_complexity()

