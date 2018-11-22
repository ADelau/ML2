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

def get_y(x):
    return sin(x) + 0.5*sin(3*x) + random.gauss(0, 0.1)

def generate_dataset(size):
	x = []
	y = []

	for _ in range(size):
	    newX = random.uniform(-4,4)
	    x.append([newX])
	    y.append(get_y(newX))

	x = np.array(x)
	x.reshape(-1, 1)
	y = np.array(y)
	y.reshape(-1, 1)

	return (x, y)

def get_fitted_classifiers(x, y):
	linearClassifier = LinearRegression()
	linearClassifier.fit(x, y)

	nonLinearClassifier = linearClassifier #À CHANGER !!!

	return linearClassifier, nonLinearClassifier

def res_error(x, y):
    return np.var(y)

def squared_bias(trainX, trainY, testX):
	Ey = np.mean(trainY)

	linearClassifier, nonLinearClassifier = get_fitted_classifiers(trainX, trainY)

	linearYPred = linearClassifier.predict(testX)
	nonLinearYPred = nonLinearClassifier.predict(testX)

	linearEYPred = np.mean(linearYPred)
	nonLinearEYPred = np.mean(nonLinearYPred)

	linearSquaredBias = (Ey - linearEYPred)**2
	nonLinearSquaredBias = (Ey - nonLinearEYPred)**2

	return linearSquaredBias, nonLinearSquaredBias


def variance(trainX ,trainY, testX):
	linearClassifier, nonLinearClassifier = get_fitted_classifiers(trainX, trainY)
	linearYPred = linearClassifier.predict(testX)
	nonLinearYPred = nonLinearClassifier.predict(testX)

	linearVariance = np.var(linearYPred)
	nonLinearVariance = np.var(nonLinearYPred)

	return linearVariance, nonLinearVariance
	

if __name__ == "__main__":
	random.seed(11)
	SIZE = 2000

	trainX, trainY = generate_dataset(SIZE)
	testX, testY = generate_dataset(SIZE)

	print("Residual error = {} \n".format(res_error(trainX, trainY)))

	linearSquaredBias, nonLinearSquaredBias = squared_bias(trainX, trainY, testX)
	print("Linear squared bias = {} \n".format(linearSquaredBias))
	print("Non-linear squared bias = {} \n".format(nonLinearSquaredBias))

	linearVariance, nonLinearVariance = variance(trainX, trainY, testX)
	print("Linear variance = {} \n".format(linearVariance))
	print("Non-linear variance = {} \n".format(nonLinearVariance))