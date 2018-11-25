#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

def generate_x(y):
	""" 
    Generate a set of input x following a normal distribution with the
    parameters set according to the class given as input

    Arguments:
    ----------
    - `y`: the class the dataset should be generated for.

    Return:
    -------
    - An array containing a set of values x_0, x_1.
    """

	if y == -1:
		mean = np.array([0, 0])
		cov = np.array([[1, 0], [0, 1]])
	else:
		mean = np.array([0, 0])
		cov = np.array([[2, 0], [0, 1/2]])

	x = np.random.multivariate_normal(mean, cov, size = 1)
	return x[0]

def generate_dataset(size):
	""" 
    Generate a dataset x and y according the distribution described in the 
    statement

    Arguments:
    ----------
    - `size`: the size of the dataset

    Return:
    -------
    - X: An array containing a set of values x_0, x_1.
    - Y: An array containing a set of classes corresponding to the inputs
    """

	classes = [-1, 1]
	y = random.choices(classes, k = size)
	X = [generate_x(item) for item in y]

	return X, y

def predict(x):
	""" 
    Predict a class according to the inputs x_0 and x_1 following the Bayes
    model

    Arguments:
    ----------
    - 'x': An array containing x_0, and x_1 the inputs

    Return:
    -------
    - The predicted class
    """
	if(x[0]**2 < 2 * x[1]**2):
		return -1
	return 1

if __name__ == "__main__":
	SIZE = 2000
	
	# Generate the dataset and compute predictions
	X, y = generate_dataset(SIZE)
	yPred = [predict(item) for item in X]

	# Compute number of wrong predictions
	wrong_predict = 0
	for i in range(SIZE):
		if y[i] != yPred[i]:
			wrong_predict += 1

	# Compute proportion of wrong predictions
	error = wrong_predict/SIZE

	print("error = {} \n".format(error))