#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

def generate_x(y):
	if y == -1:
		mean = np.array([0, 0])
		cov = np.array([[1, 0], [0, 1]])
	else:
		mean = np.array([0, 0])
		cov = np.array([[2, 0], [0, 1/2]])

	x = np.random.multivariate_normal(mean, cov, size = 1)
	return x[0]

def generate_dataset(size):
	classes = [-1, 1]
	y = random.choices(classes, k = size)
	X = [generate_x(item) for item in y]

	return X, y

def predict(x):
	if(x[0]**2 < 2 * x[1]**2):
		return -1
	return 1

if __name__ == "__main__":
	SIZE = 2000
	
	X, y = generate_dataset(SIZE)
	yPred = [predict(item) for item in X]

	wrong_predict = 0
	for i in range(SIZE):
		if y[i] != yPred[i]:
			wrong_predict += 1

	error = wrong_predict/SIZE

	#print("y = {} \n".format(y))
	#print("yPred = {} \n".format(yPred))
	print("error = {} \n".format(error))