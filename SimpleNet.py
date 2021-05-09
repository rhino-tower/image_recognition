import sys, os
import numpy as np
from loss_func import cross_entropy_error
from neuralnet import softmax

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3)

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t, one_hot=False)

		return loss
