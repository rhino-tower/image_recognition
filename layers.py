from neuralnet import sigmoid, softmax
import numpy as np
from act_func_lib import act_func
import loss_func

class AddLayer():
	def __init__(self):
		pass

	def forward(self, x, y):
		return x + y

	def backward(self, dout):
		dx = 1 * dout
		dy = 1 * dout
		return dx, dy

class MulLayer():
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		self.x = x
		self.y = y

		return x * y

	def backward(self, dout):
		dx = dout * self.y
		dy = self.x * dout

		return dx, dy

class ReLULayer():
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx

class SigmoidLayer():
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = sigmoid(x)
		self.out = out

		return out

	def backward(self, dout):
		dx = dout * self.out * (1 - self.out)

		return dx

class AffineLayer():
	def __init__(self):
		self.X = None
		self.W = None
		self.B = None

	def forward(self, X, W, B):
		self.X = X
		self.W = W
		self.B = B
		out = np.dot(X, W) + B

		return out

	def backward(self, dout):
		dX = np.dot(dout, self.W.T)
		dW = np.dot(self.X.T, dout)
		dB = np.sum(dout, axis=0)

		return dX

class SoftmaxWithLossLayer():
	def __init__(self):
		self.t = None
		self.soft = None
		self.loss = None

	def forward(self, x, t):
		self.t = t
		self.soft = softmax(x)
		self.loss = loss_func.cross_entropy_error(self.soft, t)
		out = self.loss

		return out

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = dout * (self.soft - self.t) / batch_size

		return dx
