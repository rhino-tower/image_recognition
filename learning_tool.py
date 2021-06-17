from neuralnet import sigmoid
import numpy as np

def numerical_gradient(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)

	for idx in range(x.shape[0]):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxh1 = f(x) # f(x+h)

		x[idx] = tmp_val - h
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)

		x[idx] = tmp_val # 値を元に戻す

	return grad


def gradient_descent(f, init_x, eta=0.01, step=100):
	x = init_x

	for i in range(step):
		grad = numerical_gradient(f, x)
		x -= eta * grad

	return x

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
