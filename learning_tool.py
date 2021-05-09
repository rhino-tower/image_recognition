import numpy as np

def numetrical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		#xのidx番目で偏微分
		tmp = x[idx]

		#f(x+h)の計算
		x[idx] = tmp + h
		fxh1 = f(x)
		#f(x-h)の計算
		x[idx] = tmp - h
		fxh2 = f(x)

		grad = (fxh1 - fxh2) / (2*h)
		#xを元に戻す
		x[idx] = tmp

	return grad

def gradient_descent(f, init_x, eta=0.01, step=100):
	x = init_x

	for i in range(step):
		grad = numetrical_gradient(f, x)
		x -= eta * grad

	return x
