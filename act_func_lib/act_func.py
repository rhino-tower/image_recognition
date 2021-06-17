import numpy as np
#活性化関数ライブラリ

# ステップ関数
def step_func(x):
	y = x > 0
	return y.astype(np.int)

# シグモイド関数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def ReLU(x):
	return np.maximum(0, x)

def identity_func(x):
	return x

def softmax(a):
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
	y = exp_a / sum_exp_a

	return y
