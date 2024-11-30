
# TRIG (Trigonometric) Moré Function

import numpy as np
import math

name = "more_trig"
c = 10
n = 1000

def save_solution(k, x, J, d, alfa, metrics):
	pass

def x_init():
	return np.ones(n) * 1/n

def f(x):
	g_values = _g_values(x)
	return np.sum(g_values * g_values )



def _g_values(x):
	f_values = np.zeros(n)
	sum_cos = np.sum(np.cos(x))
	for i in range(n):
		f_values[i] = n - sum_cos + i*(1 - np.cos(x[i])) - np.sin(x[i]) # 1
	return f_values

def _g_grad(i, k, x):
	if i == k:
		return math.sin(x[i]) - math.cos(x[i]) + i*math.sin(x[i])
	else:
		return math.sin(x[k])

def _g_grad2(i, k, x):
	if i == k:
		return math.cos(x[i]) + math.sin(x[i]) + i*math.cos(x[i])
	else:
		return math.cos(x[k])

def f_and_grad(x):
	g_values = _g_values(x)
	f_value = np.sum(g_values * g_values )

	f_grad = np.zeros(n)
	for k in range(n):
		f_grad[k] = 0
		for i in range(n):
			f_grad[k] += 2 * g_values[i] * _g_grad(i, k, x) # 2

	return f_value, f_grad

def f_grad2(x):
	g_values = _g_values(x)
	H_diag = np.zeros(n)
	for k in range(n):
		for i in range(n):
			H_diag[k] += 2 * pow(_g_grad(i, k, x), 2) + 2 * g_values[i] * _g_grad2(i, k, x)  # 3
	return H_diag


def P(x):
	"""
	Calculate the L1 regularization.
	"""
	return np.sum(np.abs(x))

def P_grad(x):
	"""
	Calculate the gradient of the L1 regularization.
	"""
	threshold = 1e-13
	#grad = np.sign(x)
	grad = np.where(np.abs(x) < threshold, 0, np.where(x > 0, 1, -1))
	return grad


def calculate_direction_all(x, grad_f_x, H_diag):
	'''
	Calcula direccion d de forma exacta.
	Se realiza acá porque depende de la función f, P y sus derivadas.
	'''

	# Calculate direction at P norm1 as the paper
	d = np.zeros(n)
	for j in range(n):
		abc = [ (grad_f_x[j] - c) / H_diag[j], x[j], (grad_f_x[j] + c) / H_diag[j] ]
		d[j] = - sorted(abc)[1] # Mid point
	return d

