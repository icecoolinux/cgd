
# EPS (Extended Powell Singular) Moré Function

import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial

name = "more_eps"
n = 1000

def save_solution(k, x, J, d, alfa, metrics):
	pass

def x_init():
	x0 = np.zeros(n)
	for i in range(0, n, 4):
		x0[i] = 3
		x0[i+1] = -1
		x0[i+2] = 0
		x0[i+3] = 1
	return x0


## f value and gradient

def f_parallel(x):
	g_values = _g_values_parallel(x)
	return np.sum(g_values * g_values )

def f(x):
	g_values = _g_values(x)
	return np.sum(g_values * g_values )

def _g_values_chunk(start, end, x):
	chunk = np.zeros(end - start)
	for i in range(start, end, 4):
		chunk[i - start] = x[i] + 10 * x[i + 1]
		chunk[i + 1 - start] = math.sqrt(5) * (x[i + 2] - x[i + 3] - 1)
		chunk[i + 2 - start] = pow(x[i + 1] - 2 * x[i + 2], 2)
		chunk[i + 3 - start] = math.sqrt(10) * pow(x[i] - x[i + 3], 2)
	return chunk

def _g_values_parallel(x):
	g_values = np.zeros(n)

	with ProcessPoolExecutor() as executor:
		step = 4 #max(4, n // executor._max_workers)  # Step size per chunk
		tasks = [executor.submit(_g_values_chunk, i, min(i + step, n), x) for i in range(0, n, step)]
		results = [task.result() for task in tasks]

	g_values[:len(results) * step] = np.concatenate(results)
	return g_values

def _g_values(x):
	g_values = np.zeros(n)
	for i in range(0, n, 4):
		g_values[i]   = x[i] + 10*x[i+1]
		g_values[i+1] = math.sqrt(5) * (x[i+2] - x[i+3] -1)
		g_values[i+2] = pow(x[i+1] - 2*x[i+2], 2)
		g_values[i+3] = math.sqrt(10) * pow(x[i] - x[i+3], 2)
	return g_values

def _g_grad(i, k, x):
	if i//4 != k//4:
		return 0

	if i%4 == 0:
		if k%4 == 0:
			return 1
		elif k%4 == 1:
			return 10
	elif i%4 == 1:
		if k%4 == 2:
			return math.sqrt(5)
		elif k%4 == 3:
			return -math.sqrt(5)
	elif i%4 == 2:
		if k%4 == 1:
			return 2*x[k] - 4*x[k+1]
		elif k%4 == 2:
			return 4*x[k] - 4*x[k-1]
	elif i%4 == 3:
		if k%4 == 0:
			return math.sqrt(10)*4*x[k] - math.sqrt(10)*2*x[k+3]
		elif k%4 == 3:
			return -math.sqrt(10)*2*x[k-3] + math.sqrt(10)*2*x[k]
	return 0


def compute_grad_k(k, n, x, g_values):
	result = 0
	for i in range(n):
		result += 2 * g_values[i] * _g_grad(i, k, x)
	return result

def f_and_grad_parallel(x):
	g_values = _g_values_parallel(x)
	f_value = np.sum(g_values * g_values )

	# Grad
	f_grad = np.zeros(n)
	compute_partial = partial(compute_grad_k, n=n, x=x, g_values=g_values)
	with ProcessPoolExecutor() as executor:
		results = list(executor.map(compute_partial, range(n)))
	f_grad[:] = results
	
	return f_value, f_grad

def f_and_grad(x):
	g_values = _g_values(x)
	f_value = np.sum(g_values * g_values )

	f_grad = np.zeros(n)
	for k in range(n):
		f_grad[k] = 0
		for i in range(n):
			f_grad[k] += 2 * g_values[i] * _g_grad(i, k, x)

	return f_value, f_grad


## f grad2 (hessiana)

def _g_grad2(i, k, x):
	if i//4 != k//4:
		return 0

	if i%4 == 0:
		return 0
	elif i%4 == 1:
		return 0
	elif i%4 == 2:
		if k%4 == 1:
			return 2
		elif k%4 == 2:
			return 4
	elif i%4 == 3:
		if k%4 == 0:
			return math.sqrt(10)*4
		elif k%4 == 3:
			return math.sqrt(10)*2
	return 0

def compute_H_diag_k(k, n, x, g_values):
	result = 0
	for i in range(n):
		result += 2 * pow(_g_grad(i, k, x), 2) + 2 * g_values[i] * _g_grad2(i, k, x)
	return result
	
def f_grad2_parallel(x):
	g_values = _g_values_parallel(x)
	H_diag = np.zeros(n)

	compute_partial = partial(compute_H_diag_k, n=n, x=x, g_values=g_values)

	with ProcessPoolExecutor() as executor:
		results = list(executor.map(compute_partial, range(n)))

	H_diag[:] = results
	return H_diag
	
def f_grad2(x):
	g_values = _g_values(x)
	H_diag = np.zeros(n)
	for k in range(n):
		for i in range(n):
			H_diag[k] += 2 * pow(_g_grad(i, k, x), 2) + 2 * g_values[i] * _g_grad2(i, k, x)
	return H_diag
	

## P, value and grad

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


## direction

def calculate_direction_all(x, grad_f_x, H_diag, c):
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


