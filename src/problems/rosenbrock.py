
# EPS (Extended Powell Singular) Moré Function

import numpy as np
import math

name = "rosenbrock"
n = 1000

l = -0.9
u = 0.9

def save_solution(k, x, J, d, alfa, metrics):
	pass

def x_init():
	return np.random.rand(n)


## f value and gradient

def f(x):
	result = 0
	for i in range(n-1):
		result += 100*pow(x[i+1]-x[i]*x[i], 2) + pow(1-x[i], 2)
	return result

def f_and_grad(x):
	f_value = f(x)

	f_grad = np.zeros(n)
	for i in range(n-1):
		f_grad[i] = -400*x[i]*(x[i+1]-x[i]*x[i])-2*(1-x[i])
	#for i in range(1,n):
	#	f_grad[i] += 200*(x[i]-x[i-1]*x[i-1])
		
	return f_value, f_grad


## f grad2 (hessiana)


def f_grad2(x):
	H_diag = np.zeros(n)
	for i in range(n-1):
		H_diag[i] = 1200*x[i]*x[i] -400*x[i+1] +2
	#H_diag[n-1] = 200
	return H_diag
	

## P, value and grad

def P(x):
	"""
	Calculate the L1 regularization.
	"""
	if np.min(x) < l or np.max(x) > u:
		return 999999999
	else:
		return 0

def P_grad(x):
	pass


## direction

def calculate_direction_all(x, grad_f_x, H_diag, c):
	'''
	Calcula direccion d de forma exacta.
	Se realiza acá porque depende de la función f, P y sus derivadas.
	'''

	# Calculate direction at P norm1 as the paper
	d = np.zeros(n)
	for i in range(n):
		abc = [ l-x[i], -grad_f_x[i] / H_diag[i], u-x[i] ]
		d[i] = sorted(abc)[1] # Mid point
	return d


