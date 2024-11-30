
import numpy as np

name = "easy"
c = 0.01

def save_solution(k, x, J, d, alfa, metrics):
	pass

def x_init():
	return np.array([43, 2])

def f(x):
	f_value = (x-1) @ (x-1)
	return f_value

def f_and_grad(x):
	f_value = (x-1) @ (x-1)
	grad = 2 * x -2
	return f_value, grad

def f_grad2(x):
	H_diag = np.ones(len(x)) * 2
	return H_diag

def P(x):
	return np.sum(np.abs(x))

def P_grad(x):
	threshold = 1e-7
	grad = np.where(np.abs(x) < threshold, 0, np.where(x > 0, 1, -1))
	return grad

# Calcula direccion d de forma exacta.
# Se realiza acá porque depende de la función f, P y sus derivadas.
def calculate_direction_all(x, grad_f_x, H_diag):
	# Calculate direction at P norm1 as the paper
	n = len(x)
	d = np.zeros(n)
	for j in range(n):
		abc = [ (grad_f_x[j] - c) / H_diag[j], 
				x[j], 
				(grad_f_x[j] + c) / H_diag[j] ]
		d[j] = - sorted(abc)[1] # Mid point
	return d

