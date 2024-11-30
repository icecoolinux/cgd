
import time
import numpy as np

from src.metrics import process_metrics, show_step_metrics
 
def choice_coordinates(k, d_all, alfa_k, v_k):
	'''
	Se selecciona el conjunto J de coordenadas a ajustar.
	Retorna: J, v en k+1
	'''
	n = len(d_all)

	def gauss_seidel():
		'''
		En el paper recorre cada coordenada en cada iteración.
		'''
		J = [k % n]
		return J, None

	def gauss_southwell_r():
		# calculate v for k+1
		if alfa_k > 1e-3:
			v = max(1e-4, v_k/10)
		elif alfa_k < 1e-6:
			v = min(0.9, 50*v_k)
		else:
			v = v_k

		# calculate max direction component.
		d_max = np.max(np.abs(d_all))
		J = []
		for i in range(n):
			if np.abs(d_all[i]) >= v*d_max:
				J.append(i)
		return J, v

	def gauss_southwell_q():
		'''
		# calculate v for k+1
		if alfa_k > 1e-3:
			v = max(1e-4, v_k/10)
		elif alfa_k < 1e-6:
			v = min(0.9, 50*v_k)
		else:
			v = v_k

		# calculate max direction component.
		gradient
		d_max = np.max(np.abs(d_all))
		J = []
		for i in range(n):
			if np.abs(d_all[i]) >= v*d_max:
				J.append(i)
		return J, v
		'''
		pass

	#return gauss_seidel()
	return gauss_southwell_r()

def calculate_H(problem, x, use_parallel):
	'''
	Calcula matriz hessiana H de la aproximación f.
	En el paper se usa solo la diagonal de la hessiana, en esta
	implementación solo vamos a utilizar un vector que representa la diagonal.
	'''
	if use_parallel:
		H_diag = problem.f_grad2_parallel(x)
	else:
		H_diag = problem.f_grad2(x)
	for i in range(len(H_diag)):
		H_diag[i] = min(max(H_diag[i], 1e-2), 1e9) # Like the paper
	return H_diag

def calulate_direction(J, d_all):
	'''
	Calcula la direccion d de descenso.
	Considero H diagonal, según el paper es un buen balance de performance.
	Además, con H diagonal tenemos que d tiene solución cerrada.
	'''
	d = np.zeros(len(d_all))
	for i in J:
		d[i] = d_all[i]
	return d

def calculate_alpha(problem, x, F_x, grad_f_x, d, H_diag, alfa_init_k, armijo, use_parallel):
	'''
	Calcula tamaño del paso alfa.
	Utiliza Armijo, los hiperparámetros utilizados son los del paper.
	Retorna alfa y H @ d, este último es utilizado para la condición de parada.
	'''
	
	c = problem.c
	
	# Solo hago el calculo con la diagonal, es una mejora importante propuesta en el paper
	#Hd = H @ d
	Hd = H_diag * d
	delta = grad_f_x @ d + (armijo['gamma'] * d @ Hd) + c*problem.P(x + d) - c*problem.P(x)
	j = 0
	while True:
		alfa = alfa_init_k * pow(armijo['beta'], j)
		xd = x + alfa * d
		if use_parallel:
			F_xd = problem.f_parallel(xd) + c * problem.P(xd)
		else:
			F_xd = problem.f(xd) + c * problem.P(xd)
		if F_xd <= F_x + alfa * armijo['sigma'] * delta:
			return alfa, Hd

		# The paper uses this stop condition.
		if alfa < armijo['stop_alfa']:
			return alfa, Hd

		j += 1

def count_nonzeros(x):
	threshold = 1e-7
	return np.count_nonzero(np.abs(x) > threshold)


def step(k, x, problem, armijo, alfa, v_gauss_southwell, use_parallel, ts_metrics):

	c = problem.c

	ts1 = time.perf_counter_ns() # Time 1
	if use_parallel:
		f_value_x, grad_f_x = problem.f_and_grad_parallel(x)
	else:
		f_value_x, grad_f_x = problem.f_and_grad(x)
	f_grad_norm = np.linalg.norm(grad_f_x)
	ts2 = time.perf_counter_ns() # Time 2
	P_value = problem.P(x)
	ts3 = time.perf_counter_ns() # Time 3
	F_x = f_value_x + c * P_value

	ts4 = time.perf_counter_ns() # Time 4

	H_diag = calculate_H(problem, x, use_parallel)
	ts5 = time.perf_counter_ns() # Time 5

	d_all = problem.calculate_direction_all(x, grad_f_x, H_diag)
	ts6 = time.perf_counter_ns() # Time 6

	J, v_gauss_southwell = choice_coordinates(k, d_all, alfa, v_gauss_southwell)
	ts7 = time.perf_counter_ns() # Time 7

	d = calulate_direction(J, d_all)
	ts8 = time.perf_counter_ns() # Time 8

	alfa, Hd = calculate_alpha(problem, x, F_x, grad_f_x, d, H_diag, alfa, armijo, use_parallel)
	ts9 = time.perf_counter_ns() # Time 9


	# Stop conditions used in the paper
	stop_loop = False
	if alfa < armijo['stop_alfa']:
		stop_loop = True
	if np.max(np.abs(Hd)) <= 1e-4:
		stop_loop = True

	# Update x
	x = x + alfa*d

	# Update next alfa, used by the paper
	alfa = min(alfa/armijo['beta'], 1)


	# Process time metrics and step metrics
	nz = count_nonzeros(x)
	lenJ = len(J)
	step_metrics = process_metrics(f_value_x, P_value, f_grad_norm, nz, alfa, lenJ, ts_metrics, ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9)
	show_step_metrics(x, k, f_value_x, f_grad_norm, lenJ, nz)

	problem.save_solution(k, x, J, d, alfa, step_metrics)
	
	return x, alfa, v_gauss_southwell, stop_loop, step_metrics


	
