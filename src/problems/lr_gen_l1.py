
# Cambios realizados:
# la etiqueta es 1 si cancer, -1 si no, esto porque veo que le da lo mismo estar
# de un lado u otro del 0 con lo que es natural colocar simetrico los labels y
# que 0 sea el punto de corte.
# Queda oscilando, será alfa muy grande?

from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import math

name = "lr_gen_l1"

# Cargo matriz de muestras genéticas de cancer.
path = "./data/data.h5"

db = h5py.File(path, mode = 'r')
X = db["RNASeq"][...]
y = db["label"][...]

# Standarizo X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Clasificacion binaria de un solo tipo de cancer.
y_binary_10 = np.where(y == 10, 1, -1)
y = y_binary_10

m, n = X.shape
X_t = X.T

# Precalculo la diagonal de la hessiana ya que es fijo.
hessian_diag = (1 / m) * np.sum(X**2, axis=0)

diseasedict = {
	'skin cutaneous melanoma':0, 'thyroid carcinoma':1, 'sarcoma':2,
	'prostate adenocarcinoma':3, 'pheochromocytoma & paraganglioma':4,
	'pancreatic adenocarcinoma':5, 'head & neck squamous cell carcinoma':6,
	'esophageal carcinoma':7, 'colon adenocarcinoma':8,
	'cervical & endocervical cancer':9, 'breast invasive carcinoma':10,
	'bladder urothelial carcinoma':11, 'testicular germ cell tumor':12,
	'kidney papillary cell carcinoma':13, 'kidney clear cell carcinoma':14,
	'acute myeloid leukemia':15, 'rectum adenocarcinoma':16,
	'ovarian serous cystadenocarcinoma':17, 'lung adenocarcinoma':18,
	'liver hepatocellular carcinoma':19,
	'uterine corpus endometrioid carcinoma':20, 'glioblastoma multiforme':21,
	'brain lower grade glioma':22, 'uterine carcinosarcoma':23, 'thymoma':24,
	'stomach adenocarcinoma':25, 'diffuse large B-cell lymphoma':26,
	'lung squamous cell carcinoma':27, 'mesothelioma':28,
	'kidney chromophobe':29, 'uveal melanoma':30, 'cholangiocarcinoma':31,
	'adrenocortical cancer':32
}

def save_solution(k, x, J, d, alfa, metrics):
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	pred = X @ x
	for i in range(m):
		if y[i] == 1:
			if pred[i] > 0:
				true_positive += 1
			else:
				false_negative += 1
		else:
			if pred[i] > 0:
				false_positive += 1
			else:
				true_negative += 1
	metrics['accuracy'] = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
	print(f"Accuracy: {metrics['accuracy']:.2f}")
	if (true_positive + false_positive) == 0:
		metrics['precision'] = None
	else:
		metrics['precision'] = true_positive / (true_positive + false_positive)
		print(f"Precision: {metrics['precision']:.2f}")
	if (true_positive + false_negative) == 0:
		metrics['recall'] = None
	else:
		metrics['recall'] = true_positive / (true_positive + false_negative)
		print(f"Recall: {metrics['recall']:.2f}")

def x_init():
	return np.random.rand(n) * 1e-6

def f(x):
	'''
	Mean squared error.
	'''
	# Calulcate f value
	residual = X @ x - y
	f_value = np.sum( residual ** 2 ) / 2 * m
	return f_value

def f_and_grad(x):
	'''
	Mean squared error.
	'''
	# Calulcate f value
	residual = X @ x - y
	#f_value =  pow( np.linalg.norm(residual), 2) / 2 * m
	f_value = np.sum( residual ** 2 ) / 2 * m

	# Calculate gradient
	grad = (X_t @ residual) / m

	return f_value, grad

def f_grad2(x):
	'''
	Here the hessian is the identity.
	Only the diagonal.
	'''
	H_diag = np.zeros(n)
	for i in range(n):
		H_diag[i] = hessian_diag[i]
	return H_diag

# Hubber loss
delta=0.1

def P(x):
	'''
	Usamos Norm 2 al cuadrado
	Norma 1 no es diferenciable en todos los puntos.
	'''
	'''
	return pow( np.linalg.norm(x, ord=2), 2) / 2
	'''

	# L1
	return np.sum(np.abs(x))
	'''
	# Huber Loss (Pseudo-Hubber Loss)
	return np.sum( delta*delta*(np.sqrt(1+np.power(x/delta, 2)) -1) )
	'''

def P_grad(x):
	'''
	return x
	'''

	# L1

	threshold = 1e-13
	#grad = np.sign(x)
	grad = np.where(np.abs(x) < threshold, 0, np.where(x > 0, 1, -1))
	return grad
	'''

	# Huber Loss (Pseudo-Hubber Loss)
	return x / np.sqrt( 1 + np.power(x/delta, 2) )
	'''

def calculate_direction_all(x, grad_f_x, H_diag, c):
	'''
	Calcula direccion d de forma exacta.
	Se realiza acá porque depende de la función f, P y sus derivadas.
	'''
	'''
	return - (grad_f_x + c * x) / (H_diag + c)
	'''


	# Calculate direction at P norm1 as the paper
	d = np.zeros(n)
	for j in range(n):
		abc = [ (grad_f_x[j] - c) / H_diag[j],
				x[j],
				(grad_f_x[j] + c) / H_diag[j] ]
		d[j] = - sorted(abc)[1] # Mid point
	return d
	'''
	d = np.zeros(n)
	grad_pseudo_huber = P_grad(x)
	for j in range(n):
		abc = [
			(grad_f_x[j] - c * grad_pseudo_huber[j]) / H_diag[j],
			x[j],
			(grad_f_x[j] + c * grad_pseudo_huber[j]) / H_diag[j]
		]
		d[j] = -sorted(abc)[1]  # Midpoint
	return d
	'''
