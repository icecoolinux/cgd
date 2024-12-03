
# CUTEr Functions

'''
https://jfowkes.github.io/pycutest/_build/html/install.html
https://github.com/jfowkes/pycutest


apt install python3-dev
apt install gfortran gcc

mkdir cutest
cd cutest
git clone https://github.com/ralna/ARCHDefs ./archdefs
git clone https://github.com/ralna/SIFDecode ./sifdecode
git clone https://github.com/ralna/CUTEst ./cutest
git clone https://bitbucket.org/optrove/sif ./mastsif

# Se agrega a .bashrc
export ARCHDEFS=/home/icecool/Descargas/tao/cutest/archdefs
export SIFDECODE=/home/icecool/Descargas/tao/cutest/sifdecode
export MASTSIF=/home/icecool/Descargas/tao/cutest/mastsif
export CUTEST=/home/icecool/Descargas/tao/cutest/cutest
export MYARCH="pc64.lnx.gfo"

ls ~/.bashrc # load above environment variables
source ~/.bashrc # load above environment variables

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"

# Para testear
cd $SIFDECODE/src ; make -f $SIFDECODE/makefiles/$MYARCH test
cd $CUTEST/src ; make -f $CUTEST/makefiles/$MYARCH test


pip install pycutest
'''


import pycutest
import numpy as np
import math

name = "cuter_eg2"

problem_name = 'EG2'
#problem_name = 'DIXON3DQ'

#pycutest.print_available_sif_params(problem_name)
#pycutest.problem_properties(problem_name)

#p = pycutest.import_problem(problem_name, sifParams={'N':1000})
p = pycutest.import_problem(problem_name)



def save_solution(k, x, J, d, alfa, metrics):
	pass

def x_init():
	return np.random.rand(p.n)

def f(x):
	return p.obj(x)

def f_and_grad(x):
	f_value = p.obj(x)
	f_grad = p.grad(x)
	return f_value, f_grad

def f_grad2(x):
	hessian = p.sphess(x)
	#hessian = p.hess(x)
	return hessian.diagonal()

def P(x):
	"""
	Calculate the L1 regularization.
	"""
	return np.sum(np.abs(x))

def P_grad(x):
	"""
	Calculate the gradient of the L1 regularization.
	"""
	threshold = 1e-7
	#grad = np.sign(x)
	grad = np.where(np.abs(x) < threshold, 0, np.where(x > 0, 1, -1))
	return grad


def calculate_direction_all(x, grad_f_x, H_diag, c):
	'''
	Calcula direccion d de forma exacta.
	Se realiza acá porque depende de la función f, P y sus derivadas.
	'''
	'''
	grad_P_x = P_grad(x)
	return - (grad_f_x + c*grad_P_x) / H_diag
	'''
	# Calculate direction at P norm1 as the paper
	n = len(x)
	d = np.zeros(n)
	for j in range(n):
		abc = [ (grad_f_x[j] - c) / H_diag[j],
				x[j],
				(grad_f_x[j] + c) / H_diag[j] ]
		d[j] = - sorted(abc)[1] # Mid point
	return d

