
import matplotlib.pyplot as plt
import numpy as np
import os

def init_metrics():
	hist = []
	ts_metrics = {}
	ts_metrics['metrics'] = 0
	ts_metrics['f_and_grad'] = 0
	ts_metrics['P_value'] = 0
	ts_metrics['H'] = 0
	ts_metrics['direction_all'] = 0
	ts_metrics['J'] = 0
	ts_metrics['direction'] = 0
	ts_metrics['alfa'] = 0
	return hist, ts_metrics

def process_metrics(f_value_x, P_value, f_grad_norm, nz, alfa, lenJ, ts_metrics, ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9):
	ts_metrics['f_and_grad'] += ts2 - ts1
	ts_metrics['P_value'] += ts3 - ts2
	ts_metrics['metrics'] += ts4 - ts3
	ts_metrics['H'] += ts5 - ts4
	ts_metrics['direction_all'] += ts6 - ts5
	ts_metrics['J'] += ts7 - ts6
	ts_metrics['direction'] += ts8 - ts7
	ts_metrics['alfa'] += ts9 - ts8
	
	step_metric = {'f_value': f_value_x, 
					'P_value': P_value, 
					'f_grad': f_grad_norm, 
					'nonzeros': nz, 
					'alfa': alfa,
					'lenJ': lenJ}
	
	return step_metric

def show_step_metrics(x, k, f_value_x, f_grad_norm, lenJ, nz):
	print("")
	print("Iteration "+str(k))
	print(f"f value:, {f_value_x:.2f}")
	print(f"f grad:, {f_grad_norm:.2f}")
	print("Lenght J: "+str(lenJ)+" / "+str(len(x)))
	print("NZ: "+str(nz)+" / "+str(len(x)))

def tiempo(filename, ts_metrics):
	# Segundos de cada etapa
	
	total = 0
	etapas = []
	segundos = []

	for m in ts_metrics:
		etapas.append(m)
		seconds = ts_metrics[m]/1000000000
		segundos.append(seconds)
		total += seconds

	print(f"Total seconds: {total:.3f}")
	
	plt.figure(figsize=(10, 6))
	plt.bar(etapas, segundos)

	plt.title("Segundos por etapa")
	plt.xlabel("Etapas")
	plt.ylabel("Segundos")
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()
	

def f_grad(filename, hist):
	# Valor de gradiente de f durante el tiempo
	
	f_grad = [ h['f_grad'] for h in hist ]

	plt.figure(figsize=(10, 8))
	plt.semilogy(f_grad)
	#plt.plot(f_grad)

	plt.title("Valor del gradiente de la función f en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("Gradiente")
	plt.grid()
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def f_value(filename, hist):
	# Valor de f durante el tiempo

	f_value = [ h['f_value'] for h in hist ]

	plt.figure(figsize=(10, 8))
	#plt.semilogy(f_value)
	plt.plot(f_value)

	plt.title("Valor de función f en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("Valor")
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def P_value(filename, hist):
	# Valor de P durante el tiempo

	P_value = [ h['P_value'] for h in hist ]

	plt.figure(figsize=(10, 6))
	plt.plot(P_value)

	plt.title("Valor de función P en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("Valor")
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def nonzeros(filename, hist):
	# Cantidad de valores no ceros

	nonzeros = [ h['nonzeros'] for h in hist ]

	plt.figure(figsize=(10, 6))
	plt.plot(nonzeros)

	plt.title("Cantidad de valores no ceros en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("Valores no ceros")
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def alfa(filename, hist):
	# ALFA

	alfas = [ h['alfa'] for h in hist ]

	plt.figure(figsize=(10, 8))
	plt.semilogy(alfas)
	#plt.plot(alfas)

	plt.title("Valor ALFA en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("ALFA")
	plt.grid()
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def validation(filename, hist):
	# Precision, accuracy y recall

	if 'precision' not in hist[0] or 'accuracy' not in hist[0] or 'recall' not in hist[0]:
		return
		
	precision = [ h['precision'] for h in hist ]
	accuracy = [ h['accuracy'] for h in hist ]
	recall = [ h['recall'] for h in hist ]

	plt.figure(figsize=(10, 6))
	plt.plot(precision, label="Precision")
	plt.plot(accuracy, label="Accuracy")
	plt.plot(recall, label="Recall")

	plt.title("Valor de función P en cada iteración")
	plt.xlabel("Iteración")
	plt.ylabel("Valor")
	plt.legend()
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def values_nonzeros(filename, x):
	# Muestro los valores ordenados más altos.
	# Para clasificación de cancer es útil para conocer qué genes son los claves
	# para realizar un diagnóstico.

	amount = 100

	largest_values = np.sort(x)[-amount:]
	indices = np.argsort(x)[-amount:]

	# Plot the bar chart
	plt.figure(figsize=(10, 6))  # Adjust the size of the figure
	plt.bar(range(amount), largest_values, color='skyblue')

	# Show the plot
	plt.title("Valores de x más grandes")
	plt.xlabel("X")
	plt.ylabel("Valores x más grande")
	plt.tight_layout()
	plt.grid()
	
	plt.savefig(filename, format='png', dpi=300)
	plt.show()

def generate_metrics(dir_name, x, hist, ts_metrics):
	os.makedirs(dir_name, exist_ok=True)

	tiempo(f"{dir_name}/time.png", ts_metrics)
	f_grad(f"{dir_name}/f_grad.png", hist)
	f_value(f"{dir_name}/f_value.png", hist)
	P_value(f"{dir_name}/P_value.png", hist)
	nonzeros(f"{dir_name}/nz.png", hist)
	alfa(f"{dir_name}/alfa.png", hist)
	validation(f"{dir_name}/validation.png", hist)
	values_nonzeros(f"{dir_name}/values_nz.png", x)
	
	
	
	
	
	
