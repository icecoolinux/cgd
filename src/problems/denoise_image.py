
import numpy as np
import math
from PIL import Image
from skimage.util import random_noise

name = "denoise_image"
c = 100

def read_image(image_path):
	image = Image.open(image_path)
	image_array = np.array(image)
	image_array = image_array.astype(np.float32)
	image_array = image_array / 255
	return image_array, image.size[0], image.size[1]

def crop(array, width, height):
	crop_size = 10  # Tamaño del recorte
	center_x, center_y = width // 2, height // 2
	#center_x, center_y = crop_size // 2, crop_size // 2
	start_x = max(center_x - crop_size // 2, 0)
	start_y = max(center_y - crop_size // 2, 0)
	end_x = min(center_x + crop_size // 2, width)
	end_y = min(center_y + crop_size // 2, height)
	cropped_array = array[start_y:end_y, start_x:end_x]
	return cropped_array, cropped_array.shape[0], cropped_array.shape[1]
	#cropped_image = Image.fromarray(cropped_array)

def get_noisy_image(image_array):
	image_array = random_noise(image_array, mode="s&p", salt_vs_pepper=0.1)

	return image_array

def save_image(array, filename):
	base_img = array * 255
	img = Image.fromarray(base_img.astype(np.uint8))
	img.save(filename)

array, width, height = read_image("./data/cups.jpg")
#array, width, height = crop(array, width, height)
save_image(array, f"./original_cropped.png")
noisy_image = get_noisy_image(array)

#flat_noisy_image = noisy_image.flatten()
R = noisy_image[:, :, 0].flatten()
G = noisy_image[:, :, 1].flatten()
B = noisy_image[:, :, 2].flatten()
flat_noisy_image = np.concatenate([R, G, B])


def save_solution(k, x, J, d, alfa, metrics):
	size_per_channel = width * height
	img_sol = x * 256
	R = img_sol[:size_per_channel].reshape((height, width))
	G = img_sol[size_per_channel:2 * size_per_channel].reshape((height, width))
	B = img_sol[2 * size_per_channel:].reshape((height, width))
	image_array = np.stack([R, G, B], axis=2)
	image_array = image_array.astype(np.uint8)
	image_array = np.uint8(image_array)
	image = Image.fromarray(image_array)
	image.save(f"./solution_{k:06}.png")

	'''
	# Temp
	grad_P_x = P_grad(x)
	_, f_grad = f_and_grad(x)

	f = open(f"./solution_{k:06}.txt", "w")
	for index,a in enumerate(x):
		f.write(f"{index+1}: {a:.4f}, dir {d[index]:.4f}, gradF {f_grad[index]:.4f}, gradP {grad_P_x[index]:.4f}, alfa {alfa} \n")
	f.close()
	'''

# Save noisy image
save_image(noisy_image, f"./noisy.png")

def x_init():
	return np.copy(flat_noisy_image)
	#return np.random.rand(len(flat_noisy_image))

def f(x):
	'''
	Calculate the squared euclidean distance between the noisy image
	and x the generated image.
	'''
	# Calculate f value
	diff = x - flat_noisy_image
	squared_euclidean_dist = diff @ diff
	f_value = squared_euclidean_dist / 2

	return f_value

def f_and_grad(x):
	'''
	Calculate the squared euclidean distance between the noisy image
	and x the generated image.
	'''
	# Calculate f value
	diff = x - flat_noisy_image
	squared_euclidean_dist = diff @ diff
	f_value = squared_euclidean_dist / 2

	# Calculate grad
	f_grad = x - flat_noisy_image

	return f_value, f_grad

def f_grad2(x):
	'''
	Here the hessian is the identity.
	Only the diagonal.
	'''
	H_diag = np.ones(len(x))
	return H_diag

def P(y):
	'''
	Usamos Norm 2 al cuadrado
	Norma 1 no es diferenciable en todos los puntos.
	'''
	'''
	error = 0
	# Para cada fila
	for i in range(1, height):
		# Para cada columna
		for j in range(1, width):
			# Para cada componente de color
			for k in range(3):
				# Primero están todos los componentes R, en el medio G y al final B.
				pos = k*width*height + i*width + j
				pixel = x[pos]
				pixel_left = x[pos-1]
				pixel_top = x[pos - width]
				#error += math.sqrt( pow(pixel - pixel_left, 2) ) + math.sqrt( pow(pixel - pixel_top, 2) )
				error +=  ( pow(pixel - pixel_left, 2) + pow(pixel - pixel_top, 2) ) / 2
				#error += abs(pixel - pixel_left)
	return error
	'''
	# Initialize the sum
	total = 0

	# Shape of the input grid
	rows = height
	cols = width

	# Iterate through the grid
	for i in range(rows):
		for j in range(cols):
			for k in range(3):
				pos = k * width * height + i * width + j

				# Contribution from row neighbors
				if i < rows - 1:  # y[i+1, j] - y[i, j]
					total += abs(y[pos + width] - y[pos])

				# Contribution from column neighbors
				if j < cols - 1:  # y[i, j+1] - y[i, j]
					total += abs(y[pos+1] - y[pos])

	return total


def P_grad(y):
	'''
	gradient = np.zeros( len(x) )
	epsilon = 1e-10  # Para evitar divisiones por cero

	# Para cada fila
	for i in range(1, height):
		# Para cada columna
		for j in range(1, width):
			# Para cada componente de color (R, G, B)
			for k in range(3):
				pos = k * width * height + i * width + j
				pixel = x[pos]
				pixel_left = x[pos - 1]
				pixel_top = x[pos - width]

				# Gradiente respecto al vecino izquierdo
				diff_left = pixel - pixel_left
				#diff_left = diff_left / (math.sqrt(diff_left ** 2) + epsilon)

				# Gradiente respecto al vecino superior
				diff_top = pixel - pixel_top
				#diff_top = diff_top / (math.sqrt(diff_top ** 2) + epsilon)

				# Acumular gradiente en el píxel actual
				gradient[pos] += diff_left + diff_top

				# Propagar gradiente al vecino superior
				#gradient[pos - 1] -= diff_left

				# Propagar gradiente al vecino superior
				#gradient[pos - width] -= diff_top

	return gradient
	'''
	"""
	Compute the gradient of the expression:
	sum_{i,j} |y[i+1,j] - y[i,j]| + |y[i,j+1] - y[i,j]|

	Args:
	    y (2D numpy array): A 2D grid of values y[i, j]

	Returns:
	    gradient (2D numpy array): Gradient values for each element in y
	"""
	# Initialize gradient array with zeros
	gradient = np.zeros_like(y)

	# Shape of the input grid
	rows = height
	cols = width

	# Compute gradients
	for i in range(rows):
		for j in range(cols):
			for k in range(3):
				pos = k * width * height + i * width + j

				# Contributions from row neighbors
				if i < rows - 1:  # y[i+1, j] - y[i, j]
					diff = y[pos + width] - y[pos]
					gradient[pos] -= 1 if diff > 0 else -1 if diff < 0 else 0
				'''
				if i > 0:  # y[i, j] - y[i-1, j]
					diff = y[pos] - y[pos - width]
					gradient[pos] += 1 if diff > 0 else -1 if diff < 0 else 0
				'''
				# Contributions from column neighbors
				if j < cols - 1:  # y[i, j+1] - y[i, j]
					diff = y[pos + 1] - y[pos]
					gradient[pos] -= 1 if diff > 0 else -1 if diff < 0 else 0
				'''
				if j > 0:  # y[i, j] - y[i, j-1]
					diff = y[pos] - y[pos -1]
					gradient[pos] += 1 if diff > 0 else -1 if diff < 0 else 0
				'''
	return gradient


def calculate_direction_all(x, grad_f_x, H_diag):
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
	P_grad_x = P_grad(x)
	for j in range(n):
		abc = [ (grad_f_x[j] - c*P_grad_x[j]) / H_diag[j], x[j], (grad_f_x[j] + c*P_grad_x[j]) / H_diag[j] ]
		d[j] = - sorted(abc)[1] # Mid point
	return d

