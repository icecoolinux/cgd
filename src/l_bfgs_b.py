import time
import numpy as np
from scipy.optimize import minimize

def l_bfgs_b(problem):
    def count_nonzeros(x):
        threshold = 1e-7
        return np.count_nonzero(np.abs(x) > threshold)

    # Definir la función objetivo reformulada
    def objective(params):
        n = len(params) // 2
        y = params[:n]  # Primeras n variables son y
        z = params[n:]  # Últimas n variables son z
        
        # f(y - z) + c * sum(y + z)
        return problem.f(y - z) + problem.c * np.sum(y + z)

    # Initial guess
    x0 = problem.x_init()
    n = len(x0)
    y0 = np.maximum(x0, 0)
    z0 = np.maximum(-x0, 0)

    # Unir y0 y z0
    initial_params = np.concatenate([y0, z0])

    # Restricciones: y >= 0, z >= 0
    bounds = [(0, None)] * (2 * n)

    ts1 = time.perf_counter_ns() # Time 1

    # Optimización
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

    # Recuperar y y z
    y_opt = result.x[:n]
    z_opt = result.x[n:]
    x_opt = y_opt - z_opt

    ts2 = time.perf_counter_ns() # Time 2

    return result.x, result.fun, count_nonzeros(result.x), (ts2-ts1)/1000000000

