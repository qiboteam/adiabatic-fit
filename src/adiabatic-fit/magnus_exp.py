import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from evolution import generate_schedule

# load the best_p
params = np.load("./fit-results/easy-gamma/best_p.npy")

# Define symbolic variable t
t = sp.symbols('t')

nparams = len(params)

s_t = np.sum(  (t ** np.arange(1,1+nparams)*params) )
s_t /= np.sum(params)

# Define the Hamiltonian H(t)
H = sp.Matrix([[s_t, 1 - s_t], [1 - s_t, -s_t]])

# Compute the integral of each component of H(t) from 0 to t
H_integrated = H.applyfunc(lambda elem: sp.integrate(elem, (t, 0, t)))

# Display the result
print(H_integrated)

# Convert each element in the matrix to a lambda function for numerical evaluation
H_integrated_funcs = [[sp.lambdify(t, H_integrated[i, j], "numpy") for j in range(2)] for i in range(2)]

def integral(t_val):
    # Evaluate each component of the integrated Hamiltonian at t_val
    return np.array([[f(t_val) for f in row] for row in H_integrated_funcs])