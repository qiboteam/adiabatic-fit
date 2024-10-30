import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Load parameters and define symbolic variable t
params = np.load("./fit-results/easy-gamma/best_p.npy")
t = sp.symbols('t')
nparams = len(params)

# Define s(t) and Hamiltonian H(t)
s_t = np.sum((t ** np.arange(1, 1 + nparams) * params))
s_t /= np.sum(params)
H = sp.Matrix([[s_t, 1 - s_t], [1 - s_t, -s_t]])

# Define time interval and timestep for integration up to t=0.4
t_min = 0  # Start time
t_target = 0.4  # Target time for evaluation
timestep = 0.002
time_points = np.arange(t_min, t_target, timestep)

# Convert H(t) elements to functions for numerical evaluation
H_funcs = [[sp.lambdify(t, H[i, j], "numpy") for j in range(2)] for i in range(2)]

# Function to evaluate H(t) numerically at a specific time
def H_numeric(t_val):
    return np.array([[f(t_val) for f in row] for row in H_funcs])

# Calculate Omega_1 by summing H(t) over time points
Omega_1 = np.zeros((2, 2), dtype=complex)
for t_val in time_points:
    Omega_1 += H_numeric(t_val) * timestep
Omega_1 *= -1j  # Multiply by -i to complete Omega_1

# Calculate Omega_2 by summing commutators of H(t) at pairs of time points
Omega_2 = np.zeros((2, 2), dtype=complex)
for i, t_i in enumerate(time_points):
    for j, t_j in enumerate(time_points[:i]):
        H_ti = H_numeric(t_i)
        H_tj = H_numeric(t_j)
        commutator = np.dot(H_ti, H_tj) - np.dot(H_tj, H_ti)  # [H(t_i), H(t_j)]
        Omega_2 += commutator * timestep**2
Omega_2 *= -0.5  # Multiply by -1/2 to complete Omega_2

# Compute the unitary operator U(t) = exp(Omega_1 + Omega_2)
U_t = expm(Omega_1)

print(U_t)

a = np.array([[-0.25897671-0.44314599j, -0.13541928-0.84747526j],[ 0.13541928-0.84747526j, -0.25897671+0.44314599j]])
print(f"\n{a}")