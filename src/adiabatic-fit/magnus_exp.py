import sympy as sp
import numpy as np

# Load parameters and define symbolic variable t
params = np.load("./fit-results/easy-gamma/best_p.npy")
t = sp.symbols('t')
nparams = len(params)

# Define s(t) and Hamiltonian H(t)
s_t = np.sum((t ** np.arange(1, 1 + nparams) * params))
s_t /= np.sum(params)
H = sp.Matrix([[s_t, 1 - s_t], [1 - s_t, -s_t]])

# Define the time interval and timestep
timestep = 0.002
t_max = 1  # maximum value for t

# TODO: check if this part makes sense or not!!!!!
num_steps = int(t_max / 50 / timestep)  # number of steps for t/50
time_points = np.linspace(0, t_max / 50, num_steps)  # Time steps from 0 to t/50

# Convert H(t) elements to functions for numerical evaluation
H_funcs = [[sp.lambdify(t, H[i, j], "numpy") for j in range(2)] for i in range(2)]

# Function to evaluate H(t) numerically at a specific time
def H_numeric(t_val):
    return np.array([[f(t_val) for f in row] for row in H_funcs])

# Calculate Omega_1 by summing up H(t) over time points
Omega_1 = np.zeros((2, 2), dtype=complex)
for t_val in time_points:
    Omega_1 += H_numeric(t_val) * timestep

Omega_1 *= -1j  # Multiply by -i to complete Omega_1

# Calculate Omega_2 using the commutators of H(t) at each pair of times
Omega_2 = np.zeros((2, 2), dtype=complex)
for i, t_i in enumerate(time_points):
    for j, t_j in enumerate(time_points[:i]):
        H_ti = H_numeric(t_i)
        H_tj = H_numeric(t_j)
        commutator = np.dot(H_ti, H_tj) - np.dot(H_tj, H_ti)  # [H(t_i), H(t_j)]
        Omega_2 += commutator * timestep**2

Omega_2 *= -0.5  # Multiply by -1/2 to complete Omega_2

# Compute norms of Omega_1 and Omega_2 to assess the contribution of higher-order terms
Omega_1_norm = np.linalg.norm(Omega_1, 'fro')
Omega_2_norm = np.linalg.norm(Omega_2, 'fro')
higher_order_contribution = Omega_2_norm / Omega_1_norm

# Display results
print(f"Omega_1:\n{Omega_1}\nNorm of Omega_1: {Omega_1_norm}")
print(f"Omega_2:\n{Omega_2}\nNorm of Omega_2: {Omega_2_norm}")
print(f"Contribution of higher-order terms: {higher_order_contribution}")
