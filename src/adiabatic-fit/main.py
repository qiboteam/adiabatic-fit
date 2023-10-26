import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from qibo import hamiltonians, set_backend, models, gates
from qibo.derivative import parameter_shift
from qibo.config import log, raise_error

from evolution import generate_adiabatic
from training import train_adiabatic_evolution
from rotational_circuit import RotationsGenerator
from plotscripts import show_sample, plot_energy, plot_final_results
from circuit_utils import compute_expectations, compute_derivatives, build_circuit
from generate_data import generate_gamma

set_backend("numpy")

# ------------------------------------------------ INITIALISATION of the problem

# Definition of the Adiabatic evolution
nqubits = 1
finalT = 80
dt = 1e-1

load = True

# problem dimensionality
ndim = 3
colors = ["orange", "red", "blue"]

# to fix divergencies
eps = 1e-5
nshots = 50000

# rank of the polynomial scheduling
nparams = 6

# set hamiltonianas
h0 = hamiltonians.X(nqubits, dense=True)
h1 = hamiltonians.Z(nqubits, dense=True)
# we choose a target observable
obs_target = h1

# ground states of initial and final hamiltonians
gs_h0 = h0.ground_state()
gs_h1 = h1.ground_state()

# energies at the ground states
e0 = obs_target.expectation(gs_h0)
e1 = obs_target.expectation(gs_h1)

print(f"Energy at 0: {e0}")
print(f"Energy at 1: {e1}")

# initial guess for parameters
# picking up from U(0,1) helps to get the monotony
init_params = np.random.uniform(0, 1, nparams)

# Number of steps of the adiabatic evolution
nsteps = int(finalT/dt)

# array of x values, we want it bounded in [0,1]
xarr = np.linspace(0, 1, num=nsteps+1, endpoint=True)
print("\nFirst ten evolution times:")
print(xarr[0:10])

# generate an adiabatic evolution object and an energy callbacks container
evolution, energy = generate_adiabatic(h0=h0, h1=h1, obs_target=obs_target, dt=dt, params=init_params)
# evolve until final time
_ = evolution(final_time=finalT)

# ------------------------------- Generate multi dimensional gamma distributions

cdf, sample, normed_sample = [], [], []

shapes = [2, 10, 100]
scales = [0.5, 0.5, 0.5]
swaps = [True, True, False]
map = {}

for i, (shape, scale, swap) in enumerate(zip(shapes, scales, swaps)):
    cdf_outputs = generate_gamma(
        xarr=xarr, 
        nsteps=nsteps, 
        e0=e0, 
        e1=e1, 
        shape=shape, 
        scale=scale, 
        swap=swap)
    cdf.append(cdf_outputs[0])
    sample.append(cdf_outputs[1])
    normed_sample.append(cdf_outputs[2])
    map.update({f"x{i}": sample[-1]})

df = pd.DataFrame(map)

if False:
    plt.figure(figsize=(5,5))
    sns.pairplot(df, diag_kind="hist", corner=True)
    plt.tight_layout()
    plt.savefig("corner.png")
    show_sample(times=xarr, sample=normed_sample, cdf=cdf, title="Target Cumulative Density Function")
    plot_energy(times=xarr, energies=energy.results, title="Callbacks VS eCDF", cdf=cdf)

# --------------------------------------- Load or compute best parameters vector

if not load:
    print("Training procedure starts here")
    best_params = []

    for i, this_cdf in enumerate(cdf):
        best_params.append(train_adiabatic_evolution(
            nsteps=nsteps,
            xarr=xarr,
            cdf=this_cdf,
            training_n=30,
            init_params=init_params,
            e0=e0,
            e1=e1,
            target_loss=1e-5,
            finalT=finalT,
            h0=h0,
            h1=h1,
            obs_target=obs_target,
            variable_id=f"x{i}"
        ))
    
    np.save(arr=best_params, file="best_params")
else:
    print("Loading already trained parameters.")
    best_params = np.load("best_params.npy")


generators, expectations, derivatives = [], [], []
#build circuit
circuit = build_circuit(ndim=ndim)
print("Built circuit:\n", circuit.draw())

for i in range(ndim):
    generators.append(RotationsGenerator(best_p=best_params[i], finalT=finalT))


#----------------------------- Compute CDF and PDF estimations using the circuit


real_times = np.linspace(0 + eps, finalT - eps, 100)


for i, t in enumerate(real_times):
    if i%10 == 0:
        print(f"Executing with time t={t}")

    expectations.append(
        compute_expectations(
            circuit=circuit,
            times=[t,t,t],
            generators=generators,
            ndim=ndim,
            nshots=nshots
            )
        )

    derivatives.append(
        compute_derivatives(
            circuit=circuit,
            times=[t,t,t],
            generators=generators,
            ndim=ndim,
            nshots=nshots
            )
        )


expectations = np.array(expectations).T
derivatives = np.array(derivatives).T

plt.figure(figsize=(5,5*6/8))
for i in range(ndim):
    plt.plot(xarr, -cdf[i], lw=1.5, ls='--', alpha=0.8, color=colors[i])
    plt.plot(real_times/finalT, expectations[i], lw=1.5, ls='-', alpha=0.8, color=colors[i])
plt.savefig("final_cdfs.png")

plt.figure(figsize=(5,5*6/8))
for i in range(ndim):
    #plt.plot(xarr, -cdf[i], lw=1.5, ls='--', alpha=0.8, color=colors[i])
    plt.plot(real_times/finalT, derivatives[i], lw=1.5, ls='-', alpha=0.8, color=colors[i])
    plt.xlim(0,0.8)
    plt.ylim(0,None)
plt.savefig("final_pdfs.png")

