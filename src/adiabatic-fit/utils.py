import numpy as np
import scipy.special as sps

def compute_expectations(counter, nqubits):
    """Compute N <Z> if an N qubits system is used."""

    prob0 = np.zeros(nqubits, dtype=int)
    prob1 = np.zeros(nqubits, dtype=int)

    for q in range(nqubits):
        for element in counter:
            elem = [bit for bit in element]
            if elem[q] == '0':
                prob0[q] += counter[element]
            else:
                prob1[q] += counter[element]
    
    nshots = prob0[0] + prob1[0]
    results = (prob0 - prob1) / nshots

    # fix numerical instabilities
    for i, result in enumerate(results):
        if np.abs(result) > 0.2 and result < 0:
            results[i] *= -1
                
    return results

#def compute_derivatives():
