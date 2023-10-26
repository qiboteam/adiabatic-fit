import numpy as np

from qibo.config import raise_error
from qibo import models, gates

def build_circuit(ndim):
    """
    Build quantum circuit of shape:

        q0 -- RZ -- RX -- RZ -- M
        q1 -- RZ -- RX -- RZ -- M
                 ...
        qn -- RZ -- RX -- RZ -- M
    
    composed on `ndim` qubits and three rotations per qubit.
    """
    c = models.Circuit(ndim)
    
    for i in range(ndim):
        c.add(gates.H(q=i))
        c.add(gates.RZ(q=i, theta=0))
        c.add(gates.RX(q=i, theta=0))
        c.add(gates.RZ(q=i, theta=0))
    c.add(gates.M(*range(ndim), collapse=False))

    return c


def collect_params(times, generators, ndim):
    """
    Collect circuit's parameters based on a vector of times.
    
    Args:
        times (list of float): corresponding to the normalized target variable.
        generators (list of `rotational_circuit.RotationsGenerator`): """
    
    if len(times) != ndim:
        raise_error(
            ValueError,
            f"The passed variable is {len(times)} dimensional, while problem dimensionality is {ndim}."
        )
    if len(generators) != ndim:
        raise_error(
            ValueError,
            f"The generators list is {len(generators)} dimensional, while problem dimensionality is {ndim}."
        )
    
    params = []
    for i in range(ndim):
        params.extend(generators[i].rotation_angles(times[i]))
    return np.array(params) 


def compute_expectations(circuit, times, generators, ndim, nshots, shift=None):
    """
    Compute ndim <Z>'s if an ndim-qubits system is used.
    
    Args:
        circuit: pre-computed circuit using the `build_circuit` function.
        times: ndim variable.
        generators: pre-computed RotationsGenerator.
        ndim: problem dimensionality.
        nshots: number of shots used to compute <Z>'s.
        shift (int): 
            - if None: no shift is applied;
            - if +j: the j-th rotational angle is shifted forward of pi/2;
            - if -j: the j-th rotational angle is shifted backward of pi/2.
            [j can assume the following values: (-3, -2, -1, +1, +2, +3)]
    """

    shifts = [None, -3, -2, -1, +1, +2, +3]
    if shift not in shifts:
        raise_error(
            ValueError,
            f"The required shift is not allowed. Please set one of: {shifts}"
        )

    nqubits = circuit.nqubits
    parameters = collect_params(times, generators, ndim)

    if shift is not None:
        # identify target rotations
        shifted_rotation = int(np.abs(shift))
        # select all the multiple of 3 until the parameters length is covered
        # e.g. [0,3,6,9]
        indices = np.arange(0, len(parameters), 3)
        # keep indices if |j| = 1, jump to the neighbors if |j| > 1
        indices += (shifted_rotation - 1)

        if shift > 0:
            parameters[indices] += np.pi/2
        if shift < 0:
            parameters[indices] -= np.pi/2

    circuit.set_parameters(parameters)
    counter = circuit(nshots=nshots).frequencies()

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


def compute_derivatives(circuit, times, generators, ndim, nshots):

    d_angles, circuit_derivatives = [], []

    for generator, time in zip(generators, times):
        d_angles.append(generator.derivative_rotation_angles(time))

    # after this d_angles[j] contains the ndim derivatives wrt angle_j
    # with j = 0, 1, 2 --> phi, theta, psi
    d_angles = np.array(d_angles).T

    for j in range(ndim):
        # forward exec shifting all the (j+1)-th rotations 
        forward_execution = compute_expectations(
            circuit=circuit,
            times=times,
            generators=generators,
            ndim=ndim,
            nshots=nshots,
            shift = j+1
        )
        # backward exec shifting all the (j+1)-th rotations
        backward_execution = compute_expectations(
            circuit=circuit,
            times=times,
            generators=generators,
            ndim=ndim,
            nshots=nshots,
            shift = -1 * (j+1)
        )

        circuit_derivatives.append(0.5 * (forward_execution - backward_execution))

    circuit_derivatives = np.array(circuit_derivatives)

    # d<.>/dang * dang/dt
    derivatives = circuit_derivatives * d_angles

    # d<.> / dt = dtheta/dt + dphi/dt + dpsi/dt
    derivatives = np.sum(derivatives, axis=1)

    return derivatives    