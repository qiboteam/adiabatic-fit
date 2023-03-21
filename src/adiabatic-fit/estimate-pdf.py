import argparse
import numpy as np
import qibo 
from qibo import hamiltonians
from psr import parameter_shift
from evolution import generate_schedule
import scipy
from qibo.hamiltonians import SymbolicHamiltonian 
from qibo.symbols import Z

import scipy.special as sps  
import matplotlib.pyplot as plt


qibo.set_backend('numpy')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cdf_mode", 
    default="easy", 
    help="Tackled problem, can be 'easy' (gamma), 'hard' (gaussian mixture) or 'hep' (pp->ttbar)", 
    type=str
)

parser.add_argument(
    "--dt",
    default=0.1,
    help="time-step during the evolution",
    type=float,
)

parser.add_argument(
    "--finalT",
    default=50,
    help="Total real time of the evolution in s",
    type=float,
)

parser.add_argument(
    "--nqubits",
    default=1,
    help="Number of qubits",
    type=int
)

parser.add_argument(
    "--nshots",
    default=100000,
    help="Number of qubits",
    type=int
)

def smoothing(array_raw, epsilon=2e-1):
    array = array_raw.T
    new = []
    counter = 0
    new.append(array[0])
    for idx in range(1, len(array)-2, 1):
        prev_value = new[-1]

        for i in range(20):
            new_number = (array[idx+i] + array[idx-i])/2

            if np.abs(np.mean(new_number - prev_value)) < epsilon:
                break

            counter += 1
        
        new.append(new_number)

    new.append(array[-2])
    new.append(array[-1])
    print(f'{counter}/{len(array)} elements are been corrected.')
    return np.array(new).T


def main(cdf_mode, dt, finalT, nqubits, nshots):

    # -----------------------------useful variables ----------------------------

    shots = True
    nruns = 10

    # --------------------------- loading data ---------------------------------

    path = f'results/{cdf_mode}/'

    # load data
    xarr = np.load(path+'xarr.npy')
    cdf = np.load(path+'cdf.npy')   
    best_p = np.load(path+'best_p.npy')
    sample = np.load(path+'not_normed_sample.npy')

    nsteps = int(finalT/dt)


    # ------------------------- Summary of the problem -------------------------
    # Definition of the Adiabatic evolution

    print('Target energies in the adiabatic evolution problem')

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

    # -------------------------- calculating labels ----------------------------
    print('Calculating labels from sample.\n')
        
    x_min = np.min(sample)
    x_max = np.max(sample)

    not_normed_xarr = xarr * (x_max - x_min) + x_min
    normed_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
    
    shape, scale = 10, 0.5

    def gauss(x, mu, s):
        return scipy.stats.norm.pdf(x, mu, s) * (x_max - x_min)

    def F_gauss(x, mu, s):
        return scipy.stats.norm.cdf(x, mu, s)

    if cdf_mode == 'easy':
        labels =  (not_normed_xarr**(shape-1)*(np.exp(-not_normed_xarr/scale) /  (sps.gamma(shape)*scale**shape))) * (x_max - x_min)
        e_labels = sps.gammainc(shape, not_normed_xarr/scale) * (1. / sps.gamma(shape)) 
        e_labels = (e_labels - np.min(e_labels)) / (np.max(e_labels - np.min(e_labels)))

    if cdf_mode == 'hard':
        labels = 0.6*gauss(not_normed_xarr, -10, 4) + 0.4*gauss(not_normed_xarr, 5, 5)
        e_labels = 0.6*F_gauss(not_normed_xarr, -10, 4) + 0.4*F_gauss(not_normed_xarr, 5, 5)

    # ---------------------------- integration tools ---------------------------

    from functools import lru_cache
    from scipy.integrate import quad

    poly, derpoly = generate_schedule(best_p)

    @lru_cache
    def sched(t):
        """The schedule at a time t is poly(t/finalT)"""
        return poly(t/finalT)

    @lru_cache
    def eigenval(t):
        """Compute the eigenvalue of the Hamiltonian at a time t"""
        s = sched(t)
        return np.sqrt(s**2 + (1-s)**2)

    @lru_cache
    def integral_eigen(t, n=int(1e6)):
        """Compute the integral of eigenval from 0 to t
        """
        res = quad(eigenval, 0, t)
        return res[0]

    @lru_cache
    def u00(t, swap = 1.0):
        """Compute the value of u00 (real and imag) at a time T"""
        if t == finalT:
            t -= 1e-2
        integral = integral_eigen(t)
        l = eigenval(t)
        s = sched(t)
        
        # Normalization for the change of basis matrix P^{-1}HP = H_diagonal so that PP^-1 = I
        normalize = 2.0/(1-s)*np.sqrt(l*(l-s))
        fac = swap*(l-s)/(1-s)
        
        # (the multiplication by t not sure where does it come from)
        ti = t/finalT
        real_part = swap*np.cos(integral)*(1+fac)/normalize
        imag_part = np.sin(integral)*(1-fac)/normalize
        
        return real_part, imag_part

    @lru_cache
    def u10(t):
        """Compute the value of u10 (real and imag), the offdiagonal term"""
        pr, pi = u00(t, swap=-1.0)
        return pr, pi

    @lru_cache
    def old_rotation_angles(t):
        x, y = u00(t)
        u, z = u10(t)

        a = x + 1j*y
        b = -u + 1j*z

        arga = np.angle(a)
        moda = np.absolute(a)
        argb = np.angle(b)
        
        # Another interpretation
        integral = I = integral_eigen(t)
        l = eigenval(t)
        s = sched(t)
        
        fac = (l-s)/(1-s)
        inside = 1 +  fac**2 + 2*fac*np.cos(2*integral)
        norma = (1-s)/2/np.sqrt(l*(l-s))
        
        new_moda = np.sqrt(inside)*norma
        
        theta = - 2 * np.arccos(moda) 
        psi = - 0.5 * np.pi - arga + argb
        phi = - arga + np.pi * 0.5 - argb
        return psi, theta, phi


    @lru_cache
    def sched_p(t):
        """The schedule at a time t is poly(t/finalT)"""
        return derpoly(t/finalT)/finalT

    @lru_cache
    def eigenvalp(l, s, sp):
        return sp*(2*s - 1)/l

    @lru_cache
    def n(l,s):
        return (1-s)/2/np.sqrt(l*(l-s))

    @lru_cache
    def nder(l, s, lp, sp):
        roote = l*(l-s)
        upder = (1-s)*(2*lp*l - lp*s - sp*l)
        inter = -sp - upder/2/roote
        return 1/2/np.sqrt(roote)*inter

    @lru_cache
    def f(l,s):
        return (l-s)/(1-s)

    @lru_cache
    def fp(l, s, lp, sp):
        return (lp-sp-lp*s+sp*l)/(1-s)**2

    @lru_cache
    def rotation_angles(t):
        I = integral_eigen(t)
        l = eigenval(t)
        s = sched(t)
            
        fac = f(l,s)
        
        norma = n(l,s)
        inside00 = gt = 1 +  fac**2 + 2*fac*np.cos(2*I)
        
        absu00 = norma*np.sqrt(inside00)    
        
        upf = (1-l)
        dpf = (1+l-2*s)
        sinIt = np.sin(I)
        cosIt = np.cos(I)
        
        arga = np.arctan2(sinIt*upf, cosIt*dpf)
        argb = np.arctan2(sinIt*dpf, -cosIt*upf)

        theta = -2 * np.arccos(absu00)
        psi = - 0.5 * np.pi - arga + argb
        phi = - arga + np.pi * 0.5 - argb
        
        return phi, theta, psi

    @lru_cache
    def derivative_rotation_angles(t):
        s = sched(t)
        l = eigenval(t)
        I = integral_eigen(t)
        
        derI = l
        sp = sched_p(t) 
        lp = eigenvalp(l, s, sp)
        
        nt = n(l, s)
        ntp = nder(l, s, lp, sp)
            
        ft = f(l,s)
        ftp = fp(l, s, lp, sp)
            
        gt = 1 + ft**2 + 2*ft*np.cos(2*I)    
        
        # Terms of the final sum for the derivative of the theta
        x1 = ntp*np.sqrt(gt)
        y1 = 2*ft*ftp
        
        y2 = 2*ftp*np.cos(2*I)
        y3 = -2*ft*np.sin(2*I)*(2*derI)
        
        dgt = (y1+y2+y3)
        
        absu00 = nt*np.sqrt(gt)
        dabsu = x1 + nt/np.sqrt(gt)*(dgt)/2.0
        darcos = 2.0/np.sqrt(1 - absu00**2)
        dtheta = darcos * dabsu
        
        # Let's do the derivative of the phi,psi
        upf = (1-l)
        dpf = (1+l-2*s)
        tanI = np.tan(I)
        
        dtan = l/np.cos(I)**2
        dfrac_01 = 2*(lp - sp + sp*l - s*lp)/upf**2
        dfrac_00 = -2*(lp - sp + sp*l - s*lp)/dpf**2
        
        inside_arga = tanI*(upf/dpf)
        inside_argb = -tanI*(dpf/upf)
        
        dinside_arga = dtan*(upf/dpf) + tanI*dfrac_00
        darga = dinside_arga / (1 + inside_arga**2)
        
        dinside_argb = -dtan*(dpf/upf) - tanI*dfrac_01
        dargb = dinside_argb / (1 + inside_argb**2)
        
        dpsi = -darga + dargb
        dphi = -darga - dargb
        
        return dphi, dtheta, dpsi

    @lru_cache
    def rotations_circuit(t):

        psi, theta, phi = rotation_angles(t)

        c = qibo.models.Circuit(1)
        r1 = qibo.gates.RZ(q=0, theta=psi)
        r2 = qibo.gates.RX(q=0, theta=theta)
        r3 = qibo.gates.RZ(q=0, theta=phi)
        c.add([r1, r2, r3])
        c.add(qibo.gates.M(0))

        return c

    @lru_cache
    def numeric_derivative(t, h=1e-7):
        # Do the derivative with 4 points
        a1, b1, c1 = rotation_angles(t + 2*h)
        a2, b2, c2 = rotation_angles(t + h)
        a3, b3, c3 = rotation_angles(t - h)
        a4, b4, c4 = rotation_angles(t - 2*h)

        dd1 = (-a1 + 8*a2 - 8*a3 + a4)/12/h
        dd2 = (-b1 + 8*b2 - 8*b3 + b4)/12/h
        dd3 = (-c1 + 8*c2 - 8*c3 + c4)/12/h
        return dd1, dd2, dd3
    

    def psr_energy(t, state, obs=obs_target, nshots=None, nruns=None, h=1e-7, analytic_der = True):
        """Calculates derivative of the energy with respect to the real time t."""
        c = rotations_circuit(t)

        if analytic_der:
            dd1, dd2, dd3 = derivative_rotation_angles(t)
        else:
            dd1, dd2, dd3 = numeric_derivative(tt, h=h)

        par1 = parameter_shift(circuit=c, hamiltonian=obs, parameter_index=0, initial_state=state, nshots=nshots, nruns=nruns)

        d1 = dd1 * par1
        d2 = dd2 * parameter_shift(circuit=c, hamiltonian=obs, parameter_index=1, initial_state=state, nshots=nshots, nruns=nruns)
        d3 = dd3 * parameter_shift(circuit=c, hamiltonian=obs, parameter_index=2, initial_state=state, nshots=nshots, nruns=nruns)

        return (d1 + d2 + d3)

    observable = SymbolicHamiltonian(Z(0))
    backend = h1.backend    

    print('Here we start the simulation thanks to the circuits.\n')
    print(f'We are going to use:\n {nshots} shots and {nruns} runs')


    epsilon=0.001

    times = np.linspace(0+epsilon,finalT-epsilon, nsteps+1)


    def run_experiment(shots=shots, nshots=nshots, nruns=nruns, times=times, h=1e-5):

        if shots:
            e = np.zeros((nruns, len(times))) 
            de = np.zeros((nruns, len(times))) 
        else:
            nruns = 1
            e = []
            de = []

        for exp in range(nruns):
            print('Running experiment ', exp+1, ' with shots=', shots)
            
            for i, t in enumerate(times):
                c = rotations_circuit(t)

                if shots:
                    e[exp][i] = backend.execute_circuit(
                        circuit=c, 
                        initial_state=gs_h0, 
                        nshots=nshots).expectation_from_samples(observable)
                    de[exp][i] = psr_energy(t, gs_h0, obs=observable, nshots=nshots, nruns=1, h=h)
                else:
                    e.append(obs_target.expectation(gs_h0))
                    de.append(psr_energy(t, gs_h0, obs=observable, h=h))

        return e, de


    print('Running noisy simulation')

    _, de = run_experiment(shots=True, nshots=int(nshots), nruns=nruns, times=times)

    mean_de = smoothing(np.mean(de, axis=0)*finalT)

    plt.title('PDF estimation via QAML')
    plt.plot(xarr, mean_de, color='red', alpha=0.8, label='Estimation')
    plt.hist(normed_sample, density=True, color='black', alpha=0.4, histtype='stepfilled', label='Data', bins=50)
    plt.hist(normed_sample, density=True, color='black', alpha=0.8, histtype='step', bins=50)
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)