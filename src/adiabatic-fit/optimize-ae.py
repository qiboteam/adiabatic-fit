import argparse 

import numpy as np
import qibo
import matplotlib.pyplot as plt
from qibo import hamiltonians
from evolution import perform_adiabatic

qibo.set_backend('numpy')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cdf_mode", 
    default="easy", 
    help="Tackled problem, can be 'easy' (gamma) or 'hard' (gaussian mixture)", 
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
    "--nparams",
    default=20,
    help="Number of parameters",
    type=int
)

parser.add_argument(
    "--target_loss",
    default=1e-4,
    help="Target loss function",
    type=float
)


def main(cdf_mode, finalT, dt, nqubits, nparams, target_loss):

    path = f'results/{cdf_mode}'

    # Definition of the Adiabatic evolution

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

    init_params = None

    # Number of steps of the adiabatic evolution
    nsteps = int(finalT/dt)

    # array of x values, we want it bounded in [0,1]
    xarr = np.linspace(0, 1, num=nsteps+1, endpoint=True)

    def cdf_fun(x):
        """Easy gamma function to be fitted."""
        N = nsteps
        nvals = N * 100

        shape = 10
        scale = 0.5
        
        sample = np.random.gamma(shape, scale, nvals)

        normed_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) 

        h, b = np.histogram(normed_sample, bins=nsteps, range=[0,1], density=False)
        # Sanity check
        np.testing.assert_allclose(b, x)
        return np.insert(np.cumsum(h)/len(h), 0, 0), sample, normed_sample


    def cdf_comp(x):
        """More complex function: gaussian mixture."""
        nvals = 50000

        f1 = int(0.6 * nvals)
        f2 = int(0.4 * nvals)

        ind_samples = []
        
        means = [-10, 5]
        sigma = [4, 5]
        fracs = [f1, f2]
        
        for i in range(2):
            ind_samples.append(np.random.normal(means[i], sigma[i], fracs[i]))
        
        sample = np.concatenate(ind_samples)

        normed_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) 

        h, b = np.histogram(normed_sample, bins=nsteps, range=[0,1], density=False)
        # Sanity check
        np.testing.assert_allclose(b, x)
        return np.insert(np.cumsum(h)/len(h), 0, 0), sample, normed_sample


    if cdf_mode == "easy":
        cdf_raw, old_sample, _ = cdf_fun(xarr)
    elif cdf_mode == "hard":
        cdf_raw, old_sample, _ = cdf_comp(xarr)

    # Translate the CDF such that it goes from 0 to 1
    cdf_norm = (cdf_raw - np.min(cdf_raw)) / (np.max(cdf_raw) - np.min(cdf_raw))
    # And now make it go from the E_initial to E_final (E0 to E1)
    cdf = e0 + cdf_norm*(e1 - e0)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Generate n training points
    training_n =100
    # But select those for which the difference between them is greater than some threshold
    min_step = 1e-3
    # but never go more than max_skip points without selecting one
    max_skip = 0

    if training_n > nsteps:
        raise Exception("The notebook cannot run with nsteps < training_n")

    # Select a subset of points for training, but skip first and include last
    idx_training_raw = np.linspace(0, nsteps, num=training_n, endpoint=True, dtype=int)[1:]

    # And, from this subset, remove those that do not add that much info
    idx_training = []
    cval = cdf[0]
    nskip = 0
    for p in idx_training_raw[:-2]:
        diff = cval - cdf[p]
        if diff > min_step or nskip > max_skip:
            nskip = 0
            idx_training.append(p)
            cval = cdf[p]
        else:
            nskip += 1
            
    idx_training.append(idx_training_raw[-1])

    cdf_training = cdf[idx_training]
    norm_cdf = np.abs(cdf_training) # To normalize the points according to their absolute value

    init_params = np.random.randn(nparams)

    def plot_for_energies(parameters, label="", true_law=None, title=""):
        """Plot energies, training points and CDF for a set of energies given a set of parameters
        """
        energies = perform_adiabatic(
            params=parameters,
            finalT=finalT,
            h0=h0,
            h1=h1,
            obs_target=obs_target,
        )

        plt.title(title)
        plt.plot(xarr, -np.array(cdf), label="eCDF", color='black', lw=1, ls='-')
        plt.plot(xarr, -np.array(energies), label=label, color='red', lw=2, alpha=0.6)
        plt.plot(xarr[idx_training], -np.array(cdf_training), 'o', label="Training points", color='blue', alpha=0.4, markersize=6)
        if true_law != None:
            plt.plot(xarr, true_law, c='orange', lw=1, ls='--')
        plt.xlabel("x")
        plt.ylabel("cdf")
        plt.legend()
        plt.show()

    plot_for_energies(init_params, label="Initial state", title="Not trained evolution")
    print(f"Training on {len(idx_training)} points of the total of {nsteps}")


    # --------------------------------- optimization ----------------------------

    # Definition of the loss function and optimization routine
    good_direction = 1 if (e1-e0) > 0 else -1
    penalty = True

    def loss_evaluation(p, return_penalty = False):
        """Evaluating loss function related to the cdf fit"""
        
        # Retrieve the energy per time step for this set of parameters
        energies = perform_adiabatic(
            params=p,
            finalT=finalT,
            h0=h0,
            h1=h1,
            obs_target=obs_target,
        )
        
        # Select the points we are training on
        e_train = energies[idx_training]
        
        loss = np.mean((e_train - cdf_training)**2 / norm_cdf)
        
        if penalty or return_penalty:
            # Penalty term for negative derivative
            delta_energy = good_direction*np.diff(energies)
            # Remove non-monotonous values
            delta_energy *= (delta_energy < 0)
        
            pos_penalty = np.abs(np.sum(delta_energy))
            val_loss = loss
            
            loss = val_loss + pos_penalty
            
            if return_penalty:
                return loss, val_loss, pos_penalty
            #print(f"{loss=}, {pos_penalty=}")
        
        return loss

    def optimize(force_positive=False, target=5e-2, max_iterations=50000, max_evals=500000, initial_p=None):
        """Use qibo to optimize the parameters of the schedule function"""
        
        options = {
            "verbose": -1,
            "tolfun": 1e-12,
            "ftarget": target, # Target error
            "maxiter": max_iterations, # Maximum number of iterations
            "maxfeval": max_evals, # Maximum number of function evaluations
            "maxstd": 20
        }
        
        if force_positive:
            options["bounds"] = [0, 1e5]
            
        if initial_p is None:
            initial_p = init_params
        else:
            print("Reusing previous best parameters")
            
        result = qibo.optimizers.optimize(loss_evaluation, initial_p, method="cma", options=options)

        return result, result[1]


    _, best_p = optimize(target=target_loss, force_positive=False, initial_p=None, max_iterations=50000)

    total_loss, loss_val, penalty_val = loss_evaluation(best_p, return_penalty=True)
    print(f"""\nBest set of parameters: {best_p=}
    For which loss = {loss_val:.4} and the positivity penalty is = {penalty_val:.4}
    Total loss: {total_loss:.4}""") 

    plot_for_energies(best_p, label="Initial state", title="Trained evolution")

    import os
    os.system(f'mkdir {path}')
    np.save(arr=old_sample, file=path+'/not_normed_sample')
    np.save(arr=xarr, file=path+'/xarr')
    np.save(arr=best_p, file=path+'/best_p')
    np.save(arr=cdf, file=path+'/cdf')

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)