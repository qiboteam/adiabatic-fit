## Determining probability density functions with adiabatic quantum computing

This code can be used for derivating PDF via quantum adiabatic machine learning (QAML). 
The strategy is divided into three steps:

1. we fit $F$, the Cumulative Density Function (CDF) of a sample using an adiabatic
evolution as quantum machine learning model;
2. we convert the adiabatic hamiltonian into a circuit composed of rotations;
3. we calculate the derivative of our estimated CDF (which is obtained thank to
the circuit execution) using the parameter shift rule.

More details about the QAML model can be found in [this](link) presentation. 

#### Prerequisites

Some packages are required in order to run our code. 
First of all you need `qibo`, which can be installed by following the
[official documentation](https://qibo.science/docs/qibo/stable). 

We also make use of `scipy`. Installation instructions [here](https://scipy.org/install/).

#### How encode the problem and to optimize an adiabatic evolution?

In our example code we consider two cases: a gamma distribution and a gaussian
mixture distribution. 

For simplicity, these examples are tackled using a 1-qubit adiabatic evolution
from an initial Pauli-X hamiltonian to a final Pauli-Z hamiltonian. These two
hamiltonians are useful in order to limit the problem between $F(x=0)=0$ and
$F(x=1)=1$.

The code can be executed for the easiest case by running:

```
python optimize-ae.py --cdf_mode easy --dt 0.1 --finalT 50 --nqubits 1 --nparams 20 --target_loss 1e-4
```

If you want to try to fit the gaussian mixtura just select `--cdf_mode hard`.

#### How to derivate the CDf in order to get the PDF

Once `optimize-ae.py` is run, the results are saved in a proper folder. 
At this point, the PDF can be calculated by running:

```
python estimate-pdf.py --cdf_mode easy --dt 0.1 --finalT 50 --nqubits 1 --nshots 100000
```

where you can set the number of shots used for evaluating the circuit. 
Note that you must set the same arguments defined above in order to tackle the same
problem.