import matplotlib.pyplot as plt

def plot_energy(times, energies, title, cdf=None):
    """Plot energies"""
    plt.figure(figsize=(8,5))
    plt.plot(times, energies, color="purple", lw=2, alpha=0.8, label="Energy callbacks")
    if cdf is not None:
        plt.plot(times, cdf, color="orange", lw=2, alpha=0.8, label="Empirical CDF")
    plt.title(title)
    plt.xlabel(r'$\tau$')
    plt.ylabel("E")
    plt.grid(True)
    plt.legend()
    plt.savefig("energies.png", bbox_inches="tight")

def show_sample(times, sample, cdf, title):
    """Plot energies"""
    plt.figure(figsize=(8,5))
    plt.hist(sample, bins=50, color="black", alpha=0.3, cumulative=True, density=True, label="Sample")
    plt.plot(times, -cdf, color="orange", lw=2, alpha=0.8, label="Target CDF")
    plt.title(title)
    plt.xlabel(r'$\tau$')
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.savefig("sample.png", bbox_inches="tight")

def plot_final_results(times, sample, e, de, title):
    """Plot final results"""
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.title("PDF histogram")
    plt.hist(sample, bins=20, color="orange", histtype="stepfilled", edgecolor="orange", hatch="//", alpha=0.3, density=True)
    plt.hist(sample, bins=20, color="orange", alpha=1, lw=1.5, histtype="step", density=True)
    plt.plot(times, de, color="purple", lw=2, label=r"Estimated $\rho$")
    plt.xlabel('x')
    plt.ylabel(r"$\rho$")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("CDF histogram")
    plt.hist(sample, bins=20, color="orange", histtype="stepfilled", edgecolor="orange", hatch="//", alpha=0.3, density=True, cumulative=True)
    plt.hist(sample, bins=20, color="orange", alpha=1, lw=1.5, histtype="step", density=True, cumulative=True)
    plt.plot(times, -np.array(e), color="purple", lw=2, label=r"Estimated $F$")
    plt.xlabel('x')
    plt.ylabel(r"$F$")
    plt.legend()

    plt.tight_layout()
    plt.savefig("result.png", bbox_inches="tight")
