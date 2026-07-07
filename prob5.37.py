import numpy as np
import matplotlib.pyplot as plt

k = np.linspace(0.02, 10, 50000)
E = k**2 / 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, alpha in zip(axes, [0, 1]):
    z = np.cos(k) + alpha / k * np.sin(k)
    dz_dk = -np.sin(k) - alpha / k**2 * np.sin(k) + alpha / k * np.cos(k)

    allowed = np.abs(z) <= 1.0
    denom = k * np.sqrt(np.where(allowed, 1 - z**2, np.nan))
    g = np.where(allowed, np.abs(dz_dk) / (np.pi * denom), np.nan)

    ax.plot(E, g, color="steelblue", lw=1.0)
    ax.set_xlabel(r"$E$", fontsize=13)
    ax.set_ylabel(r"$g(E)$", fontsize=13)
    ax.set_title(rf"$\alpha = {alpha}$", fontsize=14)
    ax.set_xlim(0, E[-1])
    ax.set_ylim(0, 0.5)

fig.suptitle(r"Density of States $g(E)$ — Dirac Comb (Prob 5.37)", fontsize=14)
plt.tight_layout()
plt.savefig("prob5.37_dos.png", dpi=150)
plt.show()
