"""
Problem 6.33: Degenerate perturbation theory for the 2D infinite square well.

Unperturbed (a=1) eigenfunctions, well centered on the origin, x,y in [-1/2, 1/2]:

    psi_{ab}(x,y) = 2 sin(a*pi*(x-1/2)) sin(b*pi*(y-1/2))

For a != b, psi_{ab} and psi_{ba} are degenerate; the "good" states for
degenerate perturbation theory are the symmetric/antisymmetric combinations

    psi_+ = (psi_ab + psi_ba) / sqrt(2)
    psi_- = (psi_ab - psi_ba) / sqrt(2)

Here we plot a contour map of psi_- for a=5, b=7.
"""
import numpy as np
import matplotlib.pyplot as plt


def psi(n, m, x, y):
    return 2 * np.sin(n * np.pi * (x - 0.5)) * np.sin(m * np.pi * (y - 0.5))


a, b = 5, 7

n = 400
coords = np.linspace(-0.5, 0.5, n)
X, Y = np.meshgrid(coords, coords)

psi_minus = (psi(a, b, X, Y) - psi(b, a, X, Y)) / np.sqrt(2)

vmax = np.abs(psi_minus).max()
levels = np.linspace(-vmax, vmax, 31)

plt.figure(figsize=(6, 5.5))
cs = plt.contourf(X, Y, psi_minus, levels=levels, cmap="RdBu_r")
plt.contour(X, Y, psi_minus, levels=[0], colors="k", linewidths=0.5)
plt.colorbar(cs, label=r"$\psi_-$")
plt.xlabel("x")
plt.ylabel("y")
plt.title(rf"$\psi_-$ for $a={a}$, $b={b}$")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("prob6.33_psi_minus.png", dpi=150)
plt.show()
print("Plot saved to prob6.33_psi_minus.png")
