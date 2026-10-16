"""
Problem 8.15: variational treatment of the hydrogen molecule H2 (Sugiura's
integrals).  x = R/a is the proton separation in Bohr radii.  The upper sign
is the singlet (symmetric spatial state, "+"), the lower sign the triplet
(antisymmetric spatial state, "-").

    I   = e^{-x}(1 + x + x^2/3)                       overlap
    It  = e^{ x}(1 - x + x^2/3)
    X   = (1 + x) e^{-x}
    D   = 1/x - (1 + 1/x) e^{-2x}                     one-electron direct integral

    D2  = 1/x - e^{-2x}[x^2/6 + (3/4)x + 11/8 + 1/x]
    X2  = e^{-2x}[5/8 - (23/20)x - (3/5)x^2 - (1/15)x^3]
          + (6/5)(1/x){ I^2 (gamma + ln x) + It^2 Ei(-4x) - 2 I It Ei(-2x) }

    E1  = -13.6 eV

    kinetic       = 2 ( E1 - 2 E1 (1 +- I X)/(1 +- I^2) )
    e-p potential  = -(4/2)(-2 E1) ( 1 + D +- 2 I X/(1 +- I^2) )
    e-e potential  = -2 E1 (D2 +- X2)/(1 +- I^2)

This script plots the three energy components vs x for both signs.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

E1 = -13.6  # eV

g = np.euler_gamma


def integrals(x):
    I = np.exp(-x) * (1 + x + x**2 / 3)
    It = np.exp(x) * (1 - x + x**2 / 3)
    X = (1 + x) * np.exp(-x)
    D = 1 / x - (1 + 1 / x) * np.exp(-2 * x)

    D2 = 1 / x - np.exp(-2 * x) * (x**2 / 6 + 0.75 * x + 11 / 8 + 1 / x)
    # I^2 (gamma+ln x) + It^2 Ei(-4x) - 2 I It Ei(-2x)  -- written this way so the
    # e^{+2x} in It never appears alone (no overflow) and each product -> 0 as x->inf
    ei_part = (I**2 * (g + np.log(x))
               + It**2 * expi(-4 * x)
               - 2 * I * It * expi(-2 * x))
    X2 = (np.exp(-2 * x) * (5 / 8 - (23 / 20) * x - (3 / 5) * x**2 - (1 / 15) * x**3)
          + (6 / 5) / x * ei_part)
    return I, It, X, D, D2, X2


def components(x, sign):
    """sign = +1 -> singlet, sign = -1 -> triplet."""
    I, _It, X, D, D2, X2 = integrals(x)
    denom = 1 + sign * I**2

    kinetic = 2 * (E1 - 2 * E1 * (1 + sign * I * X) / denom)
    e_p = -(4 / 2) * (-2 * E1) * (1 + D + sign * 2 * I * X / denom)
    e_e = -2 * E1 * (D2 + sign * X2) / denom
    return kinetic, e_p, e_e


x = np.linspace(0.6, 8.0, 1500)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True)
titles = ["Kinetic energy", "Electron-proton potential", "Electron-electron potential"]

for s, label, color in [(+1, "singlet (+)", "tab:red"),
                        (-1, "triplet (-)", "tab:blue")]:
    comps = components(x, s)
    for ax, comp, title in zip(axes, comps, titles):
        ax.plot(x, comp, color=color, lw=2, label=label)
        ax.set_title(title)
        ax.set_xlabel(r"$x = R/a$")
        ax.grid(alpha=0.3)

axes[0].set_ylabel("energy (eV)")
axes[0].legend()

# reference lines for the separated-atom limits (two H atoms)
axes[0].axhline(-2 * E1, color="gray", ls="--", lw=1)   # +27.2 eV
axes[1].axhline(4 * E1, color="gray", ls="--", lw=1)    # -54.4 eV
axes[2].axhline(0, color="gray", ls="--", lw=1)

# clip the y-range so the singlet (bound) branch stays readable; the triplet
# diverges as x -> 0 because of the 1/(1 - I^2) normalization factor
axes[0].set_ylim(20, 55)
axes[1].set_ylim(-170, 150)

fig.suptitle("Problem 8.15: H$_2$ energy components vs. proton separation")
fig.tight_layout()
fig.savefig("prob8.15.png", dpi=150)
print("saved prob8.15.png")

# quick numeric sanity check at large x (separated atoms).
# Total must also include proton-proton repulsion V_pp = -2 E1 / x.
for s, name in [(+1, "singlet"), (-1, "triplet")]:
    xt = np.array([60.0])
    k, ep, ee = components(xt, s)
    vpp = -2 * E1 / xt
    total = k[0] + ep[0] + ee[0] + vpp[0]
    print(f"{name:8s} x=60:  T={k[0]:8.3f}  V_ep={ep[0]:8.3f}  V_ee={ee[0]:8.3f}  "
          f"V_pp={vpp[0]:6.3f}  total={total:8.3f} eV  (expect ~ {2*E1:.1f})")
