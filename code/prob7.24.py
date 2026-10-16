"""
Problem 7.24: Weak-field Zeeman effect on top of fine structure for the
n=2 states of hydrogen.

Fine structure alone splits n=2 into two levels (Griffiths Eq. 6.65-type
result, but computed state-by-state so l=0 and l=1 pick up different
coefficients since the l=0 states get no spin-orbit term while the l=1
states get no Darwin term):

    E(l=0, j=1/2) = -3.4 eV [1 + 5 alpha^2/16]   (2-fold: mj = +-1/2)
    E(l=1, j=3/2 or 1/2) = -3.4 eV [1 + alpha^2/16]   (6-fold degenerate)

Turning on B_ext adds the weak-field (anomalous) Zeeman term g_j mu_B B mj,
which is linear in B with slope g_j*mj (in units of mu_B):

    |2 0 1/2 +-1/2>:            slope = +-1
    |2 1 3/2 +-3/2>:            slope = +-2
    |2 1 3/2 +-1/2>:            slope = +-2/3
    |2 1 1/2 +-1/2>:            slope = +-1/3

giving the 8 energies quoted in the problem. This script plots all eight
as straight lines in B_ext, labeling each with its ket and slope.
"""
import numpy as np
import matplotlib.pyplot as plt

E2 = -3.4  # eV, unperturbed n=2 energy
alpha = 1 / 137.036
mu_B = 5.7883818e-5  # eV/T, Bohr magneton

E_l0 = E2 * (1 + 5 * alpha**2 / 16)  # l=0, j=1/2 zero-field energy
E_l1 = E2 * (1 + alpha**2 / 16)      # l=1, j=3/2 and j=1/2 zero-field energy

# (label, zero-field energy, slope in units of mu_B, line style group)
lines = [
    (r"$|2\,0\,\frac{1}{2}\,\frac{1}{2}\rangle$",    E_l0, 1,    "l0"),
    (r"$|2\,0\,\frac{1}{2}\,-\frac{1}{2}\rangle$",   E_l0, -1,   "l0"),
    (r"$|2\,1\,\frac{3}{2}\,\frac{3}{2}\rangle$",    E_l1, 2,    "j32"),
    (r"$|2\,1\,\frac{3}{2}\,\frac{1}{2}\rangle$",    E_l1, 2/3,  "j32"),
    (r"$|2\,1\,\frac{3}{2}\,-\frac{1}{2}\rangle$",   E_l1, -2/3, "j32"),
    (r"$|2\,1\,\frac{3}{2}\,-\frac{3}{2}\rangle$",   E_l1, -2,   "j32"),
    (r"$|2\,1\,\frac{1}{2}\,\frac{1}{2}\rangle$",    E_l1, 1/3,  "j12"),
    (r"$|2\,1\,\frac{1}{2}\,-\frac{1}{2}\rangle$",   E_l1, -1/3, "j12"),
]

colors = {"l0": "tab:red", "j32": "tab:blue", "j12": "tab:green"}

B = np.linspace(0, 4, 200)  # Tesla

fig, ax = plt.subplots(figsize=(9, 6.5))

for label, E0, slope, group in lines:
    dE = (E0 - E2) + slope * mu_B * B  # shift relative to unperturbed E2, in eV
    ax.plot(B, dE * 1e3, color=colors[group], lw=1.8)
    from fractions import Fraction
    frac = Fraction(slope).limit_denominator(3)
    slope_label = f"{frac.numerator:+d}" if frac.denominator == 1 else f"{frac.numerator:+d}/{frac.denominator}"
    ax.annotate(
        f"{label}   slope $= {slope_label}\\,\\mu_B$",
        xy=(B[-1], dE[-1] * 1e3),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=colors[group],
    )

ax.set_xlim(0, 4)
ax.set_xlabel(r"$B_{\rm ext}$  (T)")
ax.set_ylabel(r"$E - E_2$  (meV)")
ax.set_title("n=2 hydrogen fine structure + weak-field Zeeman splitting")
ax.axhline(0, color="gray", lw=0.5, ls=":")

# legend for the three zero-field groups
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=colors["l0"], lw=1.8, label=r"$l=0,\ j=\frac{1}{2}$  (5$\alpha^2$/16 shift)"),
    Line2D([0], [0], color=colors["j32"], lw=1.8, label=r"$l=1,\ j=\frac{3}{2}$  ($\alpha^2$/16 shift)"),
    Line2D([0], [0], color=colors["j12"], lw=1.8, label=r"$l=1,\ j=\frac{1}{2}$  ($\alpha^2$/16 shift)"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=9)

plt.tight_layout()
plt.subplots_adjust(right=0.72)
plt.savefig("prob7.24_zeeman.png", dpi=150)
plt.show()
print("Plot saved to prob7.24_zeeman.png")

print("\nZero-field energies:")
print(f"  E(l=0, j=1/2) = {E_l0:.6f} eV   (shift = {(E_l0 - E2)*1e3:.4f} meV)")
print(f"  E(l=1, j=3/2 or 1/2) = {E_l1:.6f} eV   (shift = {(E_l1 - E2)*1e3:.4f} meV)")
print("\nSlopes (in units of mu_B, mu_B = 5.7884e-5 eV/T):")
for label, E0, slope, group in lines:
    from fractions import Fraction
    frac = Fraction(slope).limit_denominator(3)
    slope_label = f"{frac.numerator:+d}" if frac.denominator == 1 else f"{frac.numerator:+d}/{frac.denominator}"
    print(f"  {label:45s} slope = {slope_label} mu_B")
