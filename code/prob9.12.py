"""
Problem 9.12: WKB wavefunction for the n = 5 harmonic-oscillator state,
compared to the exact solution.

Units: hbar = m = omega = 1, so V(x) = x^2/2 and E_n = n + 1/2.
For n = 5, E = 11/2, and the classical turning point is

    x_2 = sqrt(2E) = sqrt(11)

By symmetry it's enough to work on x > 0 (n = 5 is odd, so psi is odd
under x -> -x). With

    p(x) = sqrt(x_2^2 - x^2)        (0 < x < x_2, classically allowed)
    p(x) = sqrt(x^2 - x_2^2)        (x_2 < x,      classically forbidden)

the standard WKB connection formulas at a "soft" turning point give

    psi(x) = (2D/sqrt(p(x))) sin[ (1/hbar) int_x^{x_2} p(x') dx'  + pi/4 ],   0 < x < x_2
    psi(x) = (D/sqrt(p(x)))  exp[ -(1/hbar) int_{x_2}^x p(x') dx' ],         x > x_2

but these two asymptotic forms individually blow up at x = x_2 itself (p -> 0
there). The fix is the uniform Airy-function approximation: both forms above
are just the large-|z| asymptotics of the Airy function Ai(z), which is
smooth and finite everywhere -- including through the turning point. Define

    z(x) = -[ (3/2) int_x^{x_2} p dx' ]^(2/3)   for x <= x_2   (z <= 0)
    z(x) = +[ (3/2) int_{x_2}^x p dx' ]^(2/3)   for x >= x_2   (z >= 0)

Then

    psi(x) = D * sqrt(4 pi) * Ai(z(x)) / sqrt(|dz/dx|)

reduces to the two forms above far from x_2 (checked by matching the large-
|z| asymptotics of Ai), while staying finite at x_2: dz/dx stays nonzero
there even though p(x) vanishes. Because p(x) = sqrt(x_2^2 - x^2) has no
elementary antiderivative in closed power-law form (it integrates to an
arcsin, not a 3/2 power), the phase/decay integrals and dz/dx are evaluated
numerically rather than symbolically.

D is fixed by requiring int_0^infty psi(x)^2 dx = 1/2, i.e. the x > 0 half of
a properly normalized antisymmetric full-line eigenfunction -- matching the
convention of the exact psi_5 below when it's restricted to x > 0, so the two
curves can be compared on equal footing.

Comparison (exact n = 5 oscillator eigenfunction):

    psi(x) = (1/pi)^(1/4) * 1/sqrt(2^5 5!) * (32x^5 - 160x^3 + 120x) * exp(-x^2/2)
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import airy, factorial
import matplotlib.pyplot as plt

n = 5
E = n + 0.5
x2 = np.sqrt(2 * E)          # = sqrt(11)

xmax = 7.0

# --- classically allowed region: 0 <= x <= x2 ----------------------------
x_a = np.linspace(0, x2, 3000)
p_a = np.sqrt(x2**2 - x_a**2)
cum_a = cumulative_trapezoid(p_a, x_a, initial=0.0)   # int_0^x p dx'
phase_a = cum_a[-1] - cum_a                            # int_x^{x2} p dx'
z_a = -((1.5 * phase_a) ** (2 / 3))

# --- classically forbidden region: x2 <= x <= xmax -----------------------
x_f = np.linspace(x2, xmax, 3000)
p_f = np.sqrt(x_f**2 - x2**2)
decay_f = cumulative_trapezoid(p_f, x_f, initial=0.0)  # int_{x2}^x p dx'
z_f = (1.5 * decay_f) ** (2 / 3)

# dz/dx = p(x)/sqrt(|z(x)|) on both branches (from dz/dS = 1/sqrt(|z|) and
# dS/dx = +-p(x)); this has a finite limit (2 x2)^(1/3) as x -> x2, where the
# 0/0 form must be replaced by hand.
with np.errstate(invalid="ignore"):
    dzdx_a = p_a / np.sqrt(np.abs(z_a))
    dzdx_f = p_f / np.sqrt(np.abs(z_f))
dzdx_a[-1] = dzdx_f[0] = (2 * x2) ** (1 / 3)

# --- stitch into one smooth curve through the turning point --------------
x_full = np.concatenate([x_a, x_f[1:]])
z_full = np.concatenate([z_a, z_f[1:]])
dzdx = np.concatenate([dzdx_a, dzdx_f[1:]])

Ai, _, _, _ = airy(z_full)
psi_unnorm = np.sqrt(4 * np.pi) * Ai / np.sqrt(np.abs(dzdx))

# --- normalize: int_0^infty psi^2 dx = 1/2 --------------------------------
I_unnorm = np.trapezoid(psi_unnorm**2, x_full)
D = np.sqrt(0.5 / I_unnorm)

psi_wkb = D * psi_unnorm


def psi_exact(x):
    H5 = 32 * x**5 - 160 * x**3 + 120 * x
    return (1 / np.pi) ** 0.25 / np.sqrt(2**5 * factorial(5)) * H5 * np.exp(-x**2 / 2)


psi_ex_full = psi_exact(x_full)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_full, psi_wkb, color="tab:blue", lw=1.8, label="WKB (uniform, Airy)")
ax.plot(x_full, psi_ex_full, "--", color="tab:orange", lw=1.6, label="exact (n = 5)")
ax.axvline(x2, color="gray", ls=":", lw=1)
ax.text(x2, 0.85, r"$x_2=\sqrt{11}$", color="gray", ha="left", va="top")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("x")
ax.set_ylabel(r"$\psi(x)$")
ax.set_title("Problem 9.12: WKB vs exact wavefunction, harmonic oscillator n = 5 (x > 0)")
ax.set_xlim(0, xmax)
ax.set_ylim(-0.6, 0.6)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob9.12.png", dpi=150)
print("saved prob9.12.png")
print(f"x2 = sqrt(11) = {x2:.6f}")
print(f"D  = {D:.6f}")
print(f"check: int_0^inf psi_wkb^2 dx = {D**2 * I_unnorm:.6f}  (target 0.5)")
