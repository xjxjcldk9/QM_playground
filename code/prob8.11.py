"""
Problem 8.11: Vibrational curvature of the H2+ molecular ion.

F(x) is the total energy (electron + proton repulsion) in units of -E1 = 13.6 eV,
with x = R/a the proton separation in Bohr radii (Griffiths Eq. 8.79):

    F(x) = -1 + (2/x) * { [(1 - (2/3)x^2) e^{-x} + (1 + x) e^{-2x}]
                          / [1 + (1 + x + (1/3)x^2) e^{-x}] }

Task: Taylor-expand F to second order about its minimum x0,

    F(x) ~ F(x0) + (1/2) F''(x0) (x - x0)^2      (F'(x0) = 0)

and report x0, F(x0), and the curvature F''(x0) numerically.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def F(x):
    num = (1 - (2 / 3) * x**2) * np.exp(-x) + (1 + x) * np.exp(-2 * x)
    den = 1 + (1 + x + (1 / 3) * x**2) * np.exp(-x)
    return -1 + (2 / x) * num / den


# --- minimum -----------------------------------------------------------------
res = minimize_scalar(F, bounds=(1.0, 5.0), method="bounded",
                      options={"xatol": 1e-12})
x0 = res.x
F0 = F(x0)

# --- derivatives by central finite differences ------------------------------
h = 1e-5
Fp = (F(x0 + h) - F(x0 - h)) / (2 * h)
Fpp = (F(x0 + h) - 2 * F(x0) + F(x0 - h)) / h**2

print(f"x0        = {x0:.6f}        (R = {x0*0.529:.4f} angstrom)")
print(f"F(x0)     = {F0:.6f}        ->  E_bind = {(-1 - F0)*13.6:.4f} eV below (H + p)")
print(f"F'(x0)    = {Fp:.3e}   (~0, confirms minimum)")
print(f"F''(x0)   = {Fpp:.6f}")
print()
print("Second-order Taylor expansion about the minimum:")
print(f"    F(x) ~ {F0:.6f} + {0.5*Fpp:.6f} * (x - {x0:.6f})**2")

# --- sanity check: compare Taylor vs exact near x0 -------------------------
print()
print("   x      F_exact     F_taylor")
for x in np.linspace(x0 - 0.6, x0 + 0.6, 7):
    taylor = F0 + 0.5 * Fpp * (x - x0) ** 2
    print(f" {x:5.3f}  {F(x): .6f}  {taylor: .6f}")

# --- vibrational frequency -------------------------------------------------
# The Taylor term is (1/2) F''(x0) (x - x0)^2 in units of |E1|, with x = R/a.
# As a potential in R:  V(R) = |E1| * c * (R - R0)^2 / a^2 ,  c = (1/2) F''(x0).
# Matching (1/2) m omega^2 (R - R0)^2  =>  (1/2) m omega^2 = |E1| * c / a^2.
c = 0.5 * Fpp                       # 0.062834
E1 = 13.6 * 1.602176634e-19         # |E1| in joules
a0 = 5.29177210903e-11             # Bohr radius, m
m_p = 1.67262192369e-27            # proton mass, kg
hbar = 1.054571817e-34

mu = m_p / 2                        # reduced mass of the two-proton system

k = 2 * c * E1 / a0**2              # spring constant, N/m
omega = np.sqrt(k / mu)            # rad/s, using the reduced mass
eV = 1.602176634e-19

print()
print(f"(1/2) mu omega^2 a0^2 = |E1| * c ,  c = (1/2) F''(x0) = {c:.6f}")
print(f"reduced mass     mu = m_p/2 = {mu:.4e} kg")
print(f"spring constant  k  = {k:.3f} N/m")
print(f"omega              = {omega:.4e} rad/s")
print(f"nu = omega/2pi     = {omega/(2*np.pi):.4e} Hz")
print(f"hbar*omega         = {hbar*omega/eV*1e3:.2f} meV")
print(f"(1/2) hbar*omega   = {0.5*hbar*omega/eV*1e3:.2f} meV  = {0.5*hbar*omega/eV:.4f} eV"
      f"  (zero-point energy)")

# --- how many bound vibrational levels: (1/2 + n) hbar omega <= well depth --
D_well = 1.8                        # eV, well depth (~ binding energy)
hw = hbar * omega / eV             # hbar*omega in eV
n_max = int(np.floor(D_well / hw - 0.5))
print()
print(f"well depth ~ {D_well} eV,  hbar*omega = {hw:.4f} eV")
print(f"(1/2 + n) hbar*omega <= {D_well} eV  ->  n <= {D_well/hw - 0.5:.3f}")
print(f"n = 0 .. {n_max}   ->  about {n_max + 1} bound vibrational levels")
