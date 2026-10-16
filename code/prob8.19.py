"""
Problem 8.19: Variational estimate of the hydrogen ground-state energy
using a normalized Gaussian trial wave function  psi(r) = A exp(-b r^2).

<T> and <V>, with the normalization A^2 = (2b/pi)^(3/2) already folded in,
combine into the expectation value of the Hamiltonian

    E(b) = (hbar^2/2m) * 4*pi * [3*sqrt(pi) / (2^(7/2) sqrt(b))] * (2b/pi)^(3/2)
           - (e^2/epsilon_0) * (1/(4b)) * (2b/pi)^(3/2)

Task: minimize E(b) over b algebraically (dE/db = 0), then evaluate the
resulting minimum energy numerically in eV.
"""
import sympy as sp

hbar, m, e, eps0, b = sp.symbols('hbar m e epsilon_0 b', positive=True)

# --- E(b) exactly as given ----------------------------------------------------
T = (hbar**2 / (2 * m)) * 4 * sp.pi * (3 * sp.sqrt(sp.pi)) / \
    (2**sp.Rational(7, 2) * sp.sqrt(b)) * (2 * b / sp.pi)**sp.Rational(3, 2)
V = (e**2 / eps0) * (sp.Rational(1, 4) / b) * (2 * b / sp.pi)**sp.Rational(3, 2)
E = sp.simplify(T - V)

print("<T>(b) =", sp.latex(sp.simplify(T)))
print("<V>(b) =", sp.latex(-sp.simplify(V)))
print("E(b)   =", sp.latex(E))
print()

# --- minimize algebraically ---------------------------------------------------
dE = sp.simplify(sp.diff(E, b))
print("dE/db  =", sp.latex(dE))

b_min = [s for s in sp.solve(sp.Eq(dE, 0), b) if s.is_positive][0]
b_min = sp.simplify(b_min)
print("b_min  =", sp.latex(b_min))

d2E = sp.diff(E, b, 2)
print("E''(b_min) > 0 ? :", bool(sp.simplify(d2E.subs(b, b_min)) > 0))

E_min = sp.simplify(E.subs(b, b_min))
print("E_min  =", sp.latex(E_min))
print()

# compare to the exact ground-state energy  E1 = -m e^4 / (2 (4 pi eps0)^2 hbar^2)
E1 = -m * e**4 / (2 * (4 * sp.pi * eps0)**2 * hbar**2)
ratio = sp.simplify(E_min / E1)
print(f"E_min / E1 = {ratio}  (exact = 8/(3*pi))")
print()

# --- plug in the numbers (SI units, CODATA values) -----------------------------
consts = {
    hbar: 1.054571817e-34,   # J s
    m:    9.1093837015e-31,  # electron mass, kg
    e:    1.602176634e-19,   # elementary charge, C
    eps0: 8.8541878128e-12,  # F/m
}
eV = 1.602176634e-19        # J

b_min_num = float(b_min.subs(consts))
E_min_num_J = float(E_min.subs(consts))
E_min_num_eV = E_min_num_J / eV
E1_num_eV = float(E1.subs(consts)) / eV

print(f"b_min   = {b_min_num:.6e}  1/m^2")
print(f"E_min   = {E_min_num_J:.6e} J  =  {E_min_num_eV:.4f} eV")
print(f"(exact ground-state energy E1 = {E1_num_eV:.4f} eV, "
      f"variational bound satisfied: E_min > E1)")
