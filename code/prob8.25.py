"""
Problem 8.25: Helium ground-state variational estimate using a trial
wavefunction built from two different (unequal) screening parameters
Z1 and Z2 for the two electrons (symmetrized hydrogenic orbitals),
psi(r1,r2) ~ [e^{-Z1 r1/a} e^{-Z2 r2/a} + e^{-Z2 r1/a} e^{-Z1 r2/a}].

Carrying out <H> for this trial state and writing it in terms of

    x = Z1 + Z2,        y = 2*sqrt(Z1*Z2)

gives (Griffiths' result, with E1 = -13.6 eV the hydrogen ground energy)

    <H> = E1/(x^6 + y^6) * [ -x^8 + (1/2) x^6 y^2 - (1/2) y^8
                              + 2 x^7 - (1/2) y^2 x^5 - (1/8) y^4 x^3
                              + (11/8) x y^6 ]

Task: evaluate <H> at the optimized screening parameters
Z1 = 1.039, Z2 = 0.283 (the values Griffiths quotes for the minimum).
"""
import sympy as sp

Z1_val, Z2_val = sp.Rational('1.039'), sp.Rational('0.283')
E1_eV = sp.Rational('-13.6')  # hydrogen ground state energy, in eV

x, y, E1, Z1, Z2 = sp.symbols('x y E1 Z1 Z2')

expr = E1 / (x**6 + y**6) * (
    -x**8 + sp.Rational(1, 2) * x**6 * y**2 - sp.Rational(1, 2) * y**8
    + 2 * x**7
    - sp.Rational(1, 2) * y**2 * x**5
    - sp.Rational(1, 8) * y**4 * x**3
    + sp.Rational(11, 8) * x * y**6
)

x_expr = Z1 + Z2
y_expr = 2 * sp.sqrt(Z1 * Z2)

print("<H>(x,y) =", sp.nsimplify(expr))
print()
print(f"Z1 = {Z1_val}, Z2 = {Z2_val}")

x_val = x_expr.subs({Z1: Z1_val, Z2: Z2_val})
y_val = y_expr.subs({Z1: Z1_val, Z2: Z2_val})
print(f"x = Z1 + Z2       = {sp.N(x_val, 10)}")
print(f"y = 2*sqrt(Z1*Z2) = {sp.N(y_val, 10)}")
print()

H_val = expr.subs({x: x_val, y: y_val, E1: E1_eV})
H_val_num = sp.N(H_val, 10)

print(f"<H> = {H_val_num} eV")
print(f"<H> ~ {float(H_val_num):.4f} eV")
