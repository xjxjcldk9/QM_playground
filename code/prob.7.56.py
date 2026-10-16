"""
Problem 7.56: evaluate

    [ 34a^2/81 - e^{-3pi/a} * ( (162a^4 + 162*pi*a^3 - 81*pi^2*a^2 + 81*pi^3*a) e^{2pi/a}
                                 - 26a^4 - 78*pi*a^3 + 45*pi^2*a^2 + 45*pi^3*a - 27*pi^4 ) / (324a^2) ]
    ------------------------------------------------------------------------------------------------
                                                2a^3

where a is the Bohr radius. Multiplying e^{-3pi/a} into the bracket collapses the two
exponentials into e^{-pi/a} and e^{-3pi/a}, and the whole mess simplifies to a clean
closed form: a pure 1/a term plus two exponentially small corrections.
"""
import sympy as sp
from scipy.constants import physical_constants, e as q, epsilon_0

a = sp.symbols('a', positive=True)
pi = sp.pi

inner = ((162 * a**4 + 162 * pi * a**3 - 81 * pi**2 * a**2 + 81 * pi**3 * a) * sp.exp(2 * pi / a)
         - 26 * a**4 - 78 * pi * a**3 + 45 * pi**2 * a**2 + 45 * pi**3 * a - 27 * pi**4)

expr = (sp.Rational(34, 81) * a**2 - sp.exp(-3 * pi / a) * inner / (324 * a**2)) / (2 * a**3)

# Collect onto the exp(-pi/a), exp(-3pi/a) basis.
e1, e3 = sp.exp(-pi / a), sp.exp(-3 * pi / a)
expanded = sp.expand(expr.rewrite(sp.exp))
c0 = sp.simplify(expanded.coeff(e1, 0).coeff(e3, 0))
c1 = sp.factor(expanded.coeff(e1))
c3 = sp.factor(expanded.coeff(e3))
closed = c0 + c1 * e1 + c3 * e3

print("Closed form:")
print(f"  {c0} + ({c1}) * exp(-pi/a) + ({c3}) * exp(-3*pi/a)")
print("Matches original expression:", sp.simplify(expr - closed) == 0)

print()
print("The exponential terms are corrections to the leading 17/(81a) piece;")
print("they vanish as a -> 0 (i.e. as pi/a -> infinity).")

print()
print("Value in units of the Bohr radius (a = 1):")
val = closed.subs(a, 1)
print(f"  exact: {val}")
print(f"  numeric: {sp.N(val, 15)}")

a0, unit, _ = physical_constants['Bohr radius']  # CODATA value, in meters
a0 = sp.Float(a0, 15)
print()
print(f"Value with a substituted by the actual Bohr radius (a = {a0} {unit}):")
val_si = closed.subs(a, a0)
print(f"  exact substitution: {sp.N(val_si, 15)} {unit}^-1")
print(f"  (17/(81*a) leading term alone: {sp.N(c0.subs(a, a0), 15)} {unit}^-1;"
      f" exponential corrections are ~exp(-pi/a) = {sp.N(sp.exp(-pi / a0), 6)}, utterly negligible)")

print()
print("Times q^2/(4*pi*epsilon_0) * 2, in eV:")
prefactor = sp.Float(q, 15)**2 / (4 * pi * sp.Float(epsilon_0, 15)) * 2  # J*m
energy_J = prefactor * val_si  # J
energy_eV = energy_J / sp.Float(q, 15)
print(f"  q^2/(4*pi*epsilon_0) = {sp.N(sp.Float(q, 15)**2 / (4 * pi * sp.Float(epsilon_0, 15)), 10)} J*m")
print(f"  energy = {sp.N(energy_J, 15)} J = {sp.N(energy_eV, 15)} eV")
