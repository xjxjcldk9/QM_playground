"""
Problem 8.14: simplify the messy exchange-type integral that comes out of the
variational calculation.  The starting expression is the sum of the two bulky
terms below (functions of R and the Bohr radius a); sympy collapses it to a
single compact form.
"""
import sympy as sp

R, a = sp.symbols('R a', positive=True)
e = sp.exp

term1 = e(-6 * R / a) * (
    24 * a**3 * e(6 * R / a)
    + (-36 * a**3 - 57 * R * a**2 - 36 * R**2 * a - 8 * R**3) * e(4 * R / a)
    + (24 * a**3 + 48 * R * a**2) * e(2 * R / a)
    - 12 * a**3 - 39 * R * a**2 - 24 * R**2 * a
) / (48 * R * a**2)

term2 = e(-6 * R / a) * (
    8 * a**2 * e(6 * R / a)
    + (-4 * a**2 - 3 * R * a) * e(4 * R / a)
    + (-8 * a**2 - 16 * R * a) * e(2 * R / a)
    + 4 * a**2 + 13 * R * a + 8 * R**2
) / (16 * R * a)

expr = term1 + term2

simplified = sp.simplify(expr)
single_fraction = sp.factor(sp.together(expr))

# distribute e^{-2R/a} and write in powers of x = R/a
distributed = a / R - (a / R + sp.Rational(11, 8)
                       + sp.Rational(3, 4) * (R / a)
                       + sp.Rational(1, 6) * (R / a)**2) * e(-2 * R / a)

print("simplify():")
print("   ", simplified)
print("    LaTeX:", sp.latex(simplified))
print()
print("single fraction:")
print("   ", single_fraction)
print("    LaTeX:", sp.latex(single_fraction))
print()
print("distributed form:")
print("   ", distributed)
print("    LaTeX:", sp.latex(distributed))
print()
print("check distributed - expr =", sp.simplify(distributed - expr))

# numeric spot check
subs = {a: 1.0, R: 1.7}
print("numeric check at a=1, R=1.7:",
      complex(expr.subs(subs)).real, "==", complex(distributed.subs(subs)).real)
