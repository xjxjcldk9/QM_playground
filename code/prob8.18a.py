"""
Problem 8.18(a): minimize the variational energy over b^2.

The trial-function calculation gives

    <H>(b) = (hbar^2 / 2m) * 4 n^2 * (2n - 1/2) / (2 b^2 (2n+1)(2n))
             + (1/2) m omega^2 b^2 * (1/2) * 1 / (2n - 3/2)

which has the form  A / u + B u  with  u = b^2:

    A = (hbar^2 / 2m) * 4 n^2 (2n - 1/2) / (2 (2n+1)(2n))
    B = (1/2) m omega^2 * (1/2) / (2n - 3/2)

Minimizing over u:  d/du (A/u + B u) = -A/u^2 + B = 0  =>  u = sqrt(A/B),
minimum value  2 sqrt(A B).  n is kept algebraic throughout.
"""
import sympy as sp

n = sp.symbols('n', positive=True)
hbar, m, omega, u = sp.symbols('hbar m omega u', positive=True)  # u = b^2

# ---- coefficients --------------------------------------------------------
A = (hbar**2 / (2 * m)) * 4 * n**2 * (2 * n - sp.Rational(1, 2)) \
    / (2 * (2 * n + 1) * (2 * n))
B = sp.Rational(1, 2) * m * omega**2 * sp.Rational(1, 2) / (2 * n - sp.Rational(3, 2))

A = sp.simplify(A)
B = sp.simplify(B)
print("A =", A)
print("B =", B)

# ---- the function to minimize -----------------------------------------
f = A / u + B * u
print("\n<H>(u) =", f, "      (u = b^2)")

# ---- minimize over u = b^2 ------------------------------------------
df = sp.diff(f, u)
print("d<H>/du =", sp.simplify(df))

roots = sp.solve(sp.Eq(df, 0), u)
# valid for n > 3/4 so that A, B > 0; pick the positive root of u^2 = A/B
u_min = [r for r in roots if r.subs(n, 1).is_positive][0]
u_min = sp.simplify(sp.radsimp(u_min))
print("\nu_min = b^2_min =", u_min)

d2f = sp.simplify(sp.diff(f, u, 2).subs(u, u_min))
print("d2<H>/du2 at u_min =", d2f, " (>0 => minimum)")

# ---- minimum energy -------------------------------------------------
H_min = sp.simplify(sp.radsimp(f.subs(u, u_min)))
H_min_alt = sp.simplify(2 * sp.sqrt(A * B))          # = 2 sqrt(A B)
print("\n<H>_min =", H_min)
print("<H>_min =", H_min_alt, "  (as 2*sqrt(A*B))")
print("<H>_min / (hbar*omega) =", sp.simplify(H_min / (hbar * omega)))

# ---- LaTeX output ------------------------------------------------------
print("\n--- LaTeX ---")
print(r"A &= " + sp.latex(A) + r" \\")
print(r"B &= " + sp.latex(B) + r" \\")
print(r"\langle H\rangle(u) &= " + sp.latex(f) + r" \qquad (u=b^2)\\")
print(r"\frac{d\langle H\rangle}{du} &= " + sp.latex(sp.simplify(df)) + r" \\")
print(r"b^2_{\min} = u_{\min} &= " + sp.latex(u_min) + r" \\")
print(r"\langle H\rangle_{\min} &= " + sp.latex(H_min_alt) + r" \\")
print(r"&= " + sp.latex(sp.sqrt(sp.simplify((H_min_alt / (hbar * omega))**2)))
      + r"\;\hbar\omega")

# ---- checks --------------------------------------------------------
print("\n--- checks ---")
print("n -> oo limit of <H>_min/(hbar omega):",
      sp.limit(sp.simplify(H_min / (hbar * omega)), n, sp.oo),
      "  (exact ground state = 1/2)")
for nv in (1, 2, 3, 10):
    val = sp.simplify((H_min / (hbar * omega)).subs(n, nv))
    print(f"  n = {nv:2d}:  <H>_min/(hbar omega) = {val} = {sp.N(val, 10)}")
