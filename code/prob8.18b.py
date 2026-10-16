"""
Problem 8.18(b): minimize the variational energy over b^2.

The integration (done separately) gives <H> as the four terms

    <H> = (hbar^2/2m) (2n-1)^2      (2n-3/2)(3/2)       / (b^2 (2n+1)(2n))
        - (hbar^2/2m) 2 (2n-1)      (2n-3/2)(2n-1/2)    / (b^2 (2n+1)(2n))
        + (hbar^2/2m) 2             (2n-3/2)(2n-1/2)(2n+1/2) / (b^2 (2n+1)(2n))
        + (1/2) m omega^2  b^2 (3/2) / (2n - 5/2)

which has the form  A / u + B u  with  u = b^2:

    A = sum of the three (hbar^2/2m)... coefficients (without 1/b^2)
    B = (1/2) m omega^2 (3/2) / (2n - 5/2)

Minimizing over u:  d/du (A/u + B u) = -A/u^2 + B = 0  =>  u = sqrt(A/B),
minimum value  2 sqrt(A B).  n is kept algebraic throughout.
"""
import sympy as sp

n = sp.symbols('n', positive=True)
hbar, m, omega, u = sp.symbols('hbar m omega u', positive=True)   # u = b^2

# ---- coefficients (u = b^2 factored out) --------------------------------
c = (hbar**2 / (2 * m)) / ((2 * n + 1) * (2 * n))
A = sp.simplify(
    c * (2 * n - 1)**2 * (2 * n - sp.Rational(3, 2)) * sp.Rational(3, 2)
    - c * 2 * (2 * n - 1) * (2 * n - sp.Rational(3, 2)) * (2 * n - sp.Rational(1, 2))
    + c * 2 * (2 * n - sp.Rational(3, 2)) * (2 * n - sp.Rational(1, 2)) * (2 * n + sp.Rational(1, 2))
)
B = sp.simplify(sp.Rational(1, 2) * m * omega**2 * sp.Rational(3, 2)
                / (2 * n - sp.Rational(5, 2)))

print("A =", A)
print("  =", sp.factor(A))
print("B =", B)

# ---- the function to minimize -------------------------------------
f = A / u + B * u
print("\n<H>(u) =", f, "      (u = b^2)")

# ---- minimize over u = b^2 --------------------------------------
df = sp.diff(f, u)
print("d<H>/du =", sp.simplify(df))

roots = sp.solve(sp.Eq(df, 0), u)
# valid where A, B > 0 (needs n > 5/4); pick the positive root of u^2 = A/B
u_min = [r for r in roots if r.subs(n, 2).is_positive][0]
u_min = sp.simplify(sp.radsimp(u_min))
print("\nu_min = b^2_min =", u_min)
print("               =", sp.factor(u_min))

d2f = sp.simplify(sp.diff(f, u, 2).subs(u, u_min))
print("d2<H>/du2 at u_min =", d2f, " (>0 => minimum)")

# ---- minimum energy --------------------------------------------
H_min = sp.simplify(sp.radsimp(f.subs(u, u_min)))
H_min_alt = sp.simplify(2 * sp.sqrt(A * B))          # = 2 sqrt(A B)
print("\n<H>_min =", H_min)
print("<H>_min =", H_min_alt, "  (as 2*sqrt(A*B))")
ratio2 = sp.simplify((H_min_alt / (hbar * omega))**2)
print("<H>_min = hbar*omega * sqrt(", ratio2, ")")
print("        = hbar*omega * sqrt(", sp.factor(ratio2), ")")

# ---- LaTeX --------------------------------------------------------
print("\n--- LaTeX ---")
print(r"A &= " + sp.latex(sp.factor(A)) + r" \\")
print(r"B &= " + sp.latex(B) + r" \\")
print(r"\langle H\rangle(u) &= " + sp.latex(f) + r" \qquad (u=b^2)\\")
print(r"\frac{d\langle H\rangle}{du} &= " + sp.latex(sp.simplify(df)) + r" \\")
print(r"b^2_{\min} = u_{\min} &= " + sp.latex(sp.factor(u_min)) + r" \\")
print(r"\langle H\rangle_{\min} &= " + sp.latex(H_min_alt) + r" \\")
print(r"&= \hbar\omega\,\sqrt{" + sp.latex(sp.factor(ratio2)) + r"}")

# ---- checks --------------------------------------------------------
print("\n--- checks ---")
print("A, B > 0 (real minimum) requires n > 5/4  ->  n = 2, 3, ...")
for nv in (2, 3, 5, 10):
    if B.subs(n, nv) > 0:
        val = sp.simplify((H_min_alt / (hbar * omega)).subs(n, nv))
        bval = sp.simplify((u_min * m * omega / hbar).subs(n, nv))
        print(f"  n = {nv:2d}:  <H>_min/(hbar omega) = {val} = {sp.N(val, 10)}"
              f",   b^2_min = ({bval}) hbar/(m omega)")
print("n -> oo :  <H>_min/(hbar omega) ->",
      sp.limit(sp.sqrt(ratio2), n, sp.oo))
