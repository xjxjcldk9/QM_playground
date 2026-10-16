"""
Problem 8.16: minimize the variational energy over the exponent p.

    f(p) = a^{4p-1} p^2 (4 B(2p-1,2p+1) + B(2p-1,2p-1) - 4 B(2p,2p-1))
           -------------------------------------------------------------
                        a^{4p+1} B(2p+1,2p+1)

The a-powers cancel: a^{4p-1}/a^{4p+1} = 1/a^2, so

    f(p) = (1/a^2) * g(p),   g(p) = p^2 * N(p) / D(p)
    N(p) = 4 B(2p-1,2p+1) + B(2p-1,2p-1) - 4 B(2p,2p-1)
    D(p) = B(2p+1,2p+1)

B(x,y) needs x,y > 0, so B(2p-1,2p-1) requires p > 1/2.
Minimizing f over p is the same as minimizing g (1/a^2 > 0 is a fixed factor).
"""
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
from scipy.special import beta

# ---- symbolic --------------------------------------------------------------
p = sp.symbols('p', positive=True)
B = sp.beta
N = 4 * B(2 * p - 1, 2 * p + 1) + B(2 * p - 1, 2 * p - 1) - 4 * B(2 * p, 2 * p - 1)
D = B(2 * p + 1, 2 * p + 1)
g = p**2 * N / D
g = sp.simplify(sp.rewrite(g, sp.gamma) if hasattr(sp, "rewrite")
                else g.rewrite(sp.gamma))
g = sp.simplify(g.rewrite(sp.gamma))
print("g(p) =", g)

dg = sp.simplify(sp.diff(g, p))
roots = sp.solve(sp.Eq(dg, 0), p)
p_star = [r for r in roots if r.is_real and r > sp.Rational(1, 2)][0]
p_star = sp.nsimplify(sp.radsimp(p_star))
g_star = sp.simplify(sp.radsimp(g.subs(p, p_star)))
print(f"stationary point:  p* = {p_star} = {sp.N(p_star, 12)}")
print(f"g(p*)             = {g_star} = {sp.N(g_star, 12)}   (units of 1/a^2)")
print(f"second derivative = {sp.N(sp.diff(g, p, 2).subs(p, p_star), 8)}  (>0 => min)")

# ---- numeric cross-check -------------------------------------------------
def g_num(pp):
    Nn = (4 * beta(2 * pp - 1, 2 * pp + 1)
          + beta(2 * pp - 1, 2 * pp - 1)
          - 4 * beta(2 * pp, 2 * pp - 1))
    Dn = beta(2 * pp + 1, 2 * pp + 1)
    return pp**2 * Nn / Dn

res = minimize_scalar(g_num, bounds=(0.51, 5.0), method="bounded",
                      options={"xatol": 1e-12})
print()
print(f"numeric:  p* = {res.x:.10f},  g(p*) = {res.fun:.10f} / a^2")
print(f"compare exact hydrogen/well value pi^2/8 = {np.pi**2 / 8:.10f}")
