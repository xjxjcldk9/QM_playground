"""
Problem 7.53: does the implicit equation

    v*sinh(v) + Lambda*sinh^2(v/2) = 0

have a nonzero solution for v?

For real v, v*sinh(v) >= 0 (v and sinh(v) always share a sign) and
sinh^2(v/2) >= 0, so for Lambda >= 0 the sum can only vanish at the trivial
point v = 0. A nontrivial root requires Lambda < 0. Using
sinh(v) = 2 sinh(v/2) cosh(v/2) and dividing out one power of sinh(v/2)
(nonzero for v != 0) turns f(v) = 0 into the explicit curve

    Lambda(v) = -2 v coth(v/2)

As v -> 0, v*coth(v/2) -> 2, so Lambda(v) -> -4, and Lambda(v) decreases
monotonically (more negative) as v grows. So the range of Lambda(v) for
v > 0 is (-inf, -4): a nontrivial solution exists only for Lambda < -4;
for -4 <= Lambda (which includes Lambda = 0 and every positive Lambda)
v = 0 is the only root. This script plots f(v) for several
Lambda straddling the critical value -4, and separately plots the
Lambda(v) bifurcation curve to show the -4 asymptote directly.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(v, L):
    return v * np.sinh(v) + L * np.sinh(v / 2) ** 2

def bisect(func, lo, hi, tol=1e-12):
    flo, fhi = func(lo), func(hi)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)

v = np.linspace(1e-3, 6, 2000)

Lambdas = [-2, -3, -4, -4.5, -5, -6, -8]
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(Lambdas)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

from matplotlib.lines import Line2D
legend_elems = []
for L, color in zip(Lambdas, cmap):
    y = f(v, L)
    ax1.plot(v, y, color=color, lw=1.8)
    root = bisect(lambda x: f(x, L), 0.05, 6.0) if L < -4 else None
    tag = f"$v^*={root:.3f}$" if root else "no root"
    legend_elems.append(Line2D([0], [0], color=color, lw=1.8, label=f"$\\Lambda={L:g}$  ({tag})"))
    if root:
        ax1.plot(root, 0, "o", color=color, ms=6, zorder=5)

ax1.axhline(0, color="gray", lw=0.8, ls=":")
ax1.set_xlim(0, 6)
ax1.set_xlabel("$v$")
ax1.set_ylabel(r"$f(v)=v\sinh v+\Lambda\sinh^2(v/2)$")
ax1.set_title("$f(v)$ for $\\Lambda$ straddling the critical value $-4$")
ax1.set_ylim(-30, 60)
ax1.legend(handles=legend_elems, fontsize=8, loc="upper left")

vv = np.linspace(1e-3, 5, 2000)
Lam_v = -2 * vv / np.tanh(vv / 2)
ax2.plot(vv, Lam_v, color="tab:blue", lw=2, label=r"$\Lambda(v)=-2v\coth(v/2)$")
ax2.axhline(-4, color="tab:red", lw=1.2, ls="--", label=r"$\Lambda=-4$ (critical value)")
ax2.set_xlim(0, 5)
ax2.set_ylim(-30, 5)
ax2.set_xlabel("$v$ (nontrivial root)")
ax2.set_ylabel(r"$\Lambda$ that makes $v$ a root")
ax2.set_title("Bifurcation curve: root exists only for $\\Lambda<-4$")
ax2.legend(fontsize=8.5, loc="lower left")

plt.tight_layout()
plt.savefig("prob7.53_implicit.png", dpi=150)
plt.show()
print("Plot saved to prob7.53_implicit.png")

print("\nCritical value: Lambda_c = -4 (exact, from lim v->0 of v*coth(v/2) = 2)")
print("For Lambda >= -4 (including Lambda = 0 and every positive Lambda) v = 0 is the only solution.")
print("For Lambda < -4 a nontrivial root appears; roots found by bisection:")
for L in Lambdas:
    root = bisect(lambda x: f(x, L), 0.05, 6.0) if L < -4 else None
    print(f"  Lambda = {L:>5g}:  v* = {root:.6f}" if root else f"  Lambda = {L:>5g}:  no nonzero root (only v=0)")
