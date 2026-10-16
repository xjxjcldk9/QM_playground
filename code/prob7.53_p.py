"""
Problem 7.53 (continued): split v asymmetrically with a parameter p in (0,1):

    v*sinh(v) + Lambda*sinh(p*v)*sinh((1-p)*v) = 0

(p = 1/2 recovers the symmetric v*sinh(v) + Lambda*sinh^2(v/2) = 0 case
from before, whose critical value was Lambda_c = -4.)

Small-v expansion: v*sinh(v) ~ v^2 and sinh(pv)*sinh((1-p)v) ~ p(1-p) v^2,
so near v = 0

    f(v) ~ v^2 * [1 + Lambda*p(1-p)]

A nontrivial root bifurcates off v = 0 only once this bracket goes
negative:

    1 + Lambda*p(1-p) < 0   <=>   Lambda*p(1-p) < -1
                            <=>   |Lambda| > 1/[p(1-p)]      (Lambda < 0)

Since p(1-p) <= 1/4 (maximum at p = 1/2), no p at all works unless
|Lambda| > 4, and then solutions exist only for p inside the symmetric
window bounded by p(1-p) = 1/|Lambda|, i.e.

    p_edge = [1 -+ sqrt(1 - 4/|Lambda|)] / 2

This script solves f(v, p, Lambda) = 0 for v(p) at Lambda = -4.1, -5, -10
and overlays the predicted window edges on the numerically found ones.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(v, p, L):
    return v * np.sinh(v) + L * np.sinh(p * v) * np.sinh((1 - p) * v)

def bisect(func, lo, hi):
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

def p_window(L):
    disc = 1 - 4 / abs(L)
    if disc < 0:
        return None
    s = np.sqrt(disc)
    return (1 - s) / 2, (1 + s) / 2

Lambdas = [-4.1, -5, -10]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Lambdas)))
ps = np.linspace(0.001, 0.999, 2000)

fig, ax = plt.subplots(figsize=(8, 6))

for L, color in zip(Lambdas, colors):
    vs = np.array([bisect(lambda v: f(v, p, L), 1e-4, 30.0) for p in ps], dtype=float)
    ax.plot(ps, vs, color=color, lw=2, label=f"$\\Lambda={L:g}$")
    p1, p2 = p_window(L)
    ax.axvline(p1, color=color, lw=0.8, ls=":")
    ax.axvline(p2, color=color, lw=0.8, ls=":")

ax.axvline(0.5, color="gray", lw=0.6, ls="-")
ax.set_xlabel("$p$")
ax.set_ylabel("$v(p)$  (nontrivial root)")
ax.set_title(r"$v(p)$ solving $v\sinh v+\Lambda\sinh(pv)\sinh((1-p)v)=0$"
             "\n(dotted lines: predicted window edges $p(1-p)=1/|\\Lambda|$)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig("prob7.53_p.png", dpi=150)
plt.show()
print("Plot saved to prob7.53_p.png")

print("\nVerification of |Lambda| > 1/[p(1-p)] as the existence condition:")
for L in Lambdas:
    p1, p2 = p_window(L)
    vs = np.array([bisect(lambda v: f(v, p, L), 1e-4, 30.0) for p in ps], dtype=float)
    idx = np.where(~np.isnan(vs))[0]
    v_half = bisect(lambda v: f(v, 0.5, L), 1e-4, 30.0)
    print(f"Lambda = {L:g}:")
    print(f"  predicted window   p in ({p1:.4f}, {p2:.4f})")
    print(f"  numerical window   p in ({ps[idx[0]]:.4f}, {ps[idx[-1]]:.4f})")
    print(f"  v(p=1/2) = {v_half:.6f}  (should match the symmetric-case root for this Lambda)")
