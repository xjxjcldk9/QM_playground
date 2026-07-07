import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import j0, j1, kv

V0 = 10.0
a0 = 1.0

E_min_margin = 1e-6


def compute_j(V, a, n_points=5000):
    """k J1(ka)/J0(ka),  k=sqrt(V+E),  valid for E in (-V, ∞)."""
    E = np.linspace(-V + E_min_margin, 3 * V, n_points)
    k = np.sqrt(V + E)
    denom = j0(k * a)
    f = k * j1(k * a) / denom
    f[np.abs(denom) < 0.05] = np.nan
    return E, f


def compute_k(V, a, n_points=5000):
    """κ K1(κa)/K0(κa),  κ=sqrt(-E),  valid for E in (-V, 0)."""
    E = np.linspace(-V + E_min_margin, -E_min_margin, n_points)
    kap = np.sqrt(-E)
    denom = kv(0, kap * a)
    f = kap * kv(1, kap * a) / denom
    f[np.abs(denom) < 1e-10] = np.nan
    return E, f


fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.25)

Ej, fj = compute_j(V0, a0)
Ek, fk = compute_k(V0, a0)

(line_j,) = ax.plot(Ej, fj, lw=1.5, label=r"$k\,J_1(ka)/J_0(ka)$")
(line_k,) = ax.plot(Ek, fk, lw=1.5, label=r"$\kappa\,K_1(\kappa a)/K_0(\kappa a)$")

ax.axhline(0, color="k", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls=":")
ax.set_xlabel("$E$")
ax.set_title("Prob 4.51  —  intersections give bound-state energies")
ax.set_xlim(-V0, 0)
ax.set_ylim(-20, 20)
ax.legend()
ax.grid(True, alpha=0.3)

ax_V = plt.axes([0.15, 0.12, 0.70, 0.03])
ax_a = plt.axes([0.15, 0.06, 0.70, 0.03])

slider_V = Slider(ax_V, "$V$", 0.5, 50.0, valinit=V0, valstep=0.5)
slider_a = Slider(ax_a, "$a$", 0.1, 5.0, valinit=a0, valstep=0.1)


def update(_):
    V = slider_V.val
    a = slider_a.val
    Ej, fj = compute_j(V, a)
    Ek, fk = compute_k(V, a)
    line_j.set_xdata(Ej)
    line_j.set_ydata(fj)
    line_k.set_xdata(Ek)
    line_k.set_ydata(fk)
    ax.set_xlim(-V, 0)
    fig.canvas.draw_idle()


slider_V.on_changed(update)
slider_a.on_changed(update)

plt.savefig("prob4.51.png", dpi=150, bbox_inches="tight")
plt.show()
