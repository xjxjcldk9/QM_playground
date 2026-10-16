"""
V(x) = V_0 * sech^2(x/a)

For E > V_0 the particle is classically allowed everywhere (no real turning
point), so the WKB reflection coefficient comes instead from the nearest
*complex* turning point. Writing x = i*a*y (y real) and using
sech(iy) = sec(y):

    V(i*a*y) = V_0 * sec^2(y/a)

so along this contour

    p(y) = sqrt(2m(E - V_0*sec^2(y/a)))
         = sqrt(2mE - 4*V_0/(2*cos(2y/a) + 2))        [since sec^2(t) = 2/(cos(2t)+1)]

p(y) is real and decreases from p(0) = sqrt(2m(E-V_0)) down to 0 at
y = y_0 = a*arccos(sqrt(V_0/E)), the (imaginary-axis) turning point.

The WKB barrier-penetration integral along this path evaluates in closed
form to

    lambda = (pi*a/hbar) * (sqrt(2mE) - sqrt(2mV_0))      [== 2 * int_0^{y0} p(y) dy / hbar,
                                                             checked numerically]

and the above-barrier reflection coefficient is

    R = e^{-2*lambda}

Plots below: V(x); p(y) for 0 <= y <= y_0 at one fixed E > V_0; and R(E)
for fixed V_0, a.
"""
import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
m = 1.0
a = 1.0
V0 = 1.0

# --- panel 1: V(x) = V0 sech^2(x/a) ---------------------------------------
x = np.linspace(-4 * a, 4 * a, 400)
V = V0 / np.cosh(x / a) ** 2

# --- panel 2: p(y) for 0 <= y <= y0, at one fixed E > V0 ------------------
E_demo = 2.0 * V0
y0 = a * np.arccos(np.sqrt(V0 / E_demo))
y = np.linspace(0, y0, 400)
p = np.sqrt(2 * m * (E_demo - V0 / np.cos(y / a) ** 2))

# --- panel 3: R(E) = exp(-2*lambda(E)) for fixed V0, a --------------------
E = np.linspace(1.001 * V0, 10 * V0, 400)
lam = (np.pi * a / hbar) * (np.sqrt(2 * m * E) - np.sqrt(2 * m * V0))
R = np.exp(-2 * lam)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(x, V, color="tab:blue")
ax.set_xlabel("x/a")
ax.set_ylabel(r"$V(x)/V_0$")
ax.set_title(r"$V(x) = V_0\,\mathrm{sech}^2(x/a)$")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(y, p, color="tab:orange")
ax.axvline(y0, color="gray", ls=":", lw=1)
ax.text(y0, 0.05, r"$y_0$", color="gray", ha="right", va="bottom")
ax.set_xlabel("y/a")
ax.set_ylabel(r"$p(y)/\sqrt{2mV_0}$")
ax.set_title(rf"$p(y)$,  $E = {E_demo/V0:g}\,V_0$")
ax.set_xlim(0, y0 * 1.05)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(E / V0, R, color="tab:green")
ax.set_xlabel(r"$E/V_0$")
ax.set_ylabel(r"$R = e^{-2\lambda}$")
ax.set_title("WKB above-barrier reflection coefficient")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("prob9.20.png", dpi=150)
print("saved prob9.20.png")
print(f"y0 (at E = {E_demo/V0:g} V0) = {y0:.6f} * a")
