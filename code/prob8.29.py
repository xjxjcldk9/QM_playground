"""
Problem 8.29: plot

    f(beta) = beta^2 (1 - 2 beta) / (2 beta + 3),        0 < beta < 1

and locate its extremum.

df/dbeta = -2 beta (4 beta^2 + 8 beta - 3) / (2 beta + 3)^2

so the only interior critical point in (0,1) is

    beta* = (sqrt(7) - 2)/2 ~ 0.3229,   f(beta*) ~ 0.0101   (a local MAXIMUM)

f falls monotonically afterward, so the minimum of f on (0,1) is approached at
the right endpoint, beta -> 1, where

    f(1) = 1^2 (1-2) / (2+3) = -1/5 = -0.2

That boundary value is annotated below as the minimum.
"""
import numpy as np
import matplotlib.pyplot as plt

beta = np.linspace(1e-4, 1, 2000)
f = beta**2 * (1 - 2 * beta) / (2 * beta + 3)

beta_max = (np.sqrt(7) - 2) / 2
f_max = beta_max**2 * (1 - 2 * beta_max) / (2 * beta_max + 3)

beta_min, f_min = 1.0, -0.2

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(beta, f, color="tab:blue", lw=2)
ax.axhline(0, color="gray", lw=0.8)

ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$f(\beta) = \dfrac{\beta^2(1-2\beta)}{2\beta+3}$")
ax.set_title(r"Problem 8.29: $f(\beta)$ on $0<\beta<1$")
ax.set_xlim(0, 1)
ax.set_ylim(-0.22, 0.14)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob8.29.png", dpi=150)
print("saved prob8.29.png")
print(f"local max at beta = {beta_max:.6f}, f = {f_max:.6f}")
print(f"minimum approached at beta -> 1, f(1) = {f_min}")
