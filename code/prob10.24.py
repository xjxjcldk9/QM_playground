"""
Plot the 1D phase shift delta of a finite square well (depth V_0, width 2a)
as a function of ka, for 0 < ka < 20, Problem 10.24.

With k = sqrt(2mE)/hbar and k' = sqrt(2m(V_0 + E))/hbar,

    delta = -ka + arccot( (k'/k) cot(k'a) ),

and k'a = sqrt((ka)^2 + z0^2), where z0 = sqrt(2mV_0) a / hbar = 10.

delta is only defined mod pi; we take arccot in (0, pi) at ka -> 0 (so
delta(0) = 0) and then follow it continuously as ka increases.
"""
import numpy as np
import matplotlib.pyplot as plt

z0 = 10.0
ka = np.linspace(1e-4, 20, 20000)
kpa = np.sqrt(ka**2 + z0**2)

# arccot(x) = arctan2(1, x) in (0, pi); written as arctan2(k, k' cot(k'a)) and
# unwrapped with period pi to remove the jumps where cot(k'a) blows up
arccot = np.unwrap(np.arctan2(ka * np.sin(kpa), kpa * np.cos(kpa)) % np.pi, period=np.pi)
delta = -ka + arccot

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ka, delta, lw=1.6)
ax.set_xlabel(r"$ka$")
ax.set_ylabel(r"$\delta$")
ax.set_title(r"Problem 10.24: phase shift of a finite square well, $\sqrt{2mV_0}\,a/\hbar = 10$")
ax.set_xlim(0, 20)
ax.set_yticks(-np.pi / 2 * np.arange(6))
ax.set_yticklabels(["0", r"$-\pi/2$", r"$-\pi$", r"$-3\pi/2$", r"$-2\pi$", r"$-5\pi/2$"])
ax.set_ylim(-5 * np.pi / 2 - 0.2, 0.2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob10.24.png", dpi=150)
print("saved prob10.24.png")
