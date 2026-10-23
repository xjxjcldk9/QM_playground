"""
Plot the total cross-section B of a hard disk (2D scattering) as a function
of ka, for 0 < ka < 2.

With c_j = -i^j J_j(ka) / H_j^(1)(ka),

    B = (4/k) sum_j |J_j(ka) / H_j^(1)(ka)|^2,

so the dimensionless quantity plotted is

    B/a = (4/(ka)) sum_j |J_j(ka) / H_j^(1)(ka)|^2.

Since J_{-j} = (-1)^j J_j and H_{-j} = (-1)^j H_j, the j and -j terms are
equal. For small ka the j = 0 term dominates and |H_0(ka)| ~ (2/pi)|ln(ka)|,
so B/a ~ pi^2 / (ka ln^2(ka)) diverges (slowly) as ka -> 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1


def B_over_a(x, jmax=30):
    total = np.abs(jv(0, x) / hankel1(0, x)) ** 2
    for j in range(1, jmax + 1):
        total += 2 * np.abs(jv(j, x) / hankel1(j, x)) ** 2
    return 4 / x * total


x = np.linspace(1e-3, 2, 2000)
B = B_over_a(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, B, lw=1.6)
ax.set_xlabel(r"$ka$")
ax.set_ylabel(r"$B/a$")
ax.set_title("Problem 10.22(f): hard-disk total cross-section")
ax.set_xlim(0, 2)
ax.set_ylim(0, 12)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob10.22f.png", dpi=150)
print("saved prob10.22f.png")

for v in [0.01, 0.1, 0.5, 1.0, 2.0]:
    print(f"ka = {v:4.2f}: B/a = {B_over_a(np.array(v)):.4f}")
