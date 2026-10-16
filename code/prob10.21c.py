"""

Plot 1/N * (sin(Nqa/2))^2 / (sin(qa/2))^2
as a function of qa for N = 1, 5, 10

Writing x = qa, the function

    F_N(x) = (1/N) sin^2(N x/2) / sin^2(x/2)

is periodic in x with period 2 pi. At x = 2 pi n the ratio is 0/0; its limit
is N^2, so F_N(2 pi n) = N. The peaks therefore get taller (height N) and
narrower (first zeros at x = 2 pi n +- 2 pi/N) as N grows, while the area
under one period stays fixed:

    int_{-pi}^{pi} F_N(x) dx = 2 pi     (for every N)

So as N -> infinity, F_N(x) -> 2 pi sum_n delta(x - 2 pi n): scattering is
concentrated at q = 2 pi n / a (the Bragg condition).
"""
import numpy as np
import matplotlib.pyplot as plt


def F(x, N):
    s = np.sin(x / 2)
    num = np.sin(N * x / 2) ** 2
    small = np.abs(s) < 1e-12
    # at x = 2 pi n use the limit value N
    return np.where(small, N, num / np.where(small, 1.0, s**2) / N)


x = np.linspace(-3 * np.pi, 3 * np.pi, 6001)
Ns = [1, 5, 10]

fig, ax = plt.subplots(figsize=(8, 5))
for N in Ns:
    ax.plot(x / np.pi, F(x, N), lw=1.6, label=f"N = {N}")
ax.set_xlabel(r"$qa/\pi$")
ax.set_ylabel(r"$\frac{1}{N}\,\sin^2(Nqa/2)\,/\,\sin^2(qa/2)$")
ax.set_title("Problem 10.21(c)")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 10.5)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob10.21c.png", dpi=150)
print("saved prob10.21c.png")

# check: area over one period should be 2 pi for every N
xp = np.linspace(-np.pi, np.pi, 200001)
for N in Ns:
    print(f"N = {N:2d}: peak = {F(np.array(0.0), N):.3f}, "
          f"int over one period / (2 pi) = {np.trapezoid(F(xp, N), xp) / (2 * np.pi):.6f}")
