import matplotlib.pyplot as plt
import numpy as np
from mpmath import besselk, mpc, re, quad as mpquad

# Order is 4j (purely imaginary, order = 4i)
order = mpc(0, 4)

x_vals = np.linspace(0.01, 10, 2000)
y_vals = np.array([float(re(besselk(order, x))) * np.sqrt(x) for x in x_vals])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, y_vals, lw=1.5, color="steelblue")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("x")
ax.set_ylabel(r"Re$\,K_{4i}(x)\,\sqrt{x}$")
ax.set_title(r"$\sqrt{x}\,K_{4i}(x)$")
ax.set_xlim(x_vals[0], x_vals[-1])

plt.tight_layout()
plt.savefig("prob2.60_K4i.png", dpi=150)

# Integrate Re[K_{4i}(x)]^2 * x from 0 to infinity
integrand = lambda x: re(besselk(order, x))**2 * x
result = mpquad(integrand, [0, float('inf')])
print(f"∫_0^∞ Re[K_{{4i}}(x)]^2 · x dx = {result}")
print(f"1/sqrt(result) = {1/result**0.5}")

# Find zero of Re[K_{4i}(x)] between 1.5 and 2
from mpmath import findroot
zero = findroot(lambda x: re(besselk(order, x)), 1.75)
print(f"Zero of Re[K_{{4i}}(x)] near [1.5, 2]: x = {zero}")
k = float(zero)

# Plot sqrt(x) * Re[K_{4i}(k*x)] from x=0 to 6
x2_vals = np.linspace(0.001, 6, 2000)
y2_vals = np.array([float(re(besselk(order, k * x))) * np.sqrt(x) if x >= 1 else 0.0 for x in x2_vals])

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(x2_vals, y2_vals, lw=1.5, color="steelblue")
ax2.axhline(0, color="gray", lw=0.5, ls="--")
ax2.set_xlabel("x")
ax2.set_ylabel(r"Re$\,K_{4i}(kx)\,\sqrt{x}$")
ax2.set_title(r"$\sqrt{x}\,\mathrm{Re}\,K_{4i}(kx)$, $k=$" + f"{k:.6f}")
ax2.set_xlim(0, 6)

plt.tight_layout()
plt.savefig("prob2.60_K4i_kx.png", dpi=150)
