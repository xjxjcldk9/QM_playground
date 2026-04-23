import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import ai_zeros, airy

# First and tenth zeros of Ai
zeros, _, _, _ = ai_zeros(10)
z1, z10 = -zeros[0], -zeros[9]
print(f"1st zero of Ai:  a1  = {z1:.10f}")
print(f"10th zero of Ai: a10 = {z10:.10f}")

int1, err1 = quad(lambda t: airy(t-z1)[0]**2, 0, np.inf)
int10, err10 = quad(lambda t: airy(t-z10)[0]**2, 0, np.inf)
print(f"\n∫_{{a1}}^∞  Ai(z)^2 dz = {int1:.10f}  (error: {err1:.2e})")
print(f"∫_{{a10}}^∞ Ai(z)^2 dz = {int10:.10f}  (error: {err10:.2e})")
print(f"\n1/sqrt(int1)  = {1/np.sqrt(int1):.10f}")
print(f"1/sqrt(int10) = {1/np.sqrt(int10):.10f}")

int_cross, err_cross = quad(lambda t: airy(t - z1)[0] * airy(t - z10)[0], 0, np.inf)
print(f"\n∫_0^∞ Ai(t-a1)·Ai(t-a10) dt = {int_cross:.10f}  (error: {err_cross:.2e})")

# Plot Ai(x - z1) and Ai(x - z10) on two axes
x = np.linspace(0, 16, 1000)
configs = [
    (z1,  r"$\psi_1(x)$"),
    (z10, r"$\psi_{10}(x)$"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (z_n, title) in zip(axes, configs):
    Ai_shifted, _, _, _ = airy(x - z_n)
    ax.plot(x, Ai_shifted, lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, 16)
    ax.set_xlabel("x")
    ax.set_title(title)


plt.tight_layout()
plt.show()
