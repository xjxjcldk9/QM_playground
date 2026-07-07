import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)

f = (2 / np.sqrt(np.pi)) * np.exp(-x**2) * (x**2 + 1/2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, f, color="steelblue")
ax.set_xlabel("x")
ax.set_ylabel("Probability density")
ax.set_title(r"Problem 5.7.4 — $\frac{2}{\sqrt{\pi}}e^{-x^2}(x^2+1/2)$")
ax.set_xlim(-5, 5)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("prob5.7.4_density.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved prob5.7.4_density.png")
