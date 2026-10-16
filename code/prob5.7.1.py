import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 500)

gauss = np.exp(-x**2 / 2)

f1 = (4 / np.pi) * gauss * (0.5 * np.sqrt(np.pi / 8) + 0.25 * x**2 * np.sqrt(np.pi / 2))
f2 = (4 / np.pi) * gauss * np.sqrt(np.pi / 8)
f3 = (2 / np.pi) * gauss * x**2 * np.sqrt(np.pi / 2)

labels = [
    r"$\frac{4}{\pi}e^{-x^2/2}\left(\frac{1}{2}\sqrt{\frac{\pi}{8}}+\frac{1}{4}x^2\sqrt{\frac{\pi}{2}}\right)$",
    r"$\frac{4}{\pi}e^{-x^2/2}\sqrt{\frac{\pi}{8}}$",
    r"$\frac{2}{\pi}e^{-x^2/2}x^2\sqrt{\frac{\pi}{2}}$",
]

fig, ax = plt.subplots(figsize=(8, 5))
for f, label in zip([f1, f2, f3], labels):
    ax.plot(x, f, label=label)

ax.set_xlabel("x")
ax.set_ylabel("Probability density")
ax.set_title("Problem 5.7.1 — Probability Densities")
ax.legend(fontsize=9)
ax.set_xlim(0, 5)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("prob5.7.1_densities.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved prob5.7.1_densities.png")
