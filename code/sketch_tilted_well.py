import numpy as np
import matplotlib.pyplot as plt

# Sketch: a wide, deep finite square well,
#   V0(x) = -V0   for |x| < a   (deep, flat floor)
#   V0(x) = 0     for |x| > a   (finite walls)
# perturbed by a linear term H' = -alpha*x, so that
#   V(x) = V0(x) - alpha*x
# tilts the floor of the well and raises one wall / lowers the other.

a = 5.0       # half-width of the well (wide: many bound states)
V0 = 10.0     # well depth (deep: floor well below the wall energy)
alpha = 0.6   # perturbation strength

x = np.linspace(-a - 3, a + 3, 2000)

# unperturbed finite square well
V_well = np.where(np.abs(x) < a, -V0, 0.0)

# perturbed potential
V_tilted = V_well - alpha * x

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, V_well, color="0.5", lw=2, ls="--", label=r"unperturbed well $V_0(x)$")
ax.plot(x, V_tilted, color="C0", lw=2.5, label=r"perturbed $V(x) = V_0(x) - \alpha x$")

ax.axvline(-a, color="0.8", lw=1, ls=":")
ax.axvline(a, color="0.8", lw=1, ls=":")
ax.axhline(0, color="0.8", lw=1)

ax.set_xlabel("x")
ax.set_ylabel("V(x)")
ax.set_title("Finite square well (wide & deep) tilted by a linear perturbation")
ax.legend(loc="upper center")
ax.set_ylim(-V0 - 6, 6)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("sketch_tilted_well.png", dpi=150)
plt.show()
print("Saved sketch_tilted_well.png")
