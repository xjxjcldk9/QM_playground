import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1, x2)

gauss1 = np.exp(-X1**2)
gauss2 = np.exp(-X2**2)

Z1 = X2 * gauss1 * gauss2
Z2 = X2 * gauss1 * gauss2 + X1 * gauss1 * gauss2
Z3 = X2 * gauss1 * gauss2 - X1 * gauss1 * gauss2

titles = [
    r"$x_2 e^{-x_1^2} e^{-x_2^2}$",
    r"$x_2 e^{-x_1^2} e^{-x_2^2} + x_1 e^{-x_1^2} e^{-x_2^2}$",
    r"$x_2 e^{-x_1^2} e^{-x_2^2} - x_1 e^{-x_1^2} e^{-x_2^2}$",
]
Zs = [Z1, Z2, Z3]

fig = plt.figure(figsize=(18, 5))
for i, (Z, title) in enumerate(zip(Zs, titles)):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, cmap="RdBu_r", alpha=0.9)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("f")
    ax.set_title(title, fontsize=10)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig("prob5.7_surfaces.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved prob5.7_surfaces.png")
