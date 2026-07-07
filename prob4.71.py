import numpy as np
import matplotlib.pyplot as plt

n = 80
coords = np.linspace(-5, 5, n)

# 2D grids for each coordinate plane (the third coordinate = 0)
X_xy, Y_xy = np.meshgrid(coords, coords)
R_xy = np.sqrt(X_xy**2 + Y_xy**2)

X_xz, Z_xz = np.meshgrid(coords, coords)
R_xz = np.sqrt(X_xz**2 + Z_xz**2)

Y_yz, Z_yz = np.meshgrid(coords, coords)
R_yz = np.sqrt(Y_yz**2 + Z_yz**2)

orbitals = [
    (r"$\psi_x = x\,e^{-r}$",
     X_xy * np.exp(-R_xy),   # z=0 plane
     X_xz * np.exp(-R_xz),   # y=0 plane
     np.zeros_like(Y_yz)),    # x=0 plane: x=0 so ψ=0
    (r"$\psi_y = y\,e^{-r}$",
     Y_xy * np.exp(-R_xy),   # z=0 plane
     np.zeros_like(X_xz),    # y=0 plane: y=0 so ψ=0
     Y_yz * np.exp(-R_yz)),   # x=0 plane
    (r"$\psi_z = z\,e^{-r}$",
     np.zeros_like(X_xy),    # z=0 plane: z=0 so ψ=0
     Z_xz * np.exp(-R_xz),   # y=0 plane
     Z_yz * np.exp(-R_yz)),   # x=0 plane
]

lim = 5
levels = np.linspace(-0.8, 0.8, 30)
cmap = "RdBu_r"

fig = plt.figure(figsize=(18, 6))

for i, (title, psi_xy, psi_xz, psi_yz) in enumerate(orbitals):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")

    # z=0 plane  (xy cross-section)
    ax.contourf(X_xy, Y_xy, psi_xy,
                zdir="z", offset=0, levels=levels, cmap=cmap, alpha=0.75)

    # y=0 plane  (xz cross-section)
    ax.contourf(X_xz, psi_xz, Z_xz,
                zdir="y", offset=0, levels=levels, cmap=cmap, alpha=0.75)

    # x=0 plane  (yz cross-section)
    ax.contourf(psi_yz, Y_yz, Z_yz,
                zdir="x", offset=0, levels=levels, cmap=cmap, alpha=0.75)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect([1, 1, 1])

plt.suptitle(r"2p hydrogen orbitals: $\psi = \{x,\,y,\,z\}\,e^{-r}$  "
             r"(cross-sections at coordinate planes)", fontsize=13)
plt.tight_layout()
plt.savefig("prob4.71_orbitals.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to prob4.71_orbitals.png")
