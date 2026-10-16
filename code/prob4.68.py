import numpy as np
import matplotlib.pyplot as plt

N = 1000
beta = 1 / 50
j = np.arange(1, N + 1, dtype=float)
scale = 1 / (2 * (beta) ** 2)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, l in enumerate([0, 1, 2]):
    vj = -2 * beta / j + l * (l + 1) / j ** 2
    H = np.zeros((N, N))
    np.fill_diagonal(H, 2 + vj)
    np.fill_diagonal(H[1:], -1)
    np.fill_diagonal(H[:, 1:], -1)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energies = eigenvalues[:3] * scale

    print(f"\nl = {l}")
    for i, E in enumerate(energies):
        print(f"  E_{i+1} = {E:.6f}")

    r = np.concatenate(([0], j / (N + 1), [1]))

    for col in range(3):
        ax = axes[row, col]
        psi = np.concatenate(([0], eigenvectors[:, col], [0]))
        ax.plot(r, psi)
        ax.set_title(f"l={l}, n={col+1},  E={energies[col]:.5f}")
        ax.set_xlabel("r")
        ax.set_ylabel("ψ")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "b"])
        ax.grid(True, alpha=0.3)

plt.suptitle("3 Lowest Radial Eigenfunctions for l = 0, 1, 2  (energies in ℏ²/ma²)", fontsize=14)
plt.tight_layout()
plt.savefig("prob4.68_eigenfunctions.png", dpi=150)
plt.show()
print("\nPlot saved to prob4.68_eigenfunctions.png")
