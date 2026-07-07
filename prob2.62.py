import numpy as np
import matplotlib.pyplot as plt


N = 100
scale = (N + 1) ** 2  # 101**2 = 10201

# Interior x points (excluding boundaries 0 and 1)
x_j = np.linspace(0, 1, N + 2)[1:-1]

# Build tridiagonal matrix with perturbation on diagonal
M = np.zeros((N, N))
np.fill_diagonal(M, 2 + 500 / (N + 1) ** 2 * np.sin(np.pi * x_j))
np.fill_diagonal(M[1:], -1)
np.fill_diagonal(M[:, 1:], -1)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(M)
scaled_eigenvalues = eigenvalues[:3] * scale

print(f"3 lowest eigenvalues (scaled by {scale}):")
for i, ev in enumerate(scaled_eigenvalues):
    print(f"  lambda_{i+1} = {ev:.6f}")

# Plot eigenvectors
x = np.linspace(0, 1, N + 2)

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):
    psi = np.concatenate(([0], eigenvectors[:, i], [0]))
    ax.plot(x, psi, label=f"n={i+1}, λ={scaled_eigenvalues[i]:.4f}")

ax.set_title("3 Lowest Eigenvectors (N=100, with sin perturbation)")
ax.set_xlabel("x")
ax.set_ylabel("ψ")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("prob2.62_eigenvectors.png", dpi=150)
plt.show()
print("\nPlot saved to prob2.62_eigenvectors.png")
