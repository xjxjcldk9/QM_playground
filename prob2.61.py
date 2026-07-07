import numpy as np
import matplotlib.pyplot as plt


def make_matrix(N):
    M = np.zeros((N, N))
    np.fill_diagonal(M, 2)
    np.fill_diagonal(M[1:], -1)
    np.fill_diagonal(M[:, 1:], -1)
    return M


scale = {10: 121, 100: 10201}

for N in [10, 100]:
    M = make_matrix(N)
    eigenvalues = np.linalg.eigvalsh(M) * scale[N]

    print(f"\n{'='*40}")
    print(f"N = {N}x{N} matrix (scaled by {scale[N]})")
    print(f"{'='*40}")
    print("Five lowest eigenvalues:")
    for i, ev in enumerate(eigenvalues[:5]):
        print(f"  lambda_{i+1} = {ev:.4f}")

# --- Plot eigenvectors for 3 lowest eigenvalues ---
plot_sizes = [1, 2, 3, 10, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, N in zip(axes, plot_sizes):
    M = make_matrix(N)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    x = np.linspace(0, 1, N + 2)
    n_plot = min(3, N)

    for i in range(n_plot):
        psi = np.concatenate(([0], eigenvectors[:, i], [0]))
        ax.plot(x, psi, marker='o' if N <= 10 else None,
                markersize=4, label=f"n={i+1}")

    ax.set_title(f"N = {N}")
    ax.set_xlabel("x")
    ax.set_ylabel("ψ")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide the unused 6th subplot
axes[-1].set_visible(False)

plt.suptitle("3 Lowest Eigenvectors for Various N", fontsize=14)
plt.tight_layout()
plt.savefig("prob2.61_eigenvectors.png", dpi=150)
plt.show()
print("\nPlot saved to prob2.61_eigenvectors.png")
