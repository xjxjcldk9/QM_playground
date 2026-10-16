"""
Problem 8.30: infinite square well with a sloping floor,

    V(x) = V0 x/a,      0 < x < a

expanded in the first N=10 unperturbed basis functions

    phi_n(x) = sqrt(2/a) sin(n pi x/a)

Matrix elements of H = p^2/2m + V (units hbar = m = a = 1 below):

    Diagonal:      H_nn = n^2 pi^2/2 + V0/2
                          (since <n|x/a|n> = 1/2 for every n)

    Off-diagonal:  H_nk = -8 V0 n k / [pi^2 (n^2-k^2)^2]   if n+k is odd
                          0                                  if n+k is even

(<n|x/a|k> vanishes by symmetry whenever n+k is even; the closed form above
was checked against direct numerical integration of <n|x/a|k>.)

Solve for the minimum eigenvalue/eigenvector of the truncated 10x10 matrix
for V0 = 100 hbar^2/(m a^2), then plot the corresponding wavefunction
psi(x) = sum_n c_n phi_n(x).
"""
import numpy as np
import matplotlib.pyplot as plt

N = 10          # number of basis functions
V0 = 100.0      # in units of hbar^2/(m a^2)
a = 1.0         # well width (hbar = m = a = 1)

n = np.arange(1, N + 1)
H = np.zeros((N, N))

for i, ni in enumerate(n):
    for j, nj in enumerate(n):
        if ni == nj:
            H[i, j] = ni**2 * np.pi**2 / 2 + V0 / 2
        elif (ni + nj) % 2 == 1:
            H[i, j] = -8 * V0 * ni * nj / (np.pi**2 * (ni**2 - nj**2)**2)
        # else stays 0

eigvals, eigvecs = np.linalg.eigh(H)
E_min = eigvals[0]
c = eigvecs[:, 0]

print("Lowest 5 eigenvalues (units of hbar^2/(m a^2)):")
print(eigvals[:5])
print(f"\nMinimum eigenvalue: E0 = {E_min:.6f} hbar^2/(m a^2)")
print("Expansion coefficients of the ground state:")
for ni, ci in zip(n, c):
    print(f"  c_{ni} = {ci: .5f}")

x = np.linspace(0, a, 500)


def phi(ni, x):
    return np.sqrt(2 / a) * np.sin(ni * np.pi * x / a)


psi = sum(ci * phi(ni, x) for ni, ci in zip(n, c))
# fix overall sign so the wavefunction is positive near the well's low side
if psi[len(psi) // 4] < 0:
    psi = -psi

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, psi, color="tab:blue", lw=2, label=r"$\psi_{\min}(x)$")
ax.plot(x, phi(1, x), color="gray", lw=1.5, ls="--", label=r"unperturbed $\phi_1(x)$")
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$\psi(x)$")
ax.set_title(rf"Problem 8.30: ground state, sloping floor ($V_0={V0:.0f}\,\hbar^2/ma^2$), $E_0={E_min:.3f}$")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob8.30.png", dpi=150)
print("\nsaved prob8.30.png")
