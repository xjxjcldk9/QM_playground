import numpy as np
import matplotlib.pyplot as plt

# Infinite square well (0 < x < a) with a step perturbation:
#   V(x) = V0  for 0 < x < a/2
#   V(x) = 0   for a/2 < x < a
# with V0 = 4 hbar^2/(m a^2).
#
# Same finite-difference method as prob2.61: discretize psi'' on a grid of
# N interior points with spacing h = a/(N+1), giving the tridiagonal matrix
# M (2 on the diagonal, -1 off-diagonal). The eigenvalues of M*(N+1)^2 are
# the dimensionless energies eps = E / (hbar^2 / 2 m a^2) for the *bare*
# well. Adding a potential just adds it to the diagonal in the same units:
#   H = M*(N+1)^2 + diag(v(x_i)),   v = V / (hbar^2 / 2 m a^2)
# so V0 = 4 hbar^2/(m a^2) becomes v0 = 8 in these units.

v0 = 8.0  # V0 in units of hbar^2/(2 m a^2)
N = 100


def make_matrix(N):
    M = np.zeros((N, N))
    np.fill_diagonal(M, 2)
    np.fill_diagonal(M[1:], -1)
    np.fill_diagonal(M[:, 1:], -1)
    return M


x = np.arange(1, N + 1) / (N + 1)  # interior grid points, a = 1
v = np.where(x < 0.5, v0, np.where(x == 0.5, v0 / 2, 0.0))
H = make_matrix(N) * (N + 1) ** 2 + np.diag(v)
eigenvalues, eigenvectors = np.linalg.eigh(H)
eps1 = eigenvalues[0]

# Unperturbed ground state and first-order perturbation theory prediction
eps0 = np.pi**2
dE1 = v0 / 2  # <V> = V0 * integral_0^{a/2} sin^2(pi x/a) (2/a) dx = V0/2
eps_PT = eps0 + dE1

print(f"N = {N}x{N} matrix")
print(f"Unperturbed ground state:   eps_1^(0) = pi^2      = {eps0:.4f}")
print(f"1st-order perturbation:     <V> = V0/2            = {dE1:.4f}")
print(f"1st-order PT estimate:      eps_1 = pi^2 + V0/2    = {eps_PT:.4f}")
print(f"Numerical (matrix) result:  eps_1                 = {eps1:.4f}")

# --- Plot ground-state wavefunction ---
psi = eigenvectors[:, 0]
psi = psi / np.sign(psi[N // 2])  # fix overall sign
psi_unperturbed = np.sqrt(2) * np.sin(np.pi * x)
norm = np.sqrt(np.trapezoid(psi**2, x))


def psi_n(n, x):
    return np.sqrt(2) * np.sin(n * np.pi * x)


# First-order correction to the ground-state wavefunction:
#   psi_1^(1) = (4 V0 m a^2 / pi^3 hbar^2) * (-4/9 psi_2 + 8/225 psi_4 - 12/1225 psi_6)
# With V0 = 4 hbar^2/(m a^2), the prefactor 4 V0 m a^2/(pi^3 hbar^2) = 16/pi^3.
prefactor = 16 / np.pi**3
psi1_correction = prefactor * (
    -4 / 9 * psi_n(2, x) + 8 / 225 * psi_n(4, x) - 12 / 1225 * psi_n(6, x)
)
psi_first_order = psi_unperturbed + psi1_correction

plt.figure(figsize=(7, 5))
plt.plot(x, psi / norm, label="perturbed ground state (numerical)")
plt.plot(x, psi_unperturbed, "--", label="unperturbed ground state")
plt.plot(x, psi_first_order, "-.", label="1st-order approximation")
plt.axvline(0.5, color="gray", ls=":", label="step at x=a/2")
plt.xlabel("x/a")
plt.ylabel(r"$\psi_1$")
plt.title(f"Ground-state wavefunction (N={N})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prob7.7_ground_state.png", dpi=150)
plt.show()
print("\nPlot saved to prob7.7_ground_state.png")
