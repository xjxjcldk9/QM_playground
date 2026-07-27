"""
Problem 6.6: Band structure of a periodic square well via Bloch's theorem.

Matrix equation (from the finite-difference reduction of the Schrodinger
equation with psi_j = e^{i q x_j} u_j, u periodic over one cell):

    (-lam - i*q*lam*dx) u_{j+1}
  + (2*lam + V_j + q^2*lam*dx^2) u_j
  + (-lam + i*q*lam*dx) u_{j-1}  =  E u_j

with lam = hbar^2 / (2 m dx^2).

Potential: V(x) = -20 hbar^2/(2 m a^2) for a/4 < x < 3a/4, else 0,
periodic with period a.

Units used here: hbar^2/(2m) = 1, a = 1, so energies come out in units
of hbar^2/(2 m a^2).
"""
import numpy as np
import matplotlib.pyplot as plt

def build_hamiltonian(N, qa, V0=-20.0, a=1.0):
    """Build the N x N Hermitian, periodic (cyclic) tridiagonal Bloch
    Hamiltonian for the reduced/periodic part u_j at Bloch phase qa."""
    dx = a / N
    lam = 1.0 / dx**2          # hbar^2/(2m) = 1
    q = qa / a

    x = (np.arange(N) + 0.5) * dx      # cell-centered grid points
    V = np.where((x > a/4) & (x < 3*a/4), V0, 0.0)

    off  = -lam - 1j*q*lam*dx   # coefficient multiplying u_{j+1}
    offc = np.conj(off)         # coefficient multiplying u_{j-1} (= -lam + i q lam dx)
    diag = 2*lam + q**2*lam*dx**2 + V

    H = np.zeros((N, N), dtype=complex)
    for j in range(N):
        H[j, j] = diag[j]
        jp = (j + 1) % N        # periodic wrap: u_N -> u_0
        jm = (j - 1) % N        # periodic wrap: u_{-1} -> u_{N-1}
        H[j, jp] += off
        H[j, jm] += offc

    return H


def lowest_energies(N, qa_list, V0=-20.0, a=1.0, num=3):
    results = {}
    for qa in qa_list:
        H = build_hamiltonian(N, qa, V0, a)
        assert np.allclose(H, H.conj().T, atol=1e-8), "H not Hermitian!"
        evals = np.linalg.eigvalsh(H)
        results[qa] = evals[:num]
    return results


if __name__ == "__main__":
    qa_list = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    labels = ["-pi", "-pi/2", "0", "pi/2", "pi"]

    # convergence check
    for N in [100, 200, 400]:
        print(f"N={N}")
        res = lowest_energies(N, qa_list)
        for lbl, qa in zip(labels, qa_list):
            print(f"  qa={lbl:6s}: lowest 3 E = {res[qa]}")
        print()

    # Richardson extrapolation (finite-difference error ~ 1/N^2)
    print("Richardson extrapolation of the lowest band energy:")
    res800 = lowest_energies(800, qa_list, num=1)
    res1600 = lowest_energies(1600, qa_list, num=1)
    for lbl, qa in zip(labels, qa_list):
        e1 = res800[qa][0]
        e2 = res1600[qa][0]
        extrap = e2 + (e2 - e1) / 3
        print(f"  qa={lbl:6s}: N=800 {e1:.6f}, N=1600 {e2:.6f}, extrapolated {extrap:.6f}")

    # --- Band plot: first two energies vs q over the full Brillouin zone ---
    N_plot = 400
    qa_scan = np.linspace(-np.pi, np.pi, 121)
    bands = np.array([lowest_energies(N_plot, [qa], num=2)[qa] for qa in qa_scan])

    plt.figure(figsize=(6, 4.5))
    plt.plot(qa_scan, bands[:, 0], label="1st band")
    plt.plot(qa_scan, bands[:, 1], label="2nd band")
    plt.xlabel(r"$qa$")
    plt.ylabel(r"$E$  (units of $\hbar^2/2ma^2$)")
    plt.title("Band structure of periodic square well")
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("prob6.6_bands.png", dpi=150)
    plt.show()
