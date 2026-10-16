"""
Problem 7.53 (finale): infinite square well on [0, 1] with a delta-function
perturbation Lambda*delta(x - 1/2) at the center. Odd-parity states are
unaffected (psi(1/2) = 0 kills the jump condition), so only the even-parity
ground state moves. This script solves the exact matching equations for the
ground state and plots the resulting wave function for several Lambda.

Ansatz:
  E > 0 (u real):   psi = A sin(u x)                  on [0, 1/2]
                     psi = C sin(u x) + D cos(u x)     on [1/2, 1]
  E < 0 (v real):   psi = A sinh(v x)                 on [0, 1/2]
                     psi = C sinh(v x) + D cosh(v x)   on [1/2, 1]

Matching conditions (continuity at x=1/2, psi(1)=0, and the derivative jump
psi'(1/2+) - psi'(1/2-) = Lambda*psi(1/2) from integrating the delta term)
give a 3x3 homogeneous system for (A, C, D); its nontrivial solutions pin
down u (or v) via solve_Eg0/solve_El0, matching the transcendental equations
analyzed in prob7.53.py: a nontrivial E < 0 root only exists for Lambda < -4,
so abs(Lambda) > 4 selects the hyperbolic (v) branch below.
"""
import numpy as np
import matplotlib.pyplot as plt


def bisect(func, lo, hi, tol=1e-12):
    flo, fhi = func(lo), func(hi)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def solve_Eg0(Lambda):
    """Ground-state root of u*sin(u) + Lambda*sin^2(u/2) = 0 (valid for Lambda > -4)."""
    if Lambda == 0:
        return np.pi

    def eq(u):
        return u * np.sin(u) + Lambda * np.sin(0.5 * u) ** 2
    return bisect(eq, 1e-8, np.pi - 1e-8)


def solve_El0(Lambda):
    """Ground-state root of v*sinh(v) + Lambda*sinh^2(v/2) = 0 (valid for Lambda < -4)."""
    def eq(v):
        return v * np.sinh(v) + Lambda * np.sinh(0.5 * v) ** 2
    return bisect(eq, 1e-8, 50.0)


def create_u_matrix(u, Lambda):
    eq1 = np.array([0, np.sin(u), np.cos(u)])
    eq2 = np.array([np.sin(0.5 * u), -np.sin(0.5 * u), -np.cos(0.5 * u)])
    eq3 = np.array([-Lambda * np.sin(0.5 * u) - u * np.cos(0.5 * u),
                     u * np.cos(0.5 * u), -u * np.sin(0.5 * u)])
    return np.array([eq1, eq2, eq3])


def create_v_matrix(v, Lambda):
    eq1 = np.array([0, np.sinh(v), np.cosh(v)])
    eq2 = np.array([np.sinh(0.5 * v), -np.sinh(0.5 * v), -np.cosh(0.5 * v)])
    eq3 = np.array([-Lambda * np.sinh(0.5 * v) - v * np.cosh(0.5 * v),
                     v * np.cosh(0.5 * v), v * np.sinh(0.5 * v)])
    return np.array([eq1, eq2, eq3])


def nullspace_vector(M):
    # M is (near-)singular by construction (u or v was chosen as a root);
    # its null vector is the right-singular vector of the smallest singular value.
    _, _, Vt = np.linalg.svd(M)
    return Vt[-1]


def plot_wave_function_u(ax, coeff, u, Lambda):
    A, C, D = coeff
    x1 = np.linspace(0, 0.5, 200)
    x2 = np.linspace(0.5, 1, 200)
    psi1 = A * np.sin(u * x1)
    psi2 = C * np.sin(u * x2) + D * np.cos(u * x2)

    x = np.concatenate([x1, x2])
    psi = np.concatenate([psi1, psi2])
    psi /= np.sqrt(np.trapezoid(psi ** 2, x))

    ax.plot(x, psi, color="tab:blue", lw=2)
    ax.axvline(0.5, color="gray", lw=0.6, ls=":")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"$\\Lambda={Lambda:g}$, $E>0$ ($u={u:.3f}$)", fontsize=9)


def plot_wave_function_v(ax, coeff, v, Lambda):
    A, C, D = coeff
    x1 = np.linspace(0, 0.5, 200)
    x2 = np.linspace(0.5, 1, 200)
    psi1 = A * np.sinh(v * x1)
    psi2 = C * np.sinh(v * x2) + D * np.cosh(v * x2)

    x = np.concatenate([x1, x2])
    psi = np.concatenate([psi1, psi2])
    psi /= np.sqrt(np.trapezoid(psi ** 2, x))

    ax.plot(x, psi, color="tab:red", lw=2)
    ax.axvline(0.5, color="gray", lw=0.6, ls=":")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"$\\Lambda={Lambda:g}$, $E<0$ ($v={v:.3f}$)", fontsize=9)


Lambdas = [0, -2, -3, -3.5, -4.5, -5, -10]

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
axes = axes.ravel()

for ax, Lambda in zip(axes, Lambdas):
    if abs(Lambda) > 4:
        v = solve_El0(Lambda)
        sys = create_v_matrix(v, Lambda)
        coeff = nullspace_vector(sys)
        plot_wave_function_v(ax, coeff, v, Lambda)
    else:
        u = solve_Eg0(Lambda)
        sys = create_u_matrix(u, Lambda)
        coeff = nullspace_vector(sys)
        plot_wave_function_u(ax, coeff, u, Lambda)

axes[-1].axis("off")
fig.suptitle(r"Problem 7.53: ground state of infinite well + $\Lambda\delta(x-1/2)$")
plt.tight_layout()
plt.savefig("prob7.53f_wavefunctions.png", dpi=150)
plt.show()
print("Plot saved to prob7.53f_wavefunctions.png")
