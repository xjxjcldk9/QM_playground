"""
lambda = (2/hbar) * int_0^{y0} sqrt(2m(E - V_0/(1-(y/a)^2))) dy,

where y0 = a*sqrt(1 - V_0/E) is where the integrand vanishes (E > V_0).

Express lambda as an elliptic integral:

Substitute y = a*k0*sin(phi), with k0 = sqrt(1 - V_0/E) (so y: 0->y0 maps to
phi: 0->pi/2). Using k0^2 = 1 - V_0/E and 1 - (y/a)^2 = 1 - k0^2*sin^2(phi):

    E - V_0/(1-(y/a)^2) = E * (k0^2*cos^2(phi)) / (1 - k0^2*sin^2(phi))

so

    lambda = (2a*sqrt(2mE)/hbar) * k0^2 * int_0^{pi/2} cos^2(phi)/sqrt(1-k0^2 sin^2 phi) dphi

The standard identity

    int_0^{pi/2} cos^2(phi)/sqrt(1-k^2 sin^2 phi) dphi = [E(k) - (1-k^2)K(k)]/k^2

(K, E = complete elliptic integrals of the 1st/2nd kind, modulus k) then gives
the closed form

    lambda(E) = (2a*sqrt(2mE)/hbar) * [ E(k0) - (V_0/E)*K(k0) ],   k0 = sqrt(1-V_0/E)

R = e^{-2*lambda}

Checked numerically below against brute-force quadrature of the original
integral before plotting R(E).
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import ellipe, ellipk
import matplotlib.pyplot as plt

hbar = 1.0
m = 1.0
a = 1.0
V0 = 1.0


def lam_direct(E):
    y0 = a * np.sqrt(1 - V0 / E)

    def integrand(y):
        return np.sqrt(2 * m * (E - V0 / (1 - (y / a) ** 2)))

    I, _ = quad(integrand, 0, y0, limit=200, points=[y0])
    return (2 / hbar) * I


def lam_elliptic(E):
    k0_sq = 1 - V0 / E  # scipy's ellipk/ellipe take the parameter m = k^2
    K = ellipk(k0_sq)
    Ei = ellipe(k0_sq)
    return (2 * a * np.sqrt(2 * m * E) / hbar) * (Ei - (V0 / E) * K)


# --- sanity check: closed form vs. direct quadrature ----------------------
for E_test in [1.2, 2.0, 5.0, 10.0]:
    d, el = lam_direct(E_test), lam_elliptic(E_test)
    print(f"E/V0={E_test:5.2f}:  lambda_direct={d:.6f}  lambda_elliptic={el:.6f}  "
          f"(diff={abs(d-el):.2e})")

# --- R(E) ------------------------------------------------------------------
E = np.linspace(1.001 * V0, 10 * V0, 400)
lam = lam_elliptic(E)
R = np.exp(-2 * lam)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(E / V0, R, color="tab:green")
ax.set_xlabel(r"$E/V_0$")
ax.set_ylabel(r"$R = e^{-2\lambda}$")
ax.set_title(r"$R(E)$ via $\lambda$ expressed as an elliptic integral")
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("prob9.20d.png", dpi=150)
print("saved prob9.20d.png")
