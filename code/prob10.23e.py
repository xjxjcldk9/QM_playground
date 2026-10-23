"""
Plot the logarithm of the Mott cross-sections for two identical charged
particles (Rutherford scattering), bosons vs. fermions, Problem 10.23(e).

With f(theta) = -q^2 / (16 pi eps0 E sin^2(theta/2)) and
sin^2((pi - theta)/2) = cos^2(theta/2),

    |f_B|^2 = C^2 (1/sin^2(theta/2) + 1/cos^2(theta/2))^2,
    |f_F|^2 = C^2 (1/sin^2(theta/2) - 1/cos^2(theta/2))^2,

where C = q^2 / (16 pi eps0 E). We plot ln(|f|^2 / C^2). Both curves are
symmetric about theta = pi/2; |f_F|^2 vanishes there, so its log -> -inf.
"""
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0.05, np.pi - 0.05, 4000)
s2 = np.sin(theta / 2) ** 2
c2 = np.cos(theta / 2) ** 2

fB = (1 / s2 + 1 / c2) ** 2
fF = (1 / s2 - 1 / c2) ** 2

with np.errstate(divide="ignore"):
    lnB = np.log(fB)
    lnF = np.log(fF)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, lnB, lw=1.6, label=r"bosons: $\ln(|f_B|^2/C^2)$")
ax.plot(theta, lnF, lw=1.6, label=r"fermions: $\ln(|f_F|^2/C^2)$")
ax.axvline(np.pi / 2, color="gray", ls="--", lw=0.8)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\ln(|f|^2 / C^2)$,  $C = q^2/(16\pi\epsilon_0 E)$")
ax.set_title("Problem 10.23(e): identical-particle Coulomb scattering")
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.set_xlim(0, np.pi)
ax.set_ylim(-5, 18)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("prob10.23e.png", dpi=150)
print("saved prob10.23e.png")
