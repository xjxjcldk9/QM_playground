import numpy as np
from scipy.integrate import quad
from scipy.special import ai_zeros, airy

# First and tenth zeros of Ai
zeros, _, _, _ = ai_zeros(10)
z1, z10 = -zeros[0], -zeros[9]

# Normalisation integrals
int1, _ = quad(lambda t: airy(t - z1)[0] ** 2, 0, np.inf)
int10, _ = quad(lambda t: airy(t - z10)[0] ** 2, 0, np.inf)

# <x> and <x^2> for psi_1
x1, _ = quad(lambda t: t * airy(t - z1)[0] ** 2 / int1, 0, np.inf)
x2_1, _ = quad(lambda t: t**2 * airy(t - z1)[0] ** 2 / int1, 0, np.inf)

# <x> and <x^2> for psi_10
x10, _ = quad(lambda t: t * airy(t - z10)[0] ** 2 / int10, 0, np.inf)
x2_10, _ = quad(lambda t: t**2 * airy(t - z10)[0] ** 2 / int10, 0, np.inf)

sigma_x1 = np.sqrt(x2_1 - x1**2)
sigma_x10 = np.sqrt(x2_10 - x10**2)

print(f"ψ_1:   <x>   = {x1:.5f}")
print(f"ψ_1:   <x²>  = {x2_1:.5f}")
print(f"ψ_1:   σ_x   = {sigma_x1:.5f}")
print(f"ψ_10:  <x>   = {x10:.5f}")
print(f"ψ_10:  <x²>  = {x2_10:.5f}")
print(f"ψ_10:  σ_x   = {sigma_x10:.5f}")

# Ai * Ai' integrals


def ai_aip(t, z):
    ai, aip, _, _ = airy(t - z)
    return ai * aip


p1, _ = quad(lambda t:  ai_aip(t, z1) / int1, 0, np.inf)
p2_1, _ = quad(lambda t: -(t-z1) * airy(t - z1)[0] ** 2 / int1, 0, np.inf)
p10, _ = quad(lambda t:  ai_aip(t, z10) / int10, 0, np.inf)
p2_10, _ = quad(lambda t: -(t-z10) * airy(t - z10)[0] ** 2 / int10, 0, np.inf)

sigma_p1 = np.sqrt(p2_1 - p1**2)
sigma_p10 = np.sqrt(p2_10 - p10**2)

print(f"\nψ_1:   ∫ t·Ai·Ai' / N   = {p1:.5f}")
print(f"ψ_1:   ∫ t²·Ai·Ai' / N  = {p2_1:.5f}")
print(f"ψ_1:   σ_p      = {sigma_p1:.5f}")
print(f"ψ_10:  ∫ t·Ai·Ai' / N   = {p10:.5f}")
print(f"ψ_10:  ∫ t²·Ai·Ai' / N  = {p2_10:.5f}")
print(f"ψ_10:  σ_p      = {sigma_p10:.5f}")

print(f"\nψ_1:   σ_x · σ_p = {sigma_x1 * sigma_p1:.5f}")
print(f"ψ_10:  σ_x · σ_p = {sigma_x10 * sigma_p10:.5f}")
