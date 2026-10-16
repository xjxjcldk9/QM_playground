"""
Problem 8.10: Hydrogen molecule ion H2+  (LCAO / variational treatment).

Building blocks (x = R/a is the proton separation in Bohr radii):

    I(x)    = e^{-x} (1 + x + x^2/3)          overlap integral
    D(x)    = 1/x - (1 + 1/x) e^{-2x}         direct integral
    X(x)    = (1 + x) e^{-x}                  exchange integral

Total energy in units of -E1 = 13.6 eV, including proton-proton repulsion 2/x:

    F(x) = -( 1 + 2 (D - X) / (1 - I) ) + 2/x

Below: simplify F(x) symbolically and print it as LaTeX, then plot it.
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.symbols('x', positive=True)

I = sp.exp(-x) * (1 + x + x**2 / 3)
D = 1 / x - (1 + 1 / x) * sp.exp(-2 * x)
X = (1 + x) * sp.exp(-x)

F = -(1 + 2 * (D - X) / (1 - I)) + 2 / x

F_simpl = sp.simplify(F)
F_ratio = sp.cancel(sp.together(F))          # single rational function of x, e^x

print("F(x) simplified:")
print("   ", F_simpl)
print()
print("LaTeX (grouped exponentials):")
print("   ", sp.latex(F_simpl))
print()
print("LaTeX (single rational fraction in e^x):")
print("   ", sp.latex(F_ratio))
print()
print("limits:  x -> 0+ :", sp.limit(F_simpl, x, 0, '+'),
      "   x -> oo :", sp.limit(F_simpl, x, sp.oo))

# ---- plot -------------------------------------------------------------------
f = sp.lambdify(x, F_simpl, "numpy")
xs = np.linspace(0.15, 12.0, 2000)
ys = f(xs)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xs, ys, color="tab:blue", lw=2,
        label=r"$F(x) = -\left(1 + \dfrac{2(D-X)}{1-I}\right) + \dfrac{2}{x}$")
ax.axhline(-1, color="gray", ls="--", lw=1,
           label=r"asymptote  $F \to -1$  ($x\to\infty$)")
ax.set_xlabel(r"$x = R/a$")
ax.set_ylabel(r"$F(x)$   (total energy in units of $-E_1 = 13.6$ eV)")
ax.set_title(r"Problem 8.10: H$_2^+$ energy vs. proton separation")
ax.set_ylim(-2.0, 3.0)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("prob8.10.png", dpi=150)
print("saved prob8.10.png")
