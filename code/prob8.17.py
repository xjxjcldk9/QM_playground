import sympy as sp

a, hbar, m, omega = sp.symbols('a hbar m omega', positive=True)


def minimize(f, var, label, scale=None):
    print(f"--- {label} ---")
    print(f"f({var})          =", sp.latex(f))

    df = sp.diff(f, var)
    print(f"f'({var})         =", sp.latex(sp.simplify(df)))

    test = {hbar: 1, m: 1, omega: 1}
    crit = [sp.simplify(c) for c in sp.solve(sp.Eq(df, 0), var)
            if c.is_real is not False
            and complex(c.subs(test)).real > 0
            and abs(complex(c.subs(test)).imag) < 1e-12]
    var_min = crit[0]
    print(f"{var}_min         =", sp.latex(var_min))

    d2f = sp.diff(f, var, 2)
    print(f"f''({var}_min)>0? :", bool(
        sp.simplify(d2f.subs(var, var_min)).subs(test) > 0))

    f_min = sp.simplify(f.subs(var, var_min))
    print("f_min         =", sp.latex(f_min))
    if scale is not None:
        print(f"f_min / ({scale}) =", sp.simplify(f_min / scale),
              "=", sp.N(f_min.subs(test), 8))
    print()
    return var_min, f_min


# Problem 8.17: infinite-square-well ground state as trial function for the
# harmonic oscillator, minimized over the well half-width a.

# (a) standard oscillator potential  V = (1/2) m omega^2 x^2
f_a = sp.pi**2 * hbar**2 / (2 * a**2 * m) + (sp.pi **
                                             2 - 6) * a**2 * m * omega**2 / (24 * sp.pi**2)
minimize(f_a, a, "pi^2 hbar^2/(2 a^2 m) + (pi^2 - 6) a^2 m omega^2/(24 pi^2)",
         scale=hbar * omega)
print("   exact oscillator ground state = hbar*omega/2 = 0.5 hbar*omega\n")

# (b) second case
f_b = sp.pi**2 * hbar**2 / (2 * a**2 * m) + (2 * sp.pi **
                                             2 - 3) * a**2 * m * omega**2 / (12 * sp.pi**2)
minimize(f_b, a, "pi hbar^2/(2 a^2 m) + (2 pi^2 - 3) a^2 m omega/(12 pi^2)",
         scale=hbar * sp.sqrt(omega))
