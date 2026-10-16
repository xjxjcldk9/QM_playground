import sympy as sp

a, hbar, m, g = sp.symbols('a hbar m g', positive=True)


def minimize(f, var, label):
    print(f"--- {label} ---")
    print(f"f({var})          =", sp.latex(f))

    df = sp.diff(f, var)
    print(f"f'({var})         =", sp.latex(df))

    crit = sp.solve(sp.Eq(df, 0), var)
    crit = [sp.simplify(c) for c in crit if c.is_positive]
    var_min = crit[0]
    print(f"{var}_min         =", sp.latex(var_min))

    d2f = sp.diff(f, var, 2)
    print(f"f''({var}_min)>0? :", sp.simplify(d2f.subs(var, var_min)).is_positive)  # confirms minimum

    f_min = sp.simplify(f.subs(var, var_min))
    print("f_min         =", sp.latex(f_min))
    print("f_min (num)   =", sp.nsimplify(f_min), "=", f_min.evalf(subs={hbar: 1, m: 1, g: 1}))
    print()
    return var_min, f_min


# Problem 8.5: f(a) = 12*hbar^2/(2*m*a^2) + m*g*(3*a/16) + m*g*(5*a/16)
f = 12 * hbar**2 / (2 * m * a**2) + m * g * (3 * a / 16) + m * g * (5 * a / 16)
f = sp.simplify(f)
minimize(f, a, "12 hbar^2/(2 m a^2) + m g (3a/16) + m g (5a/16)")
