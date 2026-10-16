import sympy as sp

a, hbar, m, alpha = sp.symbols('a hbar m alpha', positive=True)


def minimize(f, var, label):
    print(f"--- {label} ---")
    print("f({v})          =".format(v=var), sp.latex(f))

    df = sp.diff(f, var)
    print("f'({v})         =".format(v=var), sp.latex(df))

    crit = sp.solve(sp.Eq(df, 0), var)
    crit = [sp.simplify(c) for c in crit if c.is_positive]
    var_min = crit[0]
    print(f"{var}_min         =", sp.latex(var_min))

    d2f = sp.diff(f, var, 2)
    print(f"f''({var}_min)>0? :", sp.simplify(d2f.subs(var, var_min)).is_positive)  # confirms minimum

    f_min = sp.simplify(f.subs(var, var_min))
    print("f_min         =", sp.latex(f_min))
    print()
    return var_min, f_min


# Problem 8.3: f(a) = 12*hbar^2/(2*m*a^2) - alpha*(a^2/4)*(12/a^3)
f = 12 * hbar**2 / (2 * m * a**2) - alpha * (a**2 / 4) * (12 / a**3)
f = sp.simplify(f)
minimize(f, a, "12 hbar^2/(2 m a^2) - alpha (a^2/4)(12/a^3)")
