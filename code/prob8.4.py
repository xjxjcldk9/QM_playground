import sympy as sp

b, hbar, m, omega = sp.symbols('b hbar m omega', positive=True)


def minimize(f, var, label):
    print(f"--- {label} ---")
    print(f"f({var})          =", sp.latex(f))

    df = sp.diff(f, var)
    print(f"f'({var})         =", sp.latex(df))

    var_min = sp.simplify(sp.solve(sp.Eq(df, 0), var)[0])
    print(f"{var}_min         =", sp.latex(var_min))

    d2f = sp.diff(f, var, 2)
    print(f"f''({var}_min)>0? :", sp.simplify(d2f.subs(var, var_min)).is_positive)  # confirms minimum

    f_min = sp.simplify(f.subs(var, var_min))
    print("f_min         =", sp.latex(f_min))
    print()
    return var_min, f_min


# Problem 8.4: f(b) = 3*b*hbar^2/(2m) + 3*m*omega^2/(8b)
f = 3 * b * hbar**2 / (2 * m) + 3 * m * omega**2 / (8 * b)
minimize(f, b, "3 b hbar^2/(2m) + 3 m omega^2/(8b)")
