import sympy as sp

b, hbar, m, omega = sp.symbols('b hbar m omega', positive=True)


def minimize(f, label):
    print(f"--- {label} ---")
    print("f(b)          =", sp.latex(f))

    df = sp.diff(f, b)
    print("f'(b)         =", sp.latex(df))

    b_min = sp.simplify(sp.solve(sp.Eq(df, 0), b)[0])
    print("b_min         =", sp.latex(b_min))

    d2f = sp.diff(f, b, 2)
    print("f''(b_min)>0? :", sp.simplify(d2f.subs(b, b_min)).is_positive)  # confirms minimum

    f_min = sp.simplify(f.subs(b, b_min))
    print("f_min         =", sp.latex(f_min))
    print()
    return b_min, f_min


# Problem 8.2: f(b) = hbar^2/(2m) * (2b^3/pi)(pi/4b^5) + (1/2)*m*omega^2 * (2b^3/pi)(pi/2b)
f = (hbar**2 / (2 * m)) * (2 * b**3 / sp.pi) * (sp.pi / (4 * b**5)) \
    + sp.Rational(1, 2) * m * omega**2 * (2 * b**3 / sp.pi) * (sp.pi / (2 * b))
f = sp.simplify(f)
minimize(f, "hbar^2/(4 m b^2) + (1/2) m omega^2 b^2")
