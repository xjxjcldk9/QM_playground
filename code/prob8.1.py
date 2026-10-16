import sympy as sp

b, hbar, m, alpha, a = sp.symbols('b hbar m alpha a', positive=True)


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


# Problem 8.1: f(b) = hbar^2*b/(2m) + alpha/sqrt(2*pi*b)
# (variational energy for a Gaussian trial function on V(x) = -alpha*delta(x))
f1 = hbar**2 * b / (2 * m) + alpha / sp.sqrt(2 * sp.pi * b)
minimize(f1, "hbar^2 b/(2m) + alpha/sqrt(2*pi*b)")

# Second function: g(b) = hbar^2*b/(2m) + 3*a/(16*b^2)
f2 = hbar**2 * b / (2 * m) + 3 * a / (16 * b**2)
minimize(f2, "hbar^2 b/(2m) + 3a/(16 b^2)")
