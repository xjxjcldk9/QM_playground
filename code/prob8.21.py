"""
Problem 8.21: Variational bound on the hydrogen ground state if the photon
had a small mass M. Coulomb's law is then screened into a Yukawa potential,
V(r) = -(e^2/4*pi*eps0) * e^{-mu r}/r,  mu = M c / hbar (the photon's inverse
Compton wavelength). Using the trial function psi = A e^{-b r} (the exact
form of the true hydrogen ground state, so this is just as good a trial
function as any) gives, after doing the <T> and <V> integrals,

    E(b) = (hbar^2/2m_e) * 4*pi * (b^3/pi) * (1/4b)
           - (e^2/epsilon_0) * (b^3/pi) * 1/(mu + 2b)^2

Task: minimize E(b) over b. Unlike 8.19/8.20, mu appears, so there is no
closed form for b_min(mu) -- instead expand the minimum perturbatively in
the small dimensionless parameter lambda = mu*a (a = Bohr radius), and
extract the leading-order shift in the ground state energy due to a
nonzero photon mass.
"""
import sympy as sp

hbar, m_e, e, eps0, b, mu = sp.symbols(
    'hbar m_e e epsilon_0 b mu', positive=True)

# --- E(b) exactly as given ----------------------------------------------------
T = (hbar**2 / (2 * m_e)) * 4 * sp.pi * (b**3 / sp.pi) * (1 / (4 * b))
V = (e**2 / eps0) * (b**3 / sp.pi) * (1 / (mu + 2 * b)**2)
E = sp.simplify(T - V)

print("<T>(b)     =", sp.latex(sp.simplify(T)))
print("<V>(b)     =", sp.latex(-sp.simplify(V)))
print("E(b, mu)   =", sp.latex(E))
print()

dE = sp.simplify(sp.diff(E, b))
print("dE/db      =", sp.latex(dE))
print("Setting dE/db = 0 and clearing denominators gives a cubic in b for")
print("fixed mu -- no closed form b_min(mu) in general (unlike 8.19/8.20).")
print()

# --- nondimensionalize: x = a*b, lambda = a*mu, a = Bohr radius --------------
a = 4 * sp.pi * eps0 * hbar**2 / (m_e * e**2)
Hartree = hbar**2 / (m_e * a**2)          # = e^2/(4 pi eps0 a) = 2*13.6 eV
x, lam = sp.symbols('x lambda', real=True)

f = sp.simplify(E.subs({b: x / a, mu: lam / a}) / Hartree)
print("E/Hartree  =", sp.latex(f), " (with x = a*b, lambda = a*mu)")
print()

# --- perturbative minimization in small lambda -------------------------------
# ansatz x_min(lambda) = 1 + a1*lambda + a2*lambda^2 + a3*lambda^3 + a4*lambda^4
a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4', real=True)
x_ansatz = 1 + a1 * lam + a2 * lam**2 + a3 * lam**3 + a4 * lam**4

dfdx = sp.diff(f, x)
eq = sp.numer(sp.together(dfdx.subs(x, x_ansatz)))
eq = sp.expand(eq)
poly = sp.Poly(eq, lam)
coeffs = poly.all_coeffs()[::-1]

sol = {}
for c_n in coeffs:
    c_sub = sp.expand(c_n.subs(sol))
    if c_sub == 0:
        continue
    unk = [u for u in (a1, a2, a3, a4) if u in c_sub.free_symbols]
    if not unk:
        continue
    sol[unk[0]] = sp.solve(sp.Eq(c_sub, 0), unk[0])[0]

x_min = sp.simplify(x_ansatz.subs(sol))
print("x_min(lambda) =", sp.latex(x_min), " + O(lambda^5)")

f_min = sp.series(f.subs(x, x_min), lam, 0, 4).removeO()
f_min = sp.expand(f_min)
print("E_min/Hartree =", sp.latex(f_min), " + O(lambda^4)")
print()

# --- leading-order energy shift due to the photon mass ------------------------
# f_min = -1/2 + lambda + O(lambda^2)  =>  Delta E = Hartree * lambda + ...
slope = f_min.coeff(lam, 1)
print(f"leading shift coefficient  d(E_min/Hartree)/d(lambda) |_0 = {slope}")

M, c_light = sp.symbols('M c', positive=True)     # photon mass, speed of light
DeltaE = sp.simplify(Hartree * slope * (M * c_light / hbar) * a)
print("Delta E = E_min(mu) - E_min(0)  ~  ", sp.latex(DeltaE), " (to leading order in mu*a)")

alpha = e**2 / (4 * sp.pi * eps0 * hbar * c_light)   # fine structure constant
print("Delta E - alpha*M*c^2 simplifies to:", sp.simplify(DeltaE - alpha * M * c_light**2))
print()
print("==> Delta E_1  ~  alpha * M * c^2   (to leading order in mu*a << 1)")
print()

# --- numeric check / illustration ---------------------------------------------
c_num = 2.99792458e8
consts = {
    hbar: 1.054571817e-34,
    m_e:  9.1093837015e-31,
    e:    1.602176634e-19,
    eps0: 8.8541878128e-12,
    c_light: c_num,
}
eV = 1.602176634e-19
alpha_num = float(alpha.subs(consts))
print(f"alpha (fine structure constant) = {alpha_num:.6f}  (1/{1/alpha_num:.2f})")

# Using the PDG experimental upper bound on the photon mass, M < 1e-18 eV/c^2,
# as an illustrative example of the resulting energy shift:
M_bound_eV = 1e-18                       # eV/c^2, illustrative PDG-type bound
M_bound_kg = M_bound_eV * eV / c_num**2
DeltaE_eV = alpha_num * M_bound_eV       # Delta E = alpha * M c^2, in eV directly
print(f"If M_gamma < {M_bound_eV:.0e} eV/c^2 (~{M_bound_kg:.3e} kg), "
      f"then Delta E_1 < {DeltaE_eV:.3e} eV")
print("(utterly negligible compared to the 13.6 eV ground-state energy itself.)")
