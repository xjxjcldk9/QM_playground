from astropy import constants
from numpy import pi


def reduced_mass(m1, m2):
    return m1 * m2 / (m1 + m2)


def hf(gp, mp, me):
    a = br(reduced_mass(mp, me))
    return constants.mu0 * gp * constants.e.si**2 / (3 * pi * mp * me * a**3) * constants.hbar**2


def br(m):
    return 4 * pi * constants.eps0 * constants.hbar**2 / (m * constants.e.si**2)


print('=======muonic hydrogen=====')
print(hf(5.59, constants.m_p, constants.m_e * 207).to('eV'))
print('=======positronium =====')
print(hf(2, constants.m_e, constants.m_e).to('eV'))
print('=======muonium =====')
print(hf(2, constants.m_e * 207, constants.m_e).to('eV'))
