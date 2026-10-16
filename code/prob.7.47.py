from astropy import constants
from numpy import pi

E = constants.mu0 * 1.71 * constants.e.si**2 / \
    (3 * pi * (constants.m_p + constants.m_n) *
     constants.m_e * constants.a0**3) * 3/2 * constants.hbar**2


print(E.to('eV'))
print((constants.c/(E / constants.h)).to('cm'))
