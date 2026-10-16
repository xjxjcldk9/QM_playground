from scipy import constants as con


T = 1 / con.k * (con.h**2 / 3 / 4 / con.u)**(3/5) * (con.atm)**(2/5)


l = con.h / (3 * con.u * con.k * 3)**(1/2)


print(l)
