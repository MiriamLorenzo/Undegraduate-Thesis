# AUXILIARY FUNCTIONS:

import numpy as np
from scipy import special

R = 1

from mag_field_GH import Bz, Btheta, dBz, dBtheta
from mag_field_Teresa import TBz, TBtheta, TdBz, TdBtheta
from mag_field_L import LBz, LBtheta, LdBz, LdBtheta

def F(r, m, k, q, alpha):
    return k*Bz(r, q, alpha) + m*Btheta(r, q, alpha)/r
def TF(r, n, mm, m, k, Cnm, tau):
    return k*TBz(r, n, mm, Cnm, tau) + m*TBtheta(r, n, mm, Cnm, tau)/r
def LF(r, m, k, alpha):
    return k*LBz(r, alpha) + m*LBtheta(r, alpha)/r

def f(r, m, k, l, q, alpha):
    return r**3*(F(r, m, k, q, alpha)**2 + l)/(m**2 + (k**2)*(r**2))
def Tf(r, n, mm, m, k, l, Cnm, tau):
    return r**3*(TF(r, n, mm, m, k, Cnm, tau)**2 + l)/(m**2 + (k**2)*(r**2))
def Lf(r, m, k, l, alpha):
    return r**3*(LF(r, m, k, alpha)**2 + l)/(m**2 + (k**2)*(r**2))

def df(r, m, k, l, q, alpha):
    dF = k*dBz(r, q, alpha) + m*(dBtheta(r, q, alpha)/r - Btheta(r, q, alpha)/r**2)
    a = r**2*(F(r, m, k, q, alpha)**2 + l)*(3*m**2 + k**2*r**2)/((m**2 + k**2*r**2)**2)
    b = 2*F(r, m, k, q, alpha)*dF*r**3/(m**2 + k**2*r**2)
    return a + b
def Tdf(r, n, mm, m, k, l, Cnm, tau):
    dF = k*TdBz(r, n, mm, Cnm, tau) + m*(TdBtheta(r, n, mm, Cnm, tau)/r - TBtheta(r, n, mm, Cnm, tau)/r**2)
    a = r**2*(TF(r, n, mm, m, k, Cnm, tau)**2 + l)*(3*m**2 + k**2*r**2)/((m**2 + k**2*r**2)**2)
    b = 2*TF(r, n, mm, m, k, Cnm, tau)*dF*r**3/(m**2 + k**2*r**2)
    return a + b
def Ldf(r, m, k, l, alpha):
    dF = k*LdBz(r, alpha) + m*(LdBtheta(r, alpha)/r - LBtheta(r, alpha)/r**2)
    a = r**2*(LF(r, m, k, alpha)**2 + l)*(3*m**2 + k**2*r**2)/((m**2 + k**2*r**2)**2)
    b = 2*LF(r, m, k, alpha)*dF*r**3/(m**2 + k**2*r**2)
    return a + b

def g(r, m, k, l, q, alpha):
    den = m**2 + k**2*r**2
    a = (1-(m**2-k**2*r**2)/den**2)*r*(F(r, m, k, q, alpha)**2 + l)
    b1 = 4*m*F(r, m, k, q, alpha)*Btheta(r, q, alpha)/den
    b2 = 2*Btheta(r, q, alpha)*dBtheta(r, q, alpha) + 2*Bz(r, q, alpha)*dBz(r, q, alpha)
    b3 = 2*Btheta(r, q, alpha)**2/r*(2*F(r, m, k, q, alpha)**2/(F(r, m, k, q, alpha)**2 + l) - 1)
    b = k**2*r**2/den*(b1 + b2 + b3)
    return a - b
def Tg(r, n, mm, m, k, l, Cnm, tau):
    den = m**2 + k**2*r**2
    a = (1-(m**2-k**2*r**2)/den**2)*r*(TF(r, n, mm, m, k, Cnm, tau)**2 + l)
    b1 = 4*m*TF(r, n, mm, m, k, Cnm, tau)*TBtheta(r, n, mm, Cnm, tau)/den
    b2 = 2*TBtheta(r, n, mm, Cnm, tau)*TdBtheta(r, n, mm, Cnm, tau) + 2*TBz(r, n, mm, Cnm, tau)*TdBz(r, n, mm, Cnm, tau)
    b3 = 2*TBtheta(r, n, mm, Cnm, tau)**2/r*(2*TF(r, n, mm, m, k, Cnm, tau)**2/(TF(r, n, mm, m, k, Cnm, tau)**2 + l) - 1)
    b = k**2*r**2/den*(b1 + b2 + b3)
    return a - b
def Lg(r, m, k, l, alpha):
    den = m**2 + k**2*r**2
    a = (1-(m**2-k**2*r**2)/den**2)*r*(LF(r, m, k, alpha)**2 + l)
    b1 = 4*m*LF(r, m, k, alpha)*LBtheta(r, alpha)/den
    b2 = 2*LBtheta(r, alpha)*LdBtheta(r, alpha) + 2*LBz(r, alpha)*LdBz(r, alpha)
    b3 = 2*LBtheta(r, alpha)**2/r*(2*LF(r, m, k, alpha)**2/(LF(r, m, k, alpha)**2 + l) - 1)
    b = k**2*r**2/den*(b1 + b2 + b3)
    return a - b

def F_euler(r, y, m, k, l, q, alpha):
    aux = (g(r, m, k, l, q, alpha)*y[0] - df(r, m, k, l, q, alpha)*y[1])/f(r, m, k, l, q, alpha)
    return np.array([y[1], aux])
def TF_euler(r, y, n, mm, m, k, l, Cnm, tau):
    aux = (Tg(r, n, mm, m, k, l, Cnm, tau)*y[0] - Tdf(r, n, mm, m, k, l, Cnm, tau)*y[1])/Tf(r, n, mm, m, k, l, Cnm, tau)
    return np.array([y[1], aux])
def LF_euler(r, y, m, k, l, alpha):
    aux = (Lg(r, m, k, l, alpha)*y[0] - Ldf(r, m, k, l, alpha)*y[1])/Lf(r, m, k, l, alpha)
    return np.array([y[1], aux])

def bound_cond_D(m, k, l, xiR, dxiR, q, alpha):
    mod2 = Bz(R, q, alpha)**2 + Btheta(R, q, alpha)**2
    aux = ((m**2 + k**2*R**2)*special.kn(m, abs(k)*R))/(abs(k)*R*special.kn(m-1, abs(k)*R) + m*special.kn(m, abs(k)*R))
    return xiR*(k**2*mod2 + l*(1+aux)) + dxiR*R*(F(R, m, k, q, alpha)**2 + l)
def Tbound_cond_D(n, mm, m, k, l, xiR, dxiR, Cnm, tau):
    mod2 = TBz(R, n, mm, Cnm, tau)**2 + TBtheta(R, n, mm, Cnm, tau)**2
    aux = ((m**2 + k**2*R**2)*special.kn(m, abs(k)*R))/(abs(k)*R*special.kn(m-1, abs(k)*R) + m*special.kn(m, abs(k)*R))
    return xiR*(k**2*mod2 + l*(1+aux)) + dxiR*R*(TF(R, n, mm, m, k, Cnm, tau)**2 + l)
def Lbound_cond_D(m, k, l, xiR, dxiR, alpha):
    mod2 = LBz(R, alpha)**2 + LBtheta(R, alpha)**2
    aux = ((m**2 + k**2*R**2)*special.kn(m, abs(k)*R))/(abs(k)*R*special.kn(m-1, abs(k)*R) + m*special.kn(m, abs(k)*R))
    return xiR*(k**2*mod2 + l*(1+aux)) + dxiR*R*(LF(R, m, k, alpha)**2 + l)