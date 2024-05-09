# Cylindrical coordinates, with Br = 0, and Btheta, Bz depending only on r. This guarantees div(B) = 0.
# B0 is the max magnetic field, expressed in Teslas.
# r is the normalized distance to the axis (divided by R).

import scipy as sp
from scipy import special

B0 = 1

def LBz(r, alpha):
    '''Axial component of the magnetic field.'''
    return B0*sp.special.jv(0, alpha*r, out=None)

def LdBz(r, alpha):
    '''Derivative of the axial component of the magnetic field.'''
    return B0*sp.special.jv(1, alpha*r, out=None)

def LBtheta(r, alpha):
    '''Azimuthal component of the magnetic field.'''
    return -B0*sp.special.jv(1, alpha*r, out=None)*alpha

def LdBtheta(r, alpha):
    '''Derivative of the azimuthal component of the magnetic field.'''
    return B0*(sp.special.jv(0, alpha*r, out=None)-sp.special.jv(2, alpha*r, out=None))*alpha/2