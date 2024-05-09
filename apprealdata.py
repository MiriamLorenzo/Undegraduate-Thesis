import numpy as np
import math

from maxgrowth import lambdamax, Tlambdamax, get_k_vals

l_vals = [-13,-1]
n_k = 100
n_l = 50

mp = 1.673e-27 # proton mass in kg
ratio = 1.16 # ratio between proton density and mass density (see equation 1.13 in thesis)
mu0 = 4*math.pi*1e-7 # magnetic permitivity
sec = 86400 # number of seconds in a day

def T_days(l, obs_B0, obs_R, obs_np):
    sigma = np.sqrt(l)
    T = 1/sigma
    obs_rho = ratio*mp*obs_np*1e6
    factor = (obs_R/obs_B0)*np.sqrt(mu0*obs_rho)
    T_d = T*factor/sec
    return T_d

def instability_times(ms, obs_B0, obs_R, obs_np, obs_alpha, obs_q):
    for m in ms:
        k_vals = get_k_vals(m, obs_q)
        [l, k] = lambdamax(obs_alpha, obs_q, m, k_vals, n_k, l_vals, n_l)
        T_d = T_days(l, obs_B0, obs_R, obs_np)
        print('m =', m)
        print('lambda =', l, ', k =', k)
        print('alpha =', obs_alpha, ', q =', obs_q)
        print('Instability time:', np.round(T_d, 1), 'days, (', np.round(T_d/365, 1), 'years )')

# Maximum axial magnetic field in teslas
obs_B0 = 50e-9 

# Radius in meters
Au = 1.496e12
obs_R = 1e-2*Au

# Number of protons per cm^3.
obs_np = 20

# Magnetic field parameters
obs_alpha = 2
obs_q = 1

# Desired instabilities
ms = [1]#[1,2,3,4]
instability_times(ms, obs_B0, obs_R, obs_np, obs_alpha, obs_q)


