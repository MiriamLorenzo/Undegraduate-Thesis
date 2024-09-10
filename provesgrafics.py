import matplotlib.pyplot as plt
import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D

N = 10

[obs_B0_min, obs_B0_max] = [10e-9,100e-9]
Au = 1.496e11
[obs_R_min, obs_R_max] = [1e-7*Au,1e-2*Au]
[obs_np_min, obs_np_max] = [10,1e6]

obs_B0 = 10**np.linspace(np.log10(obs_B0_min),np.log10(obs_B0_max),10)
obs_R = 10**np.linspace(np.log10(obs_R_min),np.log10(obs_R_max),10)
obs_np = 10**np.linspace(np.log10(obs_np_min),np.log10(obs_np_max),10)

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

M = np.zeros(shape=(N,N,N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            M[i][j][k] = np.log10(T_days(1, obs_B0[i], obs_R[j], obs_np[k]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color legend
counter = range(N)
X, Y, Z = np.meshgrid(obs_B0, obs_R, obs_np)
log_X, log_Y, log_Z = np.log10(X), np.log10(Y), np.log10(Z)
scatter = ax.scatter(log_X, log_Y, log_Z, c=M.flat, cmap='viridis')  # You can choose any colormap

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('$\log_{10}(T_{\mathrm{days}})$')

ax.set_xlabel('$\log_{10}(B_0)$')
ax.set_ylabel('$\log_{10}(R)$')
ax.set_zlabel('$\log_{10}(n_p)$')

plt.show()