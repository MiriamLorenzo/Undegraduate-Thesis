import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

N = 100

[obs_B0_min, obs_B0_max] = [10,100]
Au = 1.496e12
[obs_R_min, obs_R_max] = [1e-7*Au,1e-2*Au]
[obs_np_min, obs_np_max] = [10,1e6]

obs_B0 = np.linspace(obs_B0_min,obs_B0_max,N)
obs_R = 10**np.linspace(np.log10(obs_R_min),np.log10(obs_R_max),6)
obs_np = 10**np.linspace(np.log10(obs_np_min),np.log10(obs_np_max),N)

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

plt.rcParams.update({'font.size': 17})
l = 1
fig, axs = plt.subplots(2, 3, figsize=(11, 9.5), sharey=True)

# Create an empty list to store all the plots for colorbar
plots = []

# Variables to store the min and max values across all plots
#min_val, max_val = -1, 8
min_val, max_val = -6, 6

colors = ['#54278f', '#d73027', '#fee08b', '#4575b4', '#313695']
cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=256)

for j, R_value in enumerate(obs_R):
    M = np.zeros(shape=(N, N))
    for i in range(N):
        for k in range(N):
            #M[i][k] = np.log10(T_days(1, obs_B0[i]*1e-9, R_value, obs_np[k])*sec)
            M[i][k] = np.log10(T_days(l, obs_B0[i]*1e-9, R_value, obs_np[k]))

    X, Z = np.meshgrid(obs_B0, obs_np)


    im = axs[math.floor(j/3)%2][j%3].imshow(M, cmap=cmap_custom, extent=[X.min(), X.max(), Z.min(), Z.max()],
                       origin='lower', aspect='auto')

    axs[math.floor(j/3)%2][j%3].set_title(r'$R = 10^{%1.0f}$ Au' %(np.log10(R_value/Au)))
    
    if math.floor(j/3)%2 == 1: axs[math.floor(j/3)%2][j%3].set_xlabel('$B_0$ (nT)')
    if j%3 == 0: axs[math.floor(j/3)%2][j%3].set_ylabel('$n_p$ (cm$^{-3}$)')

    #axs[math.floor(j/3)%2][j%3].set_xscale('log')
    axs[math.floor(j/3)%2][j%3].set_yscale('log')

    #axs[math.floor(j/3)%2][j%3].set_xticks(10**np.linspace(np.log10(obs_B0_min),np.log10(obs_B0_max),2))  # Example ticks, adjust as needed
    #axs[math.floor(j/3)%2][j%3].get_xaxis().set_major_formatter(plt.ScalarFormatter())


    # Add the plot to the list
    plots.append(im)

# Normalize the colorbar based on the overall min and max values
for im in plots:
    im.set_clim(vmin=min_val, vmax=max_val)

# Create a single colorbar for all the plots
cbar = fig.colorbar(plots[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.03)
cbar.set_label(r'$\log_{10}(T_{\mathrm{days}})$')#$+\frac{1}{2}\log_{10}(\lambda)$')

fig.suptitle(r'Lifetimes of an instability for $\lambda = 1$')

plt.show()