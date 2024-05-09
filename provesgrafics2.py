import matplotlib.pyplot as plt
import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D
from thresholdfinder import critical_q, Tcritical_Cnm, obtain_taus, obtain_taus1, obtain_taus2, obtain_taus3, obtain_alphas, obtain_alphas2, comparison_lineplot

Cnms = np.linspace(0.01,1.6,30)
taus = np.linspace(1.01,5,30)
X, Y = np.meshgrid(taus, Cnms)

alphas = np.linspace(1, 6, 20)
qs = np.linspace(0.01, 1, 20)
#X, Y = np.meshgrid(alphas, qs)

Z = np.loadtxt('sheet_lambdamaxCC.csv', delimiter=',')
#Z = np.loadtxt('sheet_lambdamaxGH.csv', delimiter=',')
Z = [np.log10(z) for z in Z]


m = 1
[n, mm] = [1, 0]

plots = []

fig, axs = plt.subplots(figsize=(11, 9.5))
im = axs.imshow(Z, cmap='viridis', extent=[X.min(), X.max(), Y.min(), Y.max()],
                       origin='lower', aspect='auto')

plots.append(im)

# Normalize the colorbar based on the overall min and max values
for im in plots:
    im.set_clim(vmin=-14, vmax=3)

# Create a single colorbar for all the plots
cbar = fig.colorbar(plots[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.03)

m = 1

datax = [obtain_taus()[1:], obtain_taus()[1:], obtain_taus()[1:]]
datay = [np.loadtxt('sheet_cnms_nm10.csv', delimiter=',')[1:], np.loadtxt('sheet_cnms_nm10_1e-2.csv', delimiter=',')[1:], np.loadtxt('sheet_cnms_nm10_1.csv', delimiter=',')[1:]] #np.loadtxt('sheet_cnms_nm11.csv', delimiter=','), 
prelabel = r'$\lambda$ = '
labels = [r'$10^{-7}$',r'$10^{-2}$',r'1']
axes = [r'$\tau$', r'$C_{10}$']
title = 'Iso-value curves for the CC model'

for i in range(len(datay)):
    axs.plot(datax[i],datay[i],label=prelabel + labels[i]) #"{:.0e}".format(labels[i]))
axs.legend(fancybox=True, shadow=True)

plt.show()
