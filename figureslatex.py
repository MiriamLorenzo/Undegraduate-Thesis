import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import j1
def integrand(x):
    return j1(x)
def integrate_bessel_j1(alphas):
    results = []
    for alpha in alphas:
        result, error = quad(integrand, 0, alpha)
        results.append(result)
    return results

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (5, 3.2)
def set_basics(title, axes, markzero = False):
    fig, ax = plt.subplots()
    plt.title(title, fontsize=16)
    if markzero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(axes[0], fontsize=14)
    ax.set_ylabel(axes[1], fontsize=14)
    #ax.legend(loc='best')
    return fig, ax

alphas = np.concatenate((np.linspace(0.1,0.7,30), np.linspace(0.8,5,40)))
lambdamaxs = np.loadtxt('sheet_lambdas_L.csv', delimiter=',')

axes = [r'$\alpha$',r'$\lambda$']
title = r'Maximum $\lambda$ for the Lundquist Model'

fig, ax = set_basics(title, axes, markzero = False)

ax.plot(alphas, lambdamaxs) #integrate_bessel_j1(alphas)/alphas
ax.set_ylim([1e-4, 50])
ax.set_yscale('log')

lambdamaxs = np.loadtxt('sheet_alpha1.csv', delimiter=',')[0]
qs = np.linspace(0.2, 0.8, 20)

plt.tight_layout()

axes = [r'$q$',r'$\lambda$']
title = r'Maximum $\lambda$ for the Gold-Hoyle Model'

fig, ax = set_basics(title, axes, markzero = False)

ax.plot(qs, lambdamaxs)#np.log(1+qs**2)/qs/2
ax.set_ylim([1e-9, 1e-4])
ax.set_yscale('log')

plt.tight_layout()

plt.show()
