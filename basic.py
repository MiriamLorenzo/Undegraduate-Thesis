import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from scipy import special, optimize, integrate, interpolate
#from scipy import special, math, optimize, integrate, interpolate
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import copy
import time
import random
import pandas as pd
import math
import uuid
from scipy.interpolate import interp1d
from numpy import linalg
import matplotlib.patheffects as mpe
#%matplotlib inline

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

# The magnetic fields are in a separate file.
from mag_field_GH import Bz, Btheta, dBz, dBtheta
from mag_field_Teresa import TBz, TBtheta, TdBz, TdBtheta
from mag_field_L import LBz, LBtheta, LdBz, LdBtheta

#%config InlineBackend.figure_format='retina' # Useful for running matplotlib on high-dpi displays.

#plt.rcParams['text.usetex'] = True
#plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 17})
plt.rcParams["figure.figsize"] = (6.5, 6)
def set_basics(title, axes, markzero = False):
    fig, ax = plt.subplots()
    plt.title(title)
    if markzero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    return fig, ax

# NOTE: Any plot can be saved by using plt.savefig.
# dpi = 1200 corresponds to extremely high resolution.
# Example:
# plt.savefig(str(uuid.uuid4())+'.png', format="png", dpi=1200)

# PARAMETERS (normalized; don't change)
n_ite = 250
y0 = [1, 0] # Initial condition for Euler-Lagrange
R = 1
R0, Rf = 1e-3*R, (1-1e-3)*R
m = 1

from auxiliaryfunctions import F, TF, f, Tf, df, Tdf, g, Tg, F_euler, TF_euler, bound_cond_D, Tbound_cond_D
from dispersionfunction import disp_F, Tdisp_F, model_dispF_comparison, plot_dispersion_function, Tplot_dispersion_function, plot_dispersion_functions, Tplot_dispersion_functions

# How to evaluate the dispersion function D(l;k), given (alpha,q), (m,k) a value l for lambda.
# In this case, we also choose to plot
# For given paramters, plots the perturbation on r (normalized) and computes the value of the dispersion function
"""
alpha = 1.1; q = 0.
m = 1
n = 1; mm = 0; Cnm = 1.7; tau = 1.1

k = -3
l = 5.29e-3
model_dispF_comparison(l, l, n, mm, m, k, k, Cnm, tau, q, alpha, info = True)
#print('Dispersion function = ', Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = True))
#print('Dispersion function = ', disp_F(l, m, k, q, alpha, printsol = True))
#"""

#Plots the dispersive funcion for a given set of parameters
"""
m = 1
[alpha, q] = [1.2, 1.2]
k = -0.99
[lmax, lmin] = [-5, -7]
lambdas = 10**np.linspace(lmin,lmax,100)
plot_dispersion_function(alpha, q, m, k, lambdas, normalized = False)
#"""

# Plots the dispersive function for diferent values of k
"""
posy = 2.73
posx = 1.27e-7

m = 1
#tau=1.1; Cnm=1.2; ks = np.linspace(-3.53,-3.6,10); lambdas = np.linspace(3.7e-4,3.76e-4,300)
#tau=1.2; Cnm=1.2; ks = np.linspace(-3.33,-3.38/q,10); lambdas = np.linspace(3e-5,3.05e-5,300)
#tau = 1.3; Cnm = 1.2; ks = np.linspace(-2.77,-2.73,10); lambdas = np.linspace(3e-7,5e-7,100)
[tau, Cnm] = [5, 0.4]
ks = np.linspace(-0.63,-0.59,10)
lambdas = 10**np.linspace(-13,-4,100)
[n, mm] = [1,1]
Tplot_dispersion_functions(n, mm, Cnm, tau, m, ks, lambdas, posx, posy)

#[alpha, q] = [4, 0.8]
#ks = np.linspace(-1.01,-0.97,10) # Multiples of q
#lambdas = 10**np.linspace(-13,-5,100)
#plot_dispersion_functions(alpha, q, m, ks, lambdas, posx, posy)
#"""
 
from dispersionrelation import biggest_root, dispersion_relation, Tdispersion_relation, compare_dispersion_relation, Tcompare_dispersion_relation

# Finds the dispersion relation for several magnetic field configurations, and plots them in the same figure.
# "l_vals" is the range of exponents in base 10 for lambda which is being considered to look for a solution.
# "n_k" and "n_l" can be modified to change on the degree of accuracy.
"""
parameters = [[5, 0.4], [5, 0.45], [5, 0.5], [5, 0.55]]  #[[1.2,1.2], [1.23,1.23], [1.2,1.23], [1.23,1.2], [1.27,1.2], [1.2,1.27]] #[tau, Cnm]
[n, mm] = [1, 1]
Tcompare_dispersion_relation(n, mm, parameters, l_vals = [-13,-4], n_k = 100, n_l = 20)
#parameters = [[1.2,0.7], [1.7,0.7], [2.4,0.7], [3.5,0.8], [4,0.8]] #[alpha, q]
#compare_dispersion_relation(parameters, l_vals = [-9,-3], n_k = 60, n_l = 20)
#"""

from maxgrowth import lambdamax, Tlambdamax, get_k_vals, max_growth_plot, Tmax_growth_plot

### findinglambdamax.py

# Plot of the max growth for diferent GH parameters
"""
data1 = np.loadtxt('sheet_alpha1.csv', delimiter=',')
data2 = np.loadtxt('sheet_alpha1.5.csv', delimiter=',')
data3 = np.loadtxt('sheet_alpha1.3.csv', delimiter=',')

max_growth_plot(data1, 1)
max_growth_plot(data2, 1.5)
max_growth_plot(data3, 1.3)
#"""

# Plot of the max growth for diferent CC parameters
"""
[n, mm] = [1, 0]
taus = [1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5]
datax = []
for tau in taus :
    datax.append(np.linspace(1.5/tau, 1.5/tau + 0.2, 10))
datay = np.loadtxt('sheet_lambdas_nm10.csv', delimiter=',')
Tmax_growth_plot(datax, datay, n, mm, taus)
#"""

from thresholdfinder import critical_q, Tcritical_Cnm, obtain_taus, obtain_taus1, obtain_taus2, obtain_taus3, obtain_alphas, obtain_alphas2, comparison_lineplot

### criticalpoints.py

from lorentzVStwist import line_colorplot, Lorentz, Twist, avg

#"""
# Twist
Z = np.empty((N, N))
for i in range(N):
    for j in range(N):
        Z[i][j] = avg(Twist, a[j], q[i])
label = 'Average twist'
ticks = np.linspace(0,1,11)
cmap = 'viridis'
line_colorplot(a, q, Z, label, ticks, cmap, datax, datay, prelabel, labels, axes)

# Lorentz
Z = np.empty((N, N))
for i in range(N):
    for j in range(N):
        Z[i][j] = avg(Lorentz, a[j], q[i])
label = 'Average Lorentz force'
ticks = np.linspace(0,0.2,11)
cmap = 'plasma'
line_colorplot(a, q, Z, label, ticks, cmap, datax, datay, prelabel, labels, axes)
#"""

from modelcomparison import Teresa_to_GH, Teresa_to_GH_bis, GH_to_Teresa, model_comparison, model_comparison_bis, compare_change_of_model, plot_lim_Teresa

### APPLICATION TO REAL DATA: apprealdata.py

# thresholdcomp.py

#plots the change of model for one MF configuration
"""
N = 100
[n,mm] = [1,0]
Cnms = np.linspace(0.5,2,5) #1.78 # [0.5,2]
#Cnm = 2
taus = np.linspace(1e-4,5,50) #1.1 # [0,4]
qs = np.linspace(0, 6, 200)
alphas = np.linspace(1e-4, 10, 400)
compare_change_of_model(n, mm, Cnms, taus, qs, alphas, N = 100)
#"""

"""
m = 1
n = 1; mm = 1; Cnm = 1.1; tau = 0.2
n_k = 100
n_l = 50
l_vals = [-1, 2]
k_vals = [-40, -10]
#print(Tlambdamax(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l))
lmax = 2e-3
lmin = 1e-4
ls = np.linspace(lmin,lmax,100)
k = -40
ks = [-600,-500,-400,-300]
for k in ks:
    Tplot_dispersion_function(n, mm, Cnm, tau, m, k, ls, normalized = False)
#"""


plt.show()

print ('end')