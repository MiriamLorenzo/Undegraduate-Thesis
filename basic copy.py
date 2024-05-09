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
#%matplotlib inline

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

# The magnetic fields are in a separate file.
from mag_field_GH import Bz, Btheta, dBz, dBtheta
from mag_field_Teresa import TBz, TBtheta, TdBz, TdBtheta

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

# AUXILIARY FUNCTIONS:

def F(r, m, k, q, alpha):
    return k*Bz(r, q, alpha) + m*Btheta(r, q, alpha)/r

def TF(r, n, mm, m, k, Cnm, tau):
    return k*TBz(r, n, mm, Cnm, tau) + m*TBtheta(r, n, mm, Cnm, tau)/r

def f(r, m, k, l, q, alpha):
    return r**3*(F(r, m, k, q, alpha)**2 + l)/(m**2 + (k**2)*(r**2))

def Tf(r, n, mm, m, k, l, Cnm, tau):
    return r**3*(TF(r, n, mm, m, k, Cnm, tau)**2 + l)/(m**2 + (k**2)*(r**2))

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

def F_euler(r, y, m, k, l, q, alpha):
    aux = (g(r, m, k, l, q, alpha)*y[0] - df(r, m, k, l, q, alpha)*y[1])/f(r, m, k, l, q, alpha)
    return np.array([y[1], aux])

def TF_euler(r, y, n, mm, m, k, l, Cnm, tau):
    aux = (Tg(r, n, mm, m, k, l, Cnm, tau)*y[0] - Tdf(r, n, mm, m, k, l, Cnm, tau)*y[1])/Tf(r, n, mm, m, k, l, Cnm, tau)
    return np.array([y[1], aux])

def bound_cond_D(m, k, l, xiR, dxiR, q, alpha):
    mod2 = Bz(R, q, alpha)**2 + Btheta(R, q, alpha)**2
    aux = ((m**2 + k**2*R**2)*special.kn(m, abs(k)*R))/(abs(k)*R*special.kn(m-1, abs(k)*R) + m*special.kn(m, abs(k)*R))
    return xiR*(k**2*mod2 + l*(1+aux)) + dxiR*R*(F(R, m, k, q, alpha)**2 + l)

def Tbound_cond_D(n, mm, m, k, l, xiR, dxiR, Cnm, tau):
    mod2 = TBz(R, n, mm, Cnm, tau)**2 + TBtheta(R, n, mm, Cnm, tau)**2
    aux = ((m**2 + k**2*R**2)*special.kn(m, abs(k)*R))/(abs(k)*R*special.kn(m-1, abs(k)*R) + m*special.kn(m, abs(k)*R))
    return xiR*(k**2*mod2 + l*(1+aux)) + dxiR*R*(TF(R, n, mm, m, k, Cnm, tau)**2 + l)

def lineplot(x, y, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y)

def comp_lineplot(x, y, axes, title, labels, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y[0], label = r'CC for $n,m$ = %1.0f,%1.0f, $C_{nm}$ = %1.1f, $\tau$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]))
    ax.plot(x,y[1], label = r'GH for $q$ = %1.1f, $\alpha$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[6], labels[7],labels[8], labels[9]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')

# Evaluation of the dispersion function.
# If printsol = True, we plot the solution xi_r.
def disp_F(l, m, k, q, alpha, printsol = False):
    aux_euler2 = lambda y, r : F_euler(r, y, m, k, l, q, alpha)
    rr = np.linspace(R0, Rf, n_ite+1)
    sol2 = integrate.odeint(aux_euler2, y0, rr)
    xiR2, dxiR2 = sol2[-1][0], sol2[-1][1]
    if printsol:
        axes = [r'$r$', r'$\xi_r$']
        title = 'Most unstable perturbation'
        perturbation = [elem[0] for elem in sol2]
        lineplot(rr, perturbation/abs(xiR2), axes, title, markzero = True)
        print('xi(0)= ', sol2[0][0], ', xi\'(0)= ', sol2[0][1])
        print('xi(R)= ', xiR2, ', xi\'(R)= ', R*dxiR2)
    return bound_cond_D(m, k, l, xiR2, dxiR2, q, alpha)

def Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = False):
    aux_euler2 = lambda y, r : TF_euler(r, y, n, mm, m, k, l, Cnm, tau)
    rr = np.linspace(R0, Rf, n_ite+1)
    sol2 = integrate.odeint(aux_euler2, y0, rr)
    xiR2, dxiR2 = sol2[-1][0], sol2[-1][1]
    if printsol:
        axes = [r'$r$', r'$\xi_r$']
        title = 'Most unstable perturbation'
        perturbation = [elem[0] for elem in sol2]
        lineplot(rr, perturbation/abs(xiR2), axes, title, markzero = True)
        print('xi(0)= ', sol2[0][0], ', xi\'(0)= ', sol2[0][1])
        print('xi(R)= ', xiR2, ', xi\'(R)= ', R*dxiR2)
    return Tbound_cond_D(n, mm, m, k, l, xiR2, dxiR2, Cnm, tau)

def model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, norm = True, info = False):
    aux_euler1 = lambda y, r : TF_euler(r, y, n, mm, m, k1, l1, Cnm, tau)
    aux_euler2 = lambda y, r : F_euler(r, y, m, k2, l2, q, alpha)
    rr = np.linspace(R0, Rf, n_ite+1)
    sol1 = integrate.odeint(aux_euler1, y0, rr)
    sol2 = integrate.odeint(aux_euler2, y0, rr)
    xiR1, dxiR1 = sol1[-1][0], sol1[-1][1]
    xiR2, dxiR2 = sol2[-1][0], sol2[-1][1]
    
    axes = [r'$r$', r'$\xi_r$']
    title = r'Most unstable perturbation for $m$ = %1.0f' %(m)
    perturbation1 = [elem[0] for elem in sol1]
    perturbation2 = [elem[0] for elem in sol2]
    labels = [n, mm, Cnm, tau, k1, l1, q, alpha, k2, l2]
    #perturbation = [perturbation1, perturbation2]
    perturbation = [perturbation1/max(perturbation1), perturbation2/max(perturbation2)] if norm else [perturbation1, perturbation2]
    comp_lineplot(rr, perturbation, axes, title, labels, markzero = True)
    
    if info:
        print(r'For the CC model:')
        print(r'$n$ = ', n, r', $mm$ = ', m, r', $C_{nm} = $', Cnm, r', $\tau$ = ', tau)
        print(r'$\lambda$ = ',l1, r', $k$ = ', k1)
        print ('Dispersion function = ', Tbound_cond_D(n, mm, m, k1, l1, xiR1, dxiR1, Cnm, tau))
        print('xi(0)= ', sol1[0][0], ', xi\'(0)= ', sol1[0][1])
        print('xi(R)= ', xiR1, ', xi\'(R)= ', R*dxiR1)

        print(r'For the GH model:')
        print(r'$q$ = ', q, r'$\alpha$ = ', alpha)
        print(r'$\lambda$ = ',l2, r', $k$ = ', k2)
        print ('Dispersion function = ', bound_cond_D(m, k2, l2, xiR2, dxiR2, q, alpha))
        print('xi(0)= ', sol2[0][0], ', xi\'(0)= ', sol2[0][1])
        print('xi(R)= ', xiR2, ', xi\'(R)= ', R*dxiR2)

"""
# How to evaluate the dispersion function D(l;k), given (alpha,q), (m,k) a value l for lambda.
# In this case, we also choose to plot
alpha = 1.1; q = 0.
m = 1
n = 1; mm = 0; Cnm = 1.7; tau = 1.1

k = -3
l = 5.29e-3
model_dispF_comparison(l, n, mm, m, k, Cnm, tau, q, alpha)
#print('Dispersion function = ', Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = True))
#print('Dispersion function = ', disp_F(l, m, k, q, alpha, printsol = True))
#"""

def lineplot(x, y, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y)
    ax.set_ylim([-1e2, 1e2])
    #plt.text(0, 2.73, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')

def plot_dispersion_function(alpha, q, m, k, ls, normalized = True):
    D = []
    for l in ls:
        D.append(disp_F(l, m, k*q, q, alpha))
    print(D)
    print(min(D))
    D = D/np.mean([abs(x) for x in D]) if normalized else D
    axes = ['$\lambda$', '$D(\lambda; k)$']
    title = r'Disp. fun. for $(\alpha, q) = (%1.1f, %1.1f)$, $k/q=%1.2f$'%(alpha, q, k)
    lineplot(ls, D, axes, title, markzero = True)

def Tplot_dispersion_function(n, mm, Cnm, tau, m, k, ls, normalized = True):
    D = []
    for l in ls:
        D.append(Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = False))
    print(D)
    print(min(D))
    D = D/np.mean([abs(x) for x in D]) if normalized else D
    axes = ['$\lambda$', '$D(\lambda; k)$']
    title = r'Disp. fun. for $(C_{nm}, \tau) = (%1.1f, %1.1f)$, $k=%1.2f$'%(Cnm, tau, k)
    lineplot(ls, D, axes, title, markzero = True)

"""
m = 1
alpha = 1.2 #1
q = 1.2 #0.30
k = -0.99
lmax = -5
lmin = -7
lambdas = 10**np.linspace(lmin,lmax,100)
plot_dispersion_function(alpha, q, m, k, lambdas, normalized = False)
#"""

def comparison_lineplot(x, ys, prelabel, labels, axes, title, posx, posy, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.5f'%(labels[i]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    # The settings below are hardcoded
    ax.set_ylim([-2.2, 3.15])
    plt.text(posx, posy, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')

def plot_dispersion_functions(alpha, q, m, ks, ls, posx, posy):
    datax = []
    datay = []
    for k in ks:
        D = []
        for l in ls:
            D.append(disp_F(l, m, k*q, q, alpha))
        datax.append(ls)
        #avg = np.mean([abs(x) for x in D]) # The dispersion function is being normalized for the comparison.
        #datay.append(D/avg)
        datay.append(D/abs(D[-1]))
    prelabel = r'$k/q$ = '
    axes = ['$\lambda$', 'normalized $D(\lambda; k)$']
    title = r'Dispersion function for $(\alpha, q) = (%1.1f, %1.1f)$'%(alpha, q)
    comparison_lineplot(datax, datay, prelabel, ks, axes, title, posx, posy, markzero = True)

def Tplot_dispersion_functions(n, mm, Cnm, tau, m, ks, ls, posx, posy):
    datax = []
    datay = []
    for k in ks:
        D = []
        for l in ls:
            D.append(Tdisp_F(l, n, mm, m, k, Cnm, tau)) 
        datax.append(ls)
        #avg = np.mean([abs(x) for x in D]) # The dispersion function is being normalized for the comparison.
        #datay.append(D/avg)
        datay.append(D/abs(D[-1]))
    prelabel = r'$k$ = '
    axes = ['$\lambda$', 'normalized $D(\lambda; k)$']
    title = r'Dispersion function for $(n,m) = (%1.0f, %1.0f)$ and $(C_{nm},\tau) = (%1.1f, %1.1f)$'%(n, mm, Cnm, tau)
    comparison_lineplot(datax, datay, prelabel, ks, axes, title, posx, posy, markzero = True)

"""
posy = 2.73
posx = 1.27e-7

m = 1

#tau=1.1,Cnm=1.2 
# ks = np.linspace(-3.53,-3.6,10)
# lambdas = np.linspace(3.7e-4,3.76e-4,300)

#tau=1.2,Cnm=1.2
# ks = np.linspace(-3.33,-3.38/q,10)
# lambdas = np.linspace(3e-5,3.05e-5,300)

#tau = 1.3
#Cnm = 1.2 
#ks = np.linspace(-2.77,-2.73,10)
#lambdas = np.linspace(3e-7,5e-7,100)

#tau = 1.2
#Cnm = 1.3 
#ks = np.linspace(-3.17*tau,-3.12*tau,10)
#lambdas = np.linspace(4e-7,6e-7,100)
#n = 1
#mm = 0 
#Tplot_dispersion_functions(n, mm, Cnm, tau, m, ks, lambdas, posx, posy)

alpha = 4 #1.5 #1.1
q = 0.8 #0.9 #0.6

ks = np.linspace(-1.01,-0.97,10) #np.linspace(-4.00/q,-4.05/q,10)#np.linspace(-0.999,-0.98,10)#(-0.986,-0.978,5) # Multiples of q
lambdas = 10**np.linspace(-13,-5,100)#10**np.linspace(-10,-4,200)#(0.4e-7,1.4e-7,30)
plot_dispersion_functions(alpha, q, m, ks, lambdas, posx, posy)

"#""
m = 2
ks = np.linspace(-1.984,-1.976,5) # Multiples of q
lambdas = np.linspace(0.2e-7,1.0e-7,30)
posx = 0.895e-7
plot_dispersion_functions(alpha, q, m, ks, lambdas, posx, posy)
#"""

""" #COMPROVAR TOT OKAY
r = 0.5
k = -3
l = 1e-8
l1 = l
l2 = l
k1 = k
k2 = k
y = [1, 2]
xiR = 1
dxiR = 2
ls = 10**np.linspace(-10,-6,100)
print(Bz(r, q, alpha) - TBz(r, n, mm, Cnm, tau))
print(dBz(r, q, alpha) - TdBz(r, n, mm, Cnm, tau))
print(Btheta(r, q, alpha) - TBtheta(r, n, mm, Cnm, tau))
print(dBtheta(r, q, alpha) - TdBtheta(r, n, mm, Cnm, tau))
print(F(r, m, k, q, alpha) - TF(r, n, mm, m, k, Cnm, tau))
print(f(r, m, k, l, q, alpha) - Tf(r, n, mm, m, k, l, Cnm, tau))
print(df(r, m, k, l, q, alpha) - Tdf(r, n, mm, m, k, l, Cnm, tau))
print(g(r, m, k, l, q, alpha) - Tg(r, n, mm, m, k, l, Cnm, tau))
print(F_euler(r, y, m, k, l, q, alpha) - TF_euler(r, y, n, mm, m, k, l, Cnm, tau))
print(bound_cond_D(m, k, l, xiR, dxiR, q, alpha) - Tbound_cond_D(n, mm, m, k, l, xiR, dxiR, Cnm, tau))
print(disp_F(l, m, k, q, alpha, printsol = False) - Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = False))
model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, norm = True, info = False)
plot_dispersion_function(alpha, q, m, k, ls, normalized = True)
Tplot_dispersion_function(n, mm, Cnm, tau, m, k, ls, normalized = True)
"""

def lininterp(x0, y0, x1, y1, x2):
    m = (y1-y0)/(x1-x0)
    y2 = y1 + m*(x2-x1)
    return y2

# Finds the biggest root of a function in a given interval.
def biggest_root(func_Dlk, l_range):
    D_lk = []
    for l in l_range:
        D_lk.append(func_Dlk(l))
    all_same_sign = all(x > 0 for x in D_lk) or all(x < 0 for x in D_lk)
    if not all_same_sign:
        i = len(D_lk) - 1
        # We are only interested in finding the biggest zero.
        while np.sign(D_lk[i])*np.sign(D_lk[i-1]) > 0:
            i = i - 1
        # Linear interpolation to find the zero. We could do a dicotomic search, but it would be slower.
        return lininterp(D_lk[i-1], l_range[i-1], D_lk[i], l_range[i], 0)
    else:
        return 0
    
# Finds the dispersion relation between k and lambda, given the magnetic field configuration.
def dispersion_relation(alpha, q, m, k_vals, n_k, l_vals, n_l):
    ks = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    ls = []
    #for k in ks:
    for i in range(len(ks)):
        if i%10 == 0: print(i)
        k = ks[i]
        D = []
        func_D = lambda l : disp_F(l, m, k, q, alpha) # Definition of a single variable function D(l).
        l = biggest_root(func_D, l_range)
        ls.append(l)
    return ks, ls

def Tdispersion_relation(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l):
    ks = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    ls = []
    #for k in ks:
    for i in range(len(ks)):
        if i%10 == 0: print(i)
        k = ks[i]
        D = []
        func_D = lambda l : Tdisp_F(l, n, mm, m, k, Cnm, tau) # Definition of a single variable function D(l).
        l = biggest_root(func_D, l_range)
        ls.append(l)
    return ks, ls

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.2f'%(labels[i][0]) + ',' + r' %1.2f'%(labels[i][1]))
        ax.set_ylim([1e-9, 1e-3])
        ax.set_yscale('log')
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')

def m_compare_dispersion_relation(ms, l_vals = [-14,-5], n_k = 60, n_l = 20):
    datax = []
    datay = []
    for m in ms:
        alpha = m[0]
        q = m[1]
        m = 1
        k_vals = [-1.02*q,-0.85*q]
        #k_vals = [-1.05*m*q,-0.9*m*q] #[-1.3*m*q,-0.5*m*q] #[-1.1*m*q,-0.8*m*q] centrat al voltant del pic de m = 1
        [k_range, ls] = dispersion_relation(alpha, q, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range/q))
        datay.append(ls)
        print(ls)
    prelabel = r'$\alpha, q$ = '
    labels = ms
    axes = ['$k/q$', '$\lambda$']
    title = r'Disp. relation for m = 1'#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)

def Tm_compare_dispersion_relation(n, mm, ms, l_vals = [-14,-5], n_k = 60, n_l = 20):
    datax = []
    datay = []
    for m in ms:
        Cnm = m[0]
        tau = m[1]
        m = 1
        k_vals = [-4.2,-1.8]
        #k_vals = [-1.05*m*q,-0.9*m*q] #[-1.3*m*q,-0.5*m*q] #[-1.1*m*q,-0.8*m*q] centrat al voltant del pic de m = 1
        [k_range, ls] = Tdispersion_relation(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range))
        datay.append(ls)
        print(ls)
    prelabel = r'$C_{nm}, \tau$ = '
    labels = ms
    axes = ['$k$', '$\lambda$']
    title = r'Disp. relation for m = 1'#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)

"""
parameters = [[1.2,1.2], [1.23,1.23], [1.2,1.23], [1.23,1.2], [1.27,1.2], [1.2,1.27]]
[n, mm] = [1, 0]
Tm_compare_dispersion_relation(n, mm, parameters, l_vals = [-9,-3], n_k = 100, n_l = 20)
parameters = [[1.2,0.7], [1.7,0.7], [2.4,0.7], [3.5,0.8], [4,0.8]]
m_compare_dispersion_relation(parameters, l_vals = [-9,-3], n_k = 60, n_l = 20)
#qs = [1.1, 1.2, 1.3]
#alphas = [1.1, 1.2, 1.3]
#for q in qs:
#    for alpha in alphas:
#        m_compare_dispersion_relation(ms, alpha, q, l_vals = [-9,-3], n_k = 200)
#q = 1.3 #0.5
#alpha = 1.3 #1.1
#m_compare_dispersion_relation(ms, alpha, q, l_vals = [-9,0], n_k = 100)
#"""

# Finds the dispersion relation for several magnetic field configurations, and plots them in the same figure.
# "eps" allows to focus in a given range of values of k, centered at -q.
# "l_vals" is the range of exponents in base 10 for lambda which is being considered to look for a solution.
# Thus, it may have to be adjusted when changing the magnetic field configuration.
# "n_k" and "n_l" can be modified to change on the degree of accuracy.
def q_compare_dispersion_relation(alpha, qs, m, posx, posy, eps = 0.05, l_vals = [-14,-3], n_k = 60, n_l = 30):
    datax = []
    datay = []
    for q in qs:
        k_vals = [-q,-q*0.97]#[-q*1.01,-q*(1-0.4*q)]#[-q*(m+eps/4), -q*(m-eps)][-q*2,0]#
        [k_range, ls] = dispersion_relation(alpha, q, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range/q))
        datay.append(ls)
    prelabel = r'$q$ = '
    labels = qs
    axes = ['$k/q$', '$\lambda$']
    title = r'Dispersion relation for $\alpha =%f$'%(alpha)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title, posx, posy)

def Tq_compare_dispersion_relation(n, mm, Cnm, taus, m, posx, posy, l_vals = [-14,-3], n_k = 60, n_l = 30):
    datax = []
    datay = []
    for tau in taus:
        k_vals = [-4,-2]#[-q*1.01,-q*(1-0.4*q)]#[-q*(m+eps/4), -q*(m-eps)][-q*2,0]#
        [k_range, ls] = dispersion_relation(alpha, q, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range/q))
        datay.append(ls)
    prelabel = r'$q$ = '
    labels = qs
    axes = ['$k/q$', '$\lambda$']
    title = r'Dispersion relation for $\alpha =%f$'%(alpha)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title, posx, posy)

def comparison_lineplot(x, ys, prelabel, labels, axes, title, posx, posy, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.2f'%(labels[i]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    # The settings below are hardcoded
    ax.set_ylim([0, 1e-6])
    #ax.set_yscale('log')
    plt.text(posx, posy, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')

"""
alpha = 1.7
qs = [0.6]#, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #[0.26, 0.28, 0.30, 0.32, 0.34]
posy = 2.9e-7

m=1
posx = -0.9715
q_compare_dispersion_relation(alpha, qs, m, posx, posy, eps = 0.5, l_vals = [-15,-6], n_k = 31)
"#""
m = 2
posx = -1.9675
q_compare_dispersion_relation(alpha, qs, m, posx, posy, eps = 0.04, l_vals = [-10,-6], n_k = 40)
#"""

def lambdamax(alpha, q, m, k_vals, n_k, l_vals, n_l):
    k_range = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    lmax = 0
    kmax = 0
    for k in k_range:
        func_Dlk = lambda l : disp_F(l, m, k, q, alpha) # Definition of a single variable function D(l).
        l = biggest_root(func_Dlk, l_range)
        if l > lmax:
            lmax = l
            kmax = k
    return lmax, kmax

def Tlambdamax(n, mm, Cnm, tau, k_vals, n_k, l_vals, n_l):
    k_range = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    lmax = 0
    kmax = 0
    for k in k_range:
        func_Dlk = lambda l : Tdisp_F(l, n, mm, m, k, Cnm, tau) # Definition of a single variable function D(l).
        l = biggest_root(func_Dlk, l_range)
        if l > lmax:
            lmax = l
            kmax = k
    return lmax, kmax

# Range of k's where we look for the maximum of the dispersion relation l(k). Could be improved.
def get_k_vals(m, q, eps = 0.05):
    return [-q*(m+eps/3), -q*(m-eps)]

ms = [1,2,3,4]
qs = np.linspace(0.2, 0.8, 20)

"""
#alpha = 1; l_vals = [-9,-3]
alpha = 1.3; l_vals = [-14,-3]
#alpha = 1.5; l_vals = [-12,-3]

n_k = 20
n_l = 30
lambdamaxs = []
for m in ms:
    this_curve = []
    print('Obtaining curve for m =', m)
    for q in qs:
        k_vals = get_k_vals(m, q)
        l = lambdamax(alpha, q, m, k_vals, n_k, l_vals, n_l)[0]
        this_curve.append(l)
        print(r'For q = ' + str(np.round(q,3)) + ', lambda = ' + str(l))
    print('')
    lambdamaxs.append(this_curve)
    
# Save as csv
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', lambdamaxs, delimiter=',')
"""

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.0f'%(labels[i]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    # The settings below are hardcoded
    plt.grid()
    ax.set_yscale('log')
    ax.set_ylim([1e-12, 2e-4]) #ax.set_ylim([1e-9, 2e-4])

def max_growth_plot(datay, alpha):
    datax = [qs, qs, qs, qs]
    datay = datay
    prelabel = r'$m$ = '
    labels = ms
    axes = [r'$q$', r'$\lambda_{max}$']
    title = 'Max. growth rate comparison'
    title = r'Max. growth rate comparison for $\alpha =%1.1f$'%(alpha)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)

"""
data1 = np.loadtxt('sheet_alpha1.csv', delimiter=',')
data2 = np.loadtxt('sheet_alpha1.5.csv', delimiter=',')
data3 = np.loadtxt('sheet_alpha1.3.csv', delimiter=',')

max_growth_plot(data1, 1)
max_growth_plot(data2, 1.5)
max_growth_plot(data3, 1.3)
"""

def critical_q(alpha, l0, qmin, qmax, eps = 0.05, tol = 0.01, n_k = 40, n_l = 10):
    l_vals = [np.log10(l0)-0.5, np.log10(l0)+0.5]
    while (qmax - qmin > tol):
        #print(round(qmin,2), round(qmax,2))
        qmid = (qmin + qmax)/2
        k_vals = get_k_vals(m, qmid, eps = eps)
        lmid = lambdamax(alpha, qmid, m, k_vals, n_k, l_vals, n_l)[0]
        if (lmid < l0):
            qmin = qmid
        else:
            qmax = qmid
    q = (qmin + qmax)/2
    return q

"""
m = 1
alpha = 1
l0 = 1e-7
[qmin, qmax] = [0.2, 0.8]
q = critical_q(alpha, l0, qmin, qmax)
print('Instability: m =', m)
print('Threshold: l0 =', l0)
print('Force parameter: alpha =', alpha)
print('Critical twist parameter: q =', q)
"""

def obtain_alphas(N = 30, alpha_min = 1, alpha_1 = 1.05, alpha_2 = 1.2, alpha_max = 1.5):
    a1 = np.linspace(alpha_min, alpha_1, int(N/3))
    a2 = np.linspace(alpha_1 + (alpha_2 - alpha_1)*3/N, alpha_2, int(N/3))
    a3 = np.linspace(alpha_2 + (alpha_max - alpha_2)*3/N, alpha_max, int(N/3))
    return np.concatenate((a1,a2,a3))

def obtain_alphas2(N = 50, alpha_min = 1, alpha_1 = 1.05, alpha_2 = 1.2, alpha_max = 1.5):
    a1 = np.linspace(alpha_min, alpha_1, int(N/3))
    a2 = np.linspace(alpha_1 + (alpha_2 - alpha_1)*3/N, alpha_2, int(N/3))
    a3 = np.linspace(alpha_2 + (alpha_max - alpha_2)*3/N, alpha_max, int(N/3))
    a4 = np.linspace(alpha_max * (1 + 3/N), alpha_max*4, N)
    return np.concatenate((a1,a2,a3, a4))


### corba limit
"""
# Small alphas
m = 1
#alphas = obtain_alphas()
#alphas = [2.95]#np.linspace(2.6,3.6,50)
alphas = obtain_alphas2()
ls = [7e-8]#[4e-8, 7e-8, 1e-7]
qs = [[0.25, 0.8]]#[[0.20, 0.55], [0.25, 0.60], [0.25, 0.60]]

crit_qs = []
for i in range(len(ls)):
    this_curve = []
    print('Obtaining curve for lambda =', ls[i])
    for alpha in alphas:
        q = critical_q(alpha, ls[i], qs[i][0], qs[i][1], tol = 0.01)
        this_curve.append(q)
        print(r'For alpha = ' + str(np.round(alpha,3)) + ', q = ' + str(np.round(q,3)))
    print('')
    crit_qs.append(this_curve)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', crit_qs, delimiter=',')
#"""

"""
from joblib import Parallel, delayed

m = 1
alphas = np.linspace(2.6,3.6,2)
ls = [7e-8]#[4e-8, 7e-8, 1e-7]
qs = [[0.25, 0.8]]#[[0.20, 0.55], [0.25, 0.60], [0.25, 0.60]]

crit_qs = []
for i in range(len(ls)):
    this_curve = np.zeros(alphas.shape[0])
    print('Obtaining curve for lambda =', ls[i])
    this_curve = Parallel(n_jobs=8)(delayed(critical_q)(alpha, ls[i], qs[i][0], qs[i][1], tol = 0.002) for alpha in alphas)
    
    #pels prints que són ràpids
    for j, alpha in enumerate(alphas):
    	print(r'For alpha = ' + str(np.round(alpha,3)) + ', q = ' + str(np.round(this_curve[j],3)))
    print('')
    crit_qs.append(this_curve)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', crit_qs,delimiter=',')
"""

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + "{:.0e}".format(labels[i]))
    #ax.legend(fancybox=True, shadow=True)
    #plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    plt.text(1.438, 0.267, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')
    ax.set_ylim([0.25, 0.8])#([0.25, 0.58])

"""
m = 1
#datax = [np.linspace(1,1.6,50),np.linspace(1.6,2.6,50),np.linspace(2.6,3.6,50)] #[np.linspace(1,5,90), np.linspace(1,5,90)]
#datay = [np.loadtxt('sheet_m1_1-1.6.csv', delimiter=','),np.loadtxt('sheet_m1_1.6-2.6.csv', delimiter=','),np.loadtxt('sheet_m1_2.6-3.6.csv', delimiter=',')]
datax = obtain_alphas2() #[np.linspace(1,5,90), np.linspace(1,5,90)]
datay = np.loadtxt('sheet_m1l1e-7.csv', delimiter=',')
datax = datax[:-8]
datay = datay[:-8]
#np.concatenate((np.loadtxt('sheet_m1v4.csv', delimiter=','), np.loadtxt('sheet_m1v3.csv', delimiter=',')),axis=0)
datax = [datax, datax]
datay = [datay, datay]
prelabel = r'$\lambda_0$ = '
labels = [7e-8, 7e-8]
axes = [r'$\alpha$', r'$q$']
title = 'Stability curve for the threshold $\lambda_0 = 1e-7$'
comparison_lineplot(datax, datay, prelabel, labels, axes, title)
#     ax.ticklabel_format(style='plain')
#"""
"""
m = 1
datax = [obtain_alphas(), obtain_alphas(), obtain_alphas()]
datay = np.loadtxt('sheet_m1.csv', delimiter=',')
prelabel = r'$\lambda_0$ = '
labels = [4e-8, 7e-8, 1e-7]
axes = [r'$\alpha$', r'$q$']
title = 'Stability curve for different thresholds $\lambda_0$'
comparison_lineplot(datax, datay, prelabel, labels, axes, title)
#     ax.ticklabel_format(style='plain')
#"""

"""
# Small alphas
l = 1e-7
alphas = obtain_alphas()
ms = [1, 2, 3]
qs = [[0.25, 0.60], [0.2, 0.75], [0.2, 0.5]]
crit_qs = []
for i in range(len(ms)):
    this_curve = []
    m = ms[i]
    print('Obtaining curve for m =', m)
    for alpha in alphas:
        q = critical_q(alpha, l, qs[i][0], qs[i][1], tol = 0.002)
        this_curve.append(q)
        print(r'For alpha = ' + str(np.round(alpha,3)) + ', q = ' + str(np.round(q,3)))
    print('')
    crit_qs.append(this_curve)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', crit_qs, delimiter=',')
"""

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    colors = [u'#2ca02c', u'#ff7f0e', u'#1f77b4'] # inverted color cycle, to match the green line in the previous
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.0f'%(labels[i]), color = colors[i])
    ax.legend(fancybox=True, shadow=True)
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    plt.text(1.394, 0.267, r"$\lambda_0$ = {:.0e}".format(l), bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')
    ax.set_ylim([0.25, 0.58])

"""
l = 10**(-7)
datax = [obtain_alphas(), obtain_alphas(), obtain_alphas()]
datay = np.loadtxt('sheet_l1e-7.csv', delimiter=',')
prelabel = r'$m$ = '
labels = [1, 2, 3]
axes = [r'$\alpha$', r'$q$']
title = 'Stability curve for different instabilities $m$'
comparison_lineplot(datax, datay, prelabel, labels, axes, title)
"""

import matplotlib.patheffects as mpe

def line_colorplot(xi, yi, Z, label, ticks, cmap, x, ys, prelabel, labels, axes): 
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (6.5, 7.5)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    # Contourf:
    xi = np.array(xi); yi = np.array(yi); Z = np.array(Z)
    X, Y = np.meshgrid(xi, yi)
    im = plt.contourf(X, Y, Z, 100, cmap=plt.get_cmap(cmap))
    cb = plt.colorbar(im , ax = [ax], location = 'top', label=label)
    cb.set_ticks(ticks) 
    
    # Lines:
    #colors = [u'#2ca02c', u'#ff7f0e', u'#1f77b4'] # inverted color cycle, to match the green line in the previous
    pe1 = [mpe.Stroke(linewidth=3, foreground='black'), mpe.Stroke(foreground='black',alpha=1),mpe.Normal()]
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.0f'%(labels[i]), linewidth = 2.3, path_effects=pe1)
    ax.legend(fancybox=True, shadow=True)
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    ax.set_ylim([0.25, 0.58])
    plt.show()

    # Lines that will be plotted

"""
l = 10**(-7)
datax = [obtain_alphas(), obtain_alphas(), obtain_alphas()]
datay = np.loadtxt('sheet_l1e-7.csv', delimiter=',')
prelabel = r'$m$ = '
labels = [1, 2, 3]
axes = [r'$\alpha$', r'$q$']
N = 20
rr = np.linspace(0.01,1,100)
a = np.linspace(1,1.5,N)
q = np.linspace(0.25,0.58,N)
"""

def Lorentz(a, q, r):
    mu0 = 1
    return (2*q**2*r)/(mu0*(q**2*r**2 + 1)**3) - (q*r**a*((q*r**a)/(q**2*r**2 + 1) - (2*q**3*r**a*r**2)/(q**2*r**2 + 1)**2 + (a*q*r*r**(a - 1))/(q**2*r**2 + 1)))/(mu0*r*(q**2*r**2 + 1))

def Twist(a, q, r):
    return q*r**(a-1)

def avg(f, param1, param2):
    rr = np.linspace(0.01,1,100)
    return np.mean(2*np.multiply(rr, np.abs(f(param1, param2, rr))))

"""
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
"""

def get_k_vals(m, q):
    return [-1.3*m*q,-0.5*m*q]

l_vals = [-7,-1]
n_k = 20
n_l = 40

def lambdamax(alpha, q, m, k_vals, n_k, l_vals, n_l):
    k_range = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    lmax = 0
    kmax = 0
    for k in k_range:
        func_Dlk = lambda l : disp_F(l, m, k, q, alpha) # Definition of a single variable function D(l).
        l = biggest_root(func_Dlk, l_range)
        if l > lmax:
            lmax = l
            kmax = k
    return lmax, kmax

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
        l = lambdamax(obs_alpha, obs_q, m, k_vals, n_k, l_vals, n_l)[0]
        T_d = T_days(l, obs_B0, obs_R, obs_np)
        print('m =', m)
        print('Instability time:', np.round(T_d, 1), 'days')

# Maximum axial magnetic field in teslas
obs_B0 = 47e-9 

# Radius in meters
obs_R = 1.6e10

# Number of protons per cm^3.
obs_np = 20

# Magnetic field parameters
obs_alpha = 1.82
obs_q = 1.25


# Desired instabilities
ms = [1,2,3,4]
#instability_times(ms, obs_B0, obs_R, obs_np, obs_alpha, obs_q)

def Teresa_to_GH(n, m, Cnm, tau, qs, alphas, N):
    p = 1
    dist = float('inf')
    qmin = 0;
    alphamin = 0;
    r = np.linspace(0, 1, N)
    B_z = TBz(r, n, m, Cnm, tau)
    B_theta = TBtheta(r, n, m, Cnm, tau)
    for q in qs:
        for alpha in alphas:
            newB_z = Bz(r, q, alpha)
            newB_theta = Btheta(r, q, alpha)
            new_dist = np.sqrt(linalg.norm(newB_z - B_z, p)**2 + linalg.norm(newB_theta - B_theta,p)**2)
            if (new_dist < dist):
                dist = new_dist
                qmin = q
                alphamin = alpha
    return [qmin, alphamin]

def GH_to_Teresa(n, m, Cnms, taus, q, alpha, N):
    p = 1
    dist = float('inf')
    Cnmmin = 0;
    taumin = 0;
    r = np.linspace(0, 1, N)
    B_z = Bz(r, q, alpha)
    B_theta = Btheta(r, q, alpha)
    for Cnm in Cnms:
        for tau in taus:
            newB_z = TBz(r, n, m, Cnm, tau)
            newB_theta = TBtheta(r, n, m, Cnm, tau)
            new_dist = np.sqrt(linalg.norm(newB_z - B_z, p)**2 + linalg.norm(newB_theta - B_theta,p)**2)
            if (new_dist < dist):
                dist = new_dist
                Cnmmin = Cnm
                taumin = tau
    return [Cnmmin, taumin]

def model_comparison(r,n,m,Cnm,tau,q,alpha):
    plt.rcParams["figure.figsize"] = (7.5, 6)
    plt.figure()
    plt.title(r"CC ($Cnm = {}$, $\tau = {}$) adjusted by GH ($q = {}$, $\alpha = {}$)".format(np.round(Cnm,2), np.round(tau,2), np.round(q,2), np.round(alpha,2)))
    plt.xlabel('$r$')
    plt.ylabel(r"Magnetic field component")
    plt.plot(r, TBtheta(r, n, m, Cnm, tau), 'g')
    plt.plot(r, TBz(r, n, m, Cnm, tau), 'b')
    plt.plot(r, Btheta(r, q, alpha), 'g--')
    plt.plot(r, Bz(r, q, alpha), 'b--')
    plt.legend([r"$B_\theta$", "$B_z$"])
    #plt.savefig(str(uuid.uuid4()))
    plt.show()

def change_of_model_lineplot(ys, xs, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        #ax.plot(xs,y1,label=r'$q$')
        ax.plot(xs,ys[i],label=prelabel + r' %1.2f'%(labels[i]))#, color = colors[i])#label=r'$\alpha$')
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')

def compare_change_of_model(n, m, Cnms, taus, qs, alphas, N = 100):
    datay2 = []
    datay1 = []
    for Cnm in Cnms:
        print(r'Computing for $C_{nm}$ = %1.2f' %(Cnm))
        datay1aux = []
        datay2aux = []
        for tau in taus:
            [q_approx, alpha_approx] = Teresa_to_GH(n,m,Cnm,tau,qs,alphas,N)
            datay1aux.append(q_approx)
            datay2aux.append(alpha_approx)
        datay2.append(datay2aux)
        datay1.append(datay1aux)
    prelabel = r'$Cnm$ = '
    labels = Cnms
    axes = [r'$\tau$',r'q']
    title = r'GH parameter $q$' #for $C_{nm} = %1.2f$'%(Cnm)#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    change_of_model_lineplot(datay1, taus, prelabel, labels, axes, title)
    axes = [r'$\tau$',r'$\alpha$']
    title = r'GH parameter $\alpha$' #for $C_{nm} = %1.2f$'%(Cnm)#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    change_of_model_lineplot(datay2, taus, prelabel, labels, axes, title)

"""
N = 100;
[n,mm] = [1,0]
Cnms = np.linspace(0.5,2,5) #1.78 # [0.5,2]
#Cnm = 2
taus = np.linspace(1e-4,5,50) #1.1 # [0,4]
qs = np.linspace(0, 6, 200)
alphas = np.linspace(1e-4, 10, 400)
compare_change_of_model(n, mm, Cnms, taus, qs, alphas, N = 100)
        
#print('Best q: ' + str(np.round(q_approx,2)))
#print('Best alpha: ' + str(np.round(alpha_approx,2)))

r = np.linspace(0, 1, N)
#model_comparison(r,n,m,Cnm,tau,q_approx,alpha_approx)
#"""

"""
alpha = 1.1; q = 0.1
m = 1
n = 1; mm = 1; Cnm = 1.1; tau = 1.2

N = 100
qs = np.linspace(0, 6, 200)
alphas = np.linspace(1e-4, 10, 400)
[q, alpha] = Teresa_to_GH(n, mm, Cnm, tau, qs, alphas, N)

n_k = 100
n_l = 50
l_vals = [-5, -2]
k_vals1 = [-5, -4]
k_vals2 = [-1.5, -0.5]

[l1, k1] = Tlambdamax(n, mm, Cnm, tau, m, k_vals1, n_k, l_vals, n_l)
[l2, k2] = lambdamax(alpha, q, m, k_vals2, n_k, l_vals, n_l) 

#k = -1.5
#l = 5.29e-3
model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, norm = False, info = True)
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

def plot_lim_Teresa(xs, ys, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(xs,ys)

"""
m = 1
datax = obtain_alphas2() #[np.linspace(1,5,90), np.linspace(1,5,90)]
datay = np.loadtxt('sheet_m1l1e-7.csv', delimiter=',')
alphas = datax[:-8]
qs = datay[:-8]

axes = [r'$\alpha$',r'$q$']
title = 'GH thresholds $\lambda_0 = 1e-7$'
plot_lim_Teresa(alphas, qs, axes, title, markzero = False)

N = 100
n = 1
mm = 0
Cnms = np.linspace(0.2,4,400)
taus = np.linspace(1,15,600)
Cnm = []
tau = []
for i in range(len(qs)):
    [Cnm_approx, tau_approx] = GH_to_Teresa(n, mm, Cnms, taus, qs[i], alphas[i], N)
    Cnm.append(Cnm_approx)
    tau.append(tau_approx)
axes = [r'$C_{nm}$',r'$\tau$']
title = 'Teresa equiv point for the GH thresholds $\lambda_0 = 1e-7$'
plot_lim_Teresa(Cnm, tau, axes, title, markzero = False)

qs = np.linspace(0, 0.8, 200)
alphas = np.linspace(1e-4, 10, 600)
q = []
alpha = []
for i in range(len(Cnm)):
    [q_approx, alpha_approx] = Teresa_to_GH(n, mm, Cnm[i], tau[i], qs, alphas, N)
    q.append(q_approx)
    alpha.append(alpha_approx)
axes = [r'$C_{nm}$',r'$\tau$']

axes = [r'$\alpha$',r'$q$']
title = 'GH thresholds $\lambda_0 = 1e-7$ bis'
plot_lim_Teresa(alpha, q, axes, title, markzero = False)

prelabel = r'$\lambda_0$ = '
labels = [4e-8, 7e-8, 1e-7]
axes = [r'$\alpha$', r'$q$']
title = 'Stability curve for different thresholds $\lambda_0$'
#comparison_lineplot(datax, datay, prelabel, labels, axes, title)
#     ax.ticklabel_format(style='plain')
#"""

q = 0.3
alpha = 1
#print(GH_to_Teresa(n, mm, Cnms, taus, q, alpha, N))

plt.show()

print ('end')