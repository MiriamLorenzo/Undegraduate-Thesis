import numpy as np
import matplotlib.pyplot as plt

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

# plt.rcParams['text.usetex'] = True
# plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 10})
plt.rcParams["figure.figsize"] = (12, 5)
def set_basics(title, axes, markzero = False):
    fig, ax = plt.subplots()
    plt.title(title)
    if markzero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    return fig, ax

m = 1

from dispersionfunction import disp_F, Tdisp_F, Ldisp_F
from dispersionrelation import biggest_root

def comparison_lineplot(x, ys, prelabel, labels, axes, title, posx, posy, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.2f'%(labels[i]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    # The settings below are hardcoded
    ax.set_ylim([1e-10, 1e-5])
    ax.set_yscale('log')
    plt.text(posx, posy, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')

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

def Tlambdamax(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l):
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

def Llambdamax(alpha, m, k_vals, n_k, l_vals, n_l):
    k_range = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    lmax = 0
    kmax = 0
    for k in k_range:
        func_Dlk = lambda l : Ldisp_F(l, m, k, alpha) # Definition of a single variable function D(l).
        l = biggest_root(func_Dlk, l_range)
        if l > lmax:
            lmax = l
            kmax = k
    return lmax, kmax


# Range of k's where we look for the maximum of the dispersion relation l(k). Could be improved.
#def get_k_vals(m, q, eps = 0.05):
#    return [-q*(m+eps/3), -q*(m-eps)]
def get_k_vals(m, q, eps = 0.01):
    return [-1.1*m*q,0]

ms = [1,2,3,4]
qs = np.linspace(0.2, 0.8, 20)

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.1f'%(labels[i]))
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

def Tmax_growth_plot(datax, datay, n, mm, taus):
    prelabel = r'$\tau$ = '
    labels = taus
    axes = [r'$C_{nm}$', r'$\lambda_{max}$']
    title = r'Max. growth rate comparison for $(n,m) = (%1.0f, %1.0f)$'%(n, mm)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)