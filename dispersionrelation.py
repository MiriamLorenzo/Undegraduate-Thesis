import numpy as np
import matplotlib.pyplot as plt

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

#plt.rcParams['text.usetex'] = True
#plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 10})
plt.rcParams["figure.figsize"] = (12, 4)
def set_basics(title, axes, markzero = False):
    fig, ax = plt.subplots()
    plt.title(title)
    if markzero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    return fig, ax

from dispersionfunction import disp_F, Tdisp_F, Ldisp_F

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

def Ldispersion_relation(alpha, m, k_vals, n_k, l_vals, n_l):
    ks = np.linspace(k_vals[0], k_vals[1], n_k)
    l_range = 10**(np.linspace(l_vals[0], l_vals[1], n_l)) # logarithmically spaced.
    ls = []
    #for k in ks:
    for i in range(len(ks)):
        if i%10 == 0: print(i)
        k = ks[i]
        D = []
        func_D = lambda l : Ldisp_F(l, m, k, alpha) # Definition of a single variable function D(l).
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

def comparison_lineplot_bis(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.2f'%(labels[i]))
        ax.set_ylim([1e-4, 1e-1])
        ax.set_yscale('log')
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')

def compare_dispersion_relation(param, l_vals = [-14,-5], n_k = 60, n_l = 20):
    datax = []
    datay = []
    for p in param:
        alpha = p[0]
        q = p[1]
        m = 1
        k_vals = [-1.02*q,-0.85*q]
        [k_range, ls] = dispersion_relation(alpha, q, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range/q))
        datay.append(ls)
        print(ls)
    prelabel = r'$\alpha, q$ = '
    labels = param
    axes = ['$k/q$', '$\lambda$']
    title = r'Disp. relation for m = 1'#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)

def Tcompare_dispersion_relation(n, mm, param, l_vals = [-14,-5], n_k = 60, n_l = 20):
    datax = []
    datay = []
    for p in param:
        tau = p[0]
        Cnm = p[1]
        m = 1
        k_vals = [-1,-0.2]
        #k_vals = [-1.05*m*q,-0.9*m*q] #[-1.3*m*q,-0.5*m*q] #[-1.1*m*q,-0.8*m*q] centrat al voltant del pic de m = 1
        [k_range, ls] = Tdispersion_relation(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range))
        datay.append(ls)
        print(ls)
    prelabel = r'$C_{nm}, \tau$ = '
    labels = param
    axes = ['$k$', '$\lambda$']
    title = r'Disp. relation for m = 1'#$\tau =%1.1f$, $C_{nm} =%1.1f$'%(alpha, q)
    comparison_lineplot(datax, datay, prelabel, labels, axes, title)

def Lcompare_dispersion_relation(alphas, l_vals = [-14,-5], n_k = 60, n_l = 20):
    datax = []
    datay = []
    for alpha in alphas:
        m = 1
        k_vals = [-0.3,0.3]
        [k_range, ls] = Ldispersion_relation(alpha, m, k_vals, n_k, l_vals, n_l)
        datax.append(list(k_range))
        datay.append(ls)
        print(ls)
    prelabel = r'$\alpha$ = '
    labels = alphas
    axes = ['$k$', '$\lambda$']
    title = r'Disp. relation for m = 1'
    comparison_lineplot_bis(datax, datay, prelabel, labels, axes, title)