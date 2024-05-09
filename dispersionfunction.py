import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

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

# PARAMETERS (normalized; don't change)
n_ite = 250
y0 = [1, 0] # Initial condition for Euler-Lagrange
R = 1
R0, Rf = 1e-3*R, (1-1e-3)*R
m = 1

from auxiliaryfunctions import F_euler, bound_cond_D, TF_euler, Tbound_cond_D, LF_euler, Lbound_cond_D

n_ite = 250
y0 = [1, 0] # Initial condition for Euler-Lagrange
R = 1
R0, Rf = 1e-3*R, (1-1e-3)*R
m = 1

def lineplot(x, y, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y)

def comp_lineplot(x, y, axes, title, labels, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y[0], label = r'CC for $(n,m)$ = (%1.0f,%1.0f), $C_{nm}$ = %1.1f, $\tau$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]))
    ax.plot(x,y[1], label = r'GH for $q$ = %1.1f, $\alpha$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[6], labels[7],labels[8], labels[9]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')

def comp_lineplot3(x, y, axes, title, labels, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(x,y[0], label = r'CC for $(n,m)$ = (%1.0f,%1.0f), $C_{nm}$ = %1.1f, $\tau$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]))
    ax.plot(x,y[1], label = r'GH for $q$ = %1.1f, $\alpha$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[6], labels[7],labels[8], labels[9]))
    ax.plot(x,y[2], label = r'L for $\alpha$ = %1.1f, $k$ = %1.2f, $\lambda$ = %1.7f' %(labels[10], labels[11],labels[12]))
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

def Ldisp_F(l, m, k, alpha, printsol = False):
    aux_euler2 = lambda y, r : LF_euler(r, y, m, k, l, alpha)
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
    return Lbound_cond_D(m, k, l, xiR2, dxiR2, alpha)

def model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, info = False, norm = True):
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
        print('')
        print(r'For the CC model:')
        print(r'(n,m) = (' + str(n) + r', ' + str(m) + r'), Cnm = ' + str(Cnm) + r', tau = ' + str(tau))
        print(r'lambda = ' + str(l1) + r', k = ' + str(k1))
        print ('Dispersion function = ', Tbound_cond_D(n, mm, m, k1, l1, xiR1, dxiR1, Cnm, tau))
        #print('xi(0)= ' + str(sol1[0][0]) + ', xi\'(0)= ' + str(sol1[0][1]))
        print('xi(R)= ' + str(xiR1) + ', xi\'(R)= ' + str(R*dxiR1))

        print('')
        print(r'For the GH model:')
        print(r'q = ' + str(q) + r', alpha = ' + str(alpha))
        print(r'lambda = ' + str(l2) + r', k = ' + str(k2))
        print ('Dispersion function = ', bound_cond_D(m, k2, l2, xiR2, dxiR2, q, alpha))
        #print('xi(0)= ' + str(sol2[0][0]) + ', xi\'(0)= ' + str(sol2[0][1]))
        print('xi(R)= ' + str(xiR2) + ', xi\'(R)= ' + str(R*dxiR2))

def model_dispF_3comparison(ls, n, mm, m, ks, Cnm, tau, q, alpha1, alpha2, info = False, norm = True):
    aux_euler1 = lambda y, r : TF_euler(r, y, n, mm, m, ks[0], ls[0], Cnm, tau)
    aux_euler2 = lambda y, r : F_euler(r, y, m, ks[1], ls[1], q, alpha1)
    aux_euler3 = lambda y, r : LF_euler(r, y, m, ks[2], ls[2], alpha2)
    rr = np.linspace(R0, Rf, n_ite+1)
    sol1 = integrate.odeint(aux_euler1, y0, rr)
    sol2 = integrate.odeint(aux_euler2, y0, rr)
    sol3 = integrate.odeint(aux_euler3, y0, rr)
    xiR1, dxiR1 = sol1[-1][0], sol1[-1][1]
    xiR2, dxiR2 = sol2[-1][0], sol2[-1][1]
    xiR3, dxiR3 = sol3[-1][0], sol3[-1][1]
    
    axes = [r'$r$', r'$\xi_r$']
    title = r'Most unstable perturbation for $m$ = %1.0f' %(m)
    perturbation1 = [elem[0] for elem in sol1]
    perturbation2 = [elem[0] for elem in sol2]
    perturbation3 = [elem[0] for elem in sol3]
    labels = [n, mm, Cnm, tau, ks[0], ls[0], q, alpha1, ks[1], ls[1], alpha2, ks[2], ls[2]]
    #perturbation = [perturbation1, perturbation2]
    perturbation = [perturbation1/max(perturbation1), perturbation2/max(perturbation2), perturbation3/max(perturbation3)] if norm else [perturbation1, perturbation2, perturbation3]
    comp_lineplot3(rr, perturbation, axes, title, labels, markzero = True)
    
    if info:
        print('')
        print(r'For the CC model:')
        print(r'(n,m) = (' + str(n) + r', ' + str(m) + r'), Cnm = ' + str(Cnm) + r', tau = ' + str(tau))
        print(r'lambda = ' + str(ls[0]) + r', k = ' + str(ks[0]))
        print ('Dispersion function = ', Tbound_cond_D(n, mm, m, ks[0], ls[0], xiR1, dxiR1, Cnm, tau))
        #print('xi(0)= ' + str(sol1[0][0]) + ', xi\'(0)= ' + str(sol1[0][1]))
        print('xi(R)= ' + str(xiR1) + ', xi\'(R)= ' + str(R*dxiR1))

        print('')
        print(r'For the GH model:')
        print(r'q = ' + str(q) + r', alpha = ' + str(alpha1))
        print(r'lambda = ' + str(ls[1]) + r', k = ' + str(ls[1]))
        print ('Dispersion function = ', bound_cond_D(m, ks[1], ls[1], xiR2, dxiR2, q, alpha1))
        #print('xi(0)= ' + str(sol2[0][0]) + ', xi\'(0)= ' + str(sol2[0][1]))
        print('xi(R)= ' + str(xiR2) + ', xi\'(R)= ' + str(R*dxiR2))

        print('')
        print(r'For the L model:')
        print(r'alpha = ' + str(alpha2))
        print(r'lambda = ' + str(ls[2]) + r', k = ' + str(ls[2]))
        print ('Dispersion function = ', Lbound_cond_D(m, ks[2], ls[2], xiR2, dxiR2, alpha2))
        #print('xi(0)= ' + str(sol3[0][0]) + ', xi\'(0)= ' + str(sol3[0][1]))
        print('xi(R)= ' + str(xiR3) + ', xi\'(R)= ' + str(R*dxiR3))

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

def Lplot_dispersion_function(alpha, m, k, ls, normalized = True):
    D = []
    for l in ls:
        D.append(Ldisp_F(l, m, k, alpha))
    print(D)
    print(min(D))
    D = D/np.mean([abs(x) for x in D]) if normalized else D
    axes = ['$\lambda$', '$D(\lambda; k)$']
    title = r'Disp. fun. for $\alpha = %1.1f$, $k=%1.2f$'%(alpha, k)
    lineplot(ls, D, axes, title, markzero = True)

def comparison_lineplot(x, ys, prelabel, labels, axes, title, posx, posy, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + r' %1.5f'%(labels[i]))
    ax.legend(fancybox=True, shadow=True, facecolor='white')
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    # The settings below are hardcoded
    ax.set_ylim([-1, 1])
    ax.set_xscale('log')
    #plt.text(posx, posy, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')

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
        datay.append(D)#/abs(D[-1]))
    prelabel = r'$k$ = '
    axes = ['$\lambda$', 'normalized $D(\lambda; k)$']
    title = r'Dispersion function for $(n,m) = (%1.0f, %1.0f)$ and $(C_{nm},\tau) = (%1.1f, %1.1f)$'%(n, mm, Cnm, tau)
    comparison_lineplot(datax, datay, prelabel, ks, axes, title, posx, posy, markzero = True)

def Lplot_dispersion_functions(alpha, m, ks, ls, posx, posy):
    datax = []
    datay = []
    for k in ks:
        D = []
        for l in ls:
            D.append(Ldisp_F(l, m, k, alpha))
        datax.append(ls)
        #avg = np.mean([abs(x) for x in D]) # The dispersion function is being normalized for the comparison.
        #datay.append(D/avg)
        datay.append(D/abs(D[-1]))
    prelabel = r'$k$ = '
    axes = ['$\lambda$', 'normalized $D(\lambda; k)$']
    title = r'Dispersion function for $\alpha = %1.1f$'%(alpha)
    comparison_lineplot(datax, datay, prelabel, ks, axes, title, posx, posy, markzero = True)