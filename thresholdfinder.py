import numpy as np
import matplotlib.pyplot as plt

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

#plt.rcParams['text.usetex'] = True
#plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 17})
plt.rcParams["figure.figsize"] = (5, 5)
def set_basics(title, axes, markzero = False):
    fig, ax = plt.subplots()
    plt.title(title, fontsize = 14)
    if markzero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel(axes[0], fontsize = 14)
    ax.set_ylabel(axes[1], fontsize = 14)
    return fig, ax

m = 1

from maxgrowth import lambdamax, Tlambdamax, get_k_vals

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
    return q, lambdamax(alpha, qmax, m, k_vals, n_k, l_vals, n_l)[1]

def Tcritical_Cnm(n, mm, m, tau, l0, Cnmmin, Cnmmax, k, tol = 0.001, n_k = 40, n_l = 20):
    l_vals = [np.log10(l0)-3, np.log10(l0)+3]
    while (Cnmmax - Cnmmin > tol):
        Cnmmid = (Cnmmin + Cnmmax)/2
        k_vals = (k - 4, k + 4)
        [lmid, kbis] = Tlambdamax(n, mm, Cnmmid, tau, m, k_vals, n_k, l_vals, n_l)
        #print(r'For Cnm = ' + str(np.round(Cnmmid,3)) + ', lambda = ' + str(lmid) + ', k = ' + str(kbis))
        if kbis < 0: k = kbis
        if (lmid < l0):
            Cnmmax = Cnmmid
        else:
            Cnmmin = Cnmmid
    Cnm = (Cnmmin + Cnmmax)/2
    return Cnm, Tlambdamax(n, mm, Cnmmin, tau, m, k_vals, n_k, l_vals, n_l)[1]

def obtain_taus1(N = 100, tau_min = 1.1, tau_1 = 1.5, tau_2 = 2.5, tau_max = 5):
    t1 = np.linspace(tau_min, tau_1, int(N/3))
    t2 = np.linspace(tau_1 + (tau_2 - tau_1)*3/N, tau_2, int(N/3))
    t3 = np.linspace(tau_2 + (tau_max - tau_2)*3/N, tau_max, int(N/3))
    return np.concatenate((t1,t2,t3))

def obtain_taus2(N = 50, tau_min = 1.001, tau_max = 1.1):
    t1 = np.linspace(tau_max, tau_max - (tau_max-tau_min)/2, int(N/3))
    t2 = np.linspace(tau_max - (tau_max-tau_min)/2, tau_max - (tau_max-tau_min)*4/5, int(N/3))[1:]
    return np.concatenate((t1,t2))

def obtain_taus3(N = 50, tau_min = 1.001, tau_max = 1.1):
    t3 = np.linspace(tau_max - (tau_max-tau_min)*4/5, tau_min, int(N/3))[1:]
    return t3

def obtain_taus():
    return np.concatenate((obtain_taus3()[::-1], obtain_taus2()[::-1], obtain_taus1()[1:]))

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
    return np.concatenate((a1,a2,a3,a4))

def comparison_lineplot(x, ys, prelabel, labels, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + labels[i]) #"{:.0e}".format(labels[i]))
    ax.legend(fancybox=True, shadow=True)
    #plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    #plt.text(1.438, 0.267, f'$m = {m}$', bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1., 1, 1)), color = 'black')
    #x.set_ylim([0.25, 1.8])#([0.25, 0.58])
