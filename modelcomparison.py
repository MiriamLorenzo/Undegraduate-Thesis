import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

# The magnetic fields are in a separate file.
from mag_field_GH import Bz, Btheta
from mag_field_Teresa import TBz, TBtheta

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

def Teresa_to_Teresa(n, m, tau1, Cnm1, tau2, Cnms, Bs, N):
    p = 1
    dist = float('inf')
    Cnmmin = 0
    Bmin = 0
    r = np.linspace(0, 1, N)
    B_z = TBz(r, n, m, Cnm1, tau1)
    B_theta = TBtheta(r, n, m, Cnm1, tau1)
    for Cnm2 in Cnms:
        for B in Bs:
            newB_z = B*TBz(r, n, m, Cnm2, tau2)
            newB_theta = B*TBtheta(r, n, m, Cnm2, tau2)
            new_dist = np.sqrt(linalg.norm(newB_z - B_z, p)**2 + linalg.norm(newB_theta - B_theta,p)**2)
            if (new_dist < dist):
                dist = new_dist
                Cnmmin = Cnm2
                Bmin = B
    return [Cnmmin, Bmin]    

def Teresa_to_GH_bis(n, m, Cnm, tau, qs, alphas, Bs, N):
    p = 1
    dist = float('inf')
    qmin = 0;
    alphamin = 0;
    Bmin = 0;
    r = np.linspace(0, 1, N)
    B_z = TBz(r, n, m, Cnm, tau)
    B_theta = TBtheta(r, n, m, Cnm, tau)
    for q in qs:
        for alpha in alphas:
            for B in Bs:
                newB_z = B*Bz(r, q, alpha)
                newB_theta = B*Btheta(r, q, alpha)
                new_dist = np.sqrt(linalg.norm(newB_z - B_z, p)**2 + linalg.norm(newB_theta - B_theta,p)**2)
                if (new_dist < dist):
                    dist = new_dist
                    qmin = q
                    alphamin = alpha
                    Bmin = B
    return [qmin, alphamin, Bmin]

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
    #plt.show()

def model_comparison_bis(r,n,m,Cnm,tau,q,alpha,B):
    plt.rcParams["figure.figsize"] = (7.5, 6)
    plt.figure()
    plt.title(r"CC ($Cnm = {}$, $\tau = {}$) adjusted by GH ($q = {}$, $\alpha = {}$, $B_0 = {}$)".format(np.round(Cnm,2), np.round(tau,2), np.round(q,2), np.round(alpha,2), np.round(B,4)))
    plt.xlabel('$r$')
    plt.ylabel(r"Magnetic field component")
    plt.plot(r, TBtheta(r, n, m, Cnm, tau), 'g')
    plt.plot(r, TBz(r, n, m, Cnm, tau), 'b')
    plt.plot(r, B*Btheta(r, q, alpha), 'g--')
    plt.plot(r, B*Bz(r, q, alpha), 'b--')
    plt.legend([r"$B_\theta$", "$B_z$"])
    #plt.savefig(str(uuid.uuid4()))
    #plt.show()

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
        print(r'Computing for Cnm = %1.2f' %(Cnm))
        datay1aux = []
        datay2aux = []
        for tau in taus:
            Bs = np.linspace(0.9, 1.1, 10)
            [q_approx, alpha_approx] = Teresa_to_GH_bis(n,m,Cnm,tau,qs,alphas,Bs,N)[:2]
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

def plot_lim_Teresa(xs, ys, axes, title, markzero = False):
    fig, ax = set_basics(title, axes, markzero)
    ax.plot(xs,ys)
