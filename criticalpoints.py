import numpy as np
import matplotlib.pyplot as plt
import uuid

# Some warnings show up during the resolution of ODEs, but aren't important to the analysis.
import warnings
warnings.filterwarnings('ignore')

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

from thresholdfinder import critical_q, Tcritical_Cnm, obtain_taus, obtain_taus1, obtain_taus2, obtain_taus3, obtain_alphas, obtain_alphas2, comparison_lineplot

# Critical Cnm
m = 1
[n, mm] = [1, 0]
tau = 4
k = -1
l0 = 1e-7
[Cnmmin, Cnmmax] = [1.5/tau, 1.5/tau + 0.2]
[Cnm, k] = Tcritical_Cnm(n, mm, m, tau, l0, Cnmmin, Cnmmax, k)
print('Instability: m =', m)
print('Threshold: l0 =', l0)
print('tau =', tau)
print('Cnm =', Cnm)
print('k =', k)


# Critical q
m = 1
alpha = 1
l0 = 1e-7
[qmin, qmax] = [0.2, 0.8]
[q, k] = critical_q(alpha, l0, qmin, qmax)
print('Instability: m =', m)
print('Threshold: l0 =', l0)
print('Force parameter: alpha =', alpha)
print('Critical twist parameter: q =', q)
print('k =', k)


# Limit curve for CC
m = 1
[n, mm] = [1, 0]
taus = obtain_taus()
ls = [1e-7]#[4e-8, 7e-8, 1e-7]
k = -7

crit_Cnms = []
crit_k = []
for i in range(len(ls)):
    this_curve = []
    ks = []
    print('Obtaining curve for lambda =', ls[i])
    for tau in taus:
        [Cnm, k] = Tcritical_Cnm(n, mm, m, tau, ls[i], 1.5/tau, 1.5/tau + 0.2, k)
        #[Cnm, k] = Tcritical_Cnm(n, mm, m, tau, ls[i], Cnm, Cnm + 0.2, k)
        this_curve.append(Cnm)
        ks.append(k)
        print(r'For tau = ' + str(np.round(tau,3)) + ', Cnm = ' + str(np.round(Cnm,3)) + ', k = ' + str(np.round(k,3)))
    print('')
    crit_Cnms.append(this_curve)
    crit_k.append(ks)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', crit_Cnms, delimiter=',')
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', crit_k, delimiter=',')


# Limit curve for GH
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


# Plot of the CC limit curves
m = 1
datax = [obtain_taus(), obtain_taus()]
datay = [np.loadtxt('sheet_cnms_nm10.csv', delimiter=','), np.loadtxt('sheet_cnms_nm11.csv', delimiter=',')]
prelabel = r'$(n,m)$ = '
labels = [r'(1,0)',r'(1,1)']
axes = [r'$\tau$', r'$C_{nm}$']
title = 'Stability curve for the threshold $\lambda_0 = 1e-7$'
comparison_lineplot(datax, datay, prelabel, labels, axes, title)


# Plot of the GH limit curves
m = 1
datax = obtain_alphas2() 
datay = np.loadtxt('sheet_m1l1e-7.csv', delimiter=',')
datax = datax[:-8]
datay = datay[:-8]
datax = [datax, datax]
datay = [datay, datay]
prelabel = r'$\lambda_0$ = '
labels = [7e-8, 7e-8]
axes = [r'$\alpha$', r'$q$']
title = 'Stability curve for the threshold $\lambda_0 = 1e-7$'
comparison_lineplot(datax, datay, prelabel, labels, axes, title)

plt.show()