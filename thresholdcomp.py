import numpy as np
import matplotlib.pyplot as plt
import uuid

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

from thresholdfinder import obtain_taus, obtain_taus1, obtain_taus2, obtain_taus3, obtain_alphas, obtain_alphas2
from modelcomparison import Teresa_to_GH, Teresa_to_GH_bis, GH_to_Teresa, model_comparison, model_comparison_bis, compare_change_of_model, plot_lim_Teresa


### Trasnforming CC's threshold parameters to GH
#"""
N = 100
m = 1
[n, mm] = [1, 0]
tau = obtain_taus()
Cnm = np.loadtxt('sheet_cnms_nm10.csv', delimiter=',')
print(len(Cnm))

qs = np.linspace(0, 1.5, 200)
alphas = np.linspace(1e-4, 2, 400)
Bs = np.linspace(0.9, 1.1, 30)
q = []
alpha = []
for i in range(len(Cnm)):
    if i%10 == 0:
        print(i)
    #[q_approx, alpha_approx] = Teresa_to_GH(n, mm, Cnm[i], tau[i], qs, alphas, N)
    [q_approx, alpha_approx] = Teresa_to_GH_bis(n, mm, Cnm[i], tau[i], qs, alphas, Bs, N)[0:2]
    q.append(q_approx)
    alpha.append(alpha_approx)
axes = [r'$\alpha$',r'$q$']
title = "Teresa's thresholds to GH for $\lambda_0 = 1e-7$"
plot_lim_Teresa(alpha, q, axes, title, markzero = False)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', alpha, delimiter=',')
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', q, delimiter=',')
#"""


### Transforming GH's threshold parameters to CC
#"""
N = 100
m = 1
[n, mm] = [1, 0]
datax = obtain_alphas2()
datay = np.loadtxt('sheet_m1l1e-7.csv', delimiter=',')
alpha = datax[:-8]
q = datay[:-8]

Cnms = np.linspace(0.2,4,400)
taus = np.linspace(1,15,600)
Cnm = []
tau = []
print(len(q))
for i in range(len(q)):
    if i%10 == 0:
        print(i)
    [Cnm_approx, tau_approx] = GH_to_Teresa(n, mm, Cnms, taus, q[i], alpha[i], N)
    Cnm.append(Cnm_approx)
    tau.append(tau_approx)
axes = [r'$\tau$',r'$C_{nm}$']
title = "GH's thresholds to CC for $\lambda_0 = 1e-7$"
plot_lim_Teresa(tau, Cnm, axes, title, markzero = False)
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', Cnm, delimiter=',')
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', tau, delimiter=',')
#"""

### THRESHOLD COMPARISON FOR GH PARAMETERS
#"""
small_alpha = False
alpha1 = obtain_alphas2()[:-8] if not small_alpha else obtain_alphas()
#alpha2 = np.loadtxt('sheet_TtoGH_alpha.csv', delimiter=',')
alpha3 = np.loadtxt('sheet_TtoGH_alpha_nm10_2.csv', delimiter=',')
alpha4 = np.loadtxt('sheet_TtoGH_alpha_nm11_2.csv', delimiter=',')
q1 = np.loadtxt('sheet_m1l1e-7.csv', delimiter=',')[:-8] if not small_alpha else np.loadtxt('sheet_m1.csv', delimiter=',')[2]
#q2 = np.loadtxt('sheet_TtoGH_q.csv', delimiter=',')
q3 = np.loadtxt('sheet_TtoGH_q_nm10_2.csv', delimiter=',')
q4 = np.loadtxt('sheet_TtoGH_q_nm11_2.csv', delimiter=',')

title = 'Threshold comparison for GH parameters'
axes = [r'$\alpha$', r'$q$']
markzero = False
fig, ax = set_basics(title, axes, markzero)
ax.plot(alpha1, q1, label = 'Method applied to the GH')
#ax.plot(alpha2, q2, label = 'Conversion of the CC threshold')
ax.plot(alpha3, q3, label = 'Conversion of the CC threshold for (n,m) = (1,0)')
ax.plot(alpha4, q4, label = 'Conversion of the CC threshold for (n,m) = (1,1)')
ax.legend(fancybox=True, shadow=True, facecolor='white')
#plt.legend(frameon = 1).get_frame().set_edgecolor('black')


### THRESHOLD COMPARISON FOR CC PARAMETERS
tau1 = obtain_taus()
tau2 = np.loadtxt('sheet_GHtoT_tau.csv', delimiter=',')
Cnm1 = np.loadtxt('sheet_cnms_nm10.csv', delimiter=',')
Cnm2 = np.loadtxt('sheet_GHtoT_Cnm.csv', delimiter=',')

title = 'Threshold comparison for CC parameters'
axes = [r'$\tau$', r'$C_{nm}$']
markzero = False
fig, ax = set_basics(title, axes, markzero)
ax.plot(tau1, Cnm1, label = 'Method applied to the CC')
ax.plot(tau2, Cnm2, label = 'Conversion of the GH threshold')
ax.legend(fancybox=True, shadow=True, facecolor='white')
plt.legend(frameon = 1).get_frame().set_edgecolor('black')
#"""

plt.show()