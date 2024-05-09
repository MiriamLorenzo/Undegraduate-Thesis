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

from dispersionfunction import model_dispF_comparison
from maxgrowth import lambdamax, Tlambdamax
from thresholdfinder import obtain_taus, obtain_taus1, obtain_taus2, obtain_taus3, obtain_alphas, obtain_alphas2
from modelcomparison import Teresa_to_GH, Teresa_to_GH_bis, GH_to_Teresa, model_comparison, model_comparison_bis, compare_change_of_model, plot_lim_Teresa

### Considering a better ajustment with a change in B0
#"""
N = 100
[n,mm] = [1,0]
m = 1
qs = np.linspace(0, 6, 200)
alphas = np.linspace(1e-4, 10, 400)
Bs = np.linspace(0.9, 1.1, 9)

for i in range(15):

    tau = obtain_taus()[10*i]
    Cnm = np.loadtxt('sheet_cnms_nm10.csv', delimiter=',')[10*i]

    [q_approx, alpha_approx] = Teresa_to_GH(n, m, Cnm, tau, qs, alphas, N)
    [q_approx_bis, alpha_approx_bis, B_approx] = Teresa_to_GH_bis(n, m, Cnm, tau, qs, alphas, Bs, N)

    print('For (n,m) = (' + str(n) + ', ' + str(mm) + ') and (tau, Cnm): (' + str(np.round(tau,4)) + ', ' + str(np.round(Cnm,4)) + ')')
    print('Best (q, alpha): (' + str(np.round(q_approx,4)) + ', ' + str(np.round(alpha_approx,4)) + ')')
    print('Best (q, alpha) with a change in B: ' + str(np.round(q_approx_bis,4)) + ', ' + str(np.round(alpha_approx_bis,4)) + ')')
    print('Best B: ' + str(np.round(B_approx,4)))
    print('')

r = np.linspace(0, 1, N)
model_comparison(r,n,m,Cnm,tau,q_approx,alpha_approx)
model_comparison_bis(r,n,m,Cnm,tau,q_approx,alpha_approx,B_approx)
#"""

### Compares the CC and GH max growth for an specific MF configuration and plots the perturbation
#"""
m = 1
[n, mm] = [1, 0]

i = 100

tau = obtain_taus()[-i]
Cnm = np.loadtxt('sheet_cnms_nm10.csv', delimiter=',')[-i]

alpha = np.loadtxt('sheet_TtoGH_alpha.csv', delimiter=',')[-i]
q = np.loadtxt('sheet_TtoGH_q.csv', delimiter=',')[-i]

n_k = 100
n_l = 50
l_vals = [-9, -5]
k_vals1 = [-8, -5]
k_vals2 = [-1.5, -0.5]

[l1, k1] = Tlambdamax(n, mm, Cnm, tau, m, k_vals1, n_k, l_vals, n_l)
[l2, k2] = lambdamax(alpha, q, m, k_vals2, n_k, l_vals, n_l)

model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, norm = False, info = True)
#"""