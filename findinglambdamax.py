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

from maxgrowth import lambdamax, Tlambdamax, get_k_vals

# Finds lambda max for different parameters in GH
#alpha = 1; l_vals = [-9,-3]
alpha = 1.3; l_vals = [-14,-3]
#alpha = 1.5; l_vals = [-12,-3]

ms = [1,2,3,4]
qs = np.linspace(0.2, 0.8, 20)

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


# Finds lambda max for different parameters in CC
l_vals = [-14,-3]
parameters = [[1.1, -7], [1.2, -4], [1.3, -3], [1.4, -2], [1.5, -2], [2, -1], [3, -1], [4, -1], [5, -1]]#[1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5]
taus = np.linspace(1.1,1.5,33)
k = -7
#k = -7 #starting with tau = 1.1, Cnm = 1.5/tau
#k = -4 #starting with tau = 1.2, Cnm = 1.5/tau
m = 1
[n, mm] = [1, 0]
n_k = 30
n_l = 30
lambdamaxs = []
ks = []
#for p in parameters:
for tau in taus:
    #tau = p[0]
    #k = p[1]
    Cnms = np.linspace(1.5/tau, 1.5/tau + 0.2, 3)
    this_curve = []
    this_ks = []
    print(r'Obtaining curve for $\tau$ =', tau)
    for Cnm in Cnms:
        k_vals = [k - 0.5,k + 0.5]
        [l, k] = Tlambdamax(n, mm, Cnm, tau, m, k_vals, n_k, l_vals, n_l)
        this_curve.append(l)
        this_ks.append(k)
        print(r'For Cnm = ' + str(np.round(Cnm,3)) + ', lambda = ' + str(l) + ', k = ' + str(k))
    print('')
    lambdamaxs.append(this_curve)
    ks.append(this_ks)
  
# Save as csv
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', lambdamaxs, delimiter=',')
np.savetxt('sheet_'+str(uuid.uuid4())+'.csv', ks, delimiter=',')

plt.show()