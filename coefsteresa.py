from modelcomparison import Teresa_to_Teresa
import numpy as np
import matplotlib.pyplot as plt
N = 100
[n,m] = [1,0]
tau1 = 1
Cnms = np.linspace(0.01,10,20)
Bs = np.linspace(0.9,1.2,20)
tau2 = 1.1
Cnmsbis = []
Bsbis = []
for Cnm in Cnms:
    [Cnmbis, Bbis] = Teresa_to_Teresa(n, m, tau1, Cnm, tau2, np.linspace(0.01,10,200), Bs, N)
    Cnmsbis.append(Cnmbis)
    Bsbis.append(Bbis)
#plt.plot(Cnms,Cnmsbis)
#plt.ylim([-1e-2,1e-2])
plt.plot(Cnms,Bsbis)
#plt.plot(Cnms,Cnmsbis)
print((Cnmsbis[-1]-Cnmsbis[0])/(10-0.01))
plt.show()
