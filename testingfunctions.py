import numpy as np

from mag_field_GH import Bz, Btheta, dBz, dBtheta
from mag_field_Teresa import TBz, TBtheta, TdBz, TdBtheta
from auxiliaryfunctions import F, TF, f, Tf, df, Tdf, g, Tg, F_euler, TF_euler, bound_cond_D, Tbound_cond_D
from dispersionfunction import disp_F, Tdisp_F, model_dispF_comparison, plot_dispersion_function, Tplot_dispersion_function

### COMPROVAR FUNCIONS GH I CC TOT OKAY
r = 0.5
m = 1
[n, mm] = [1, 0]
k = -3
l = 1e-8
l1 = l
l2 = l
k1 = k
k2 = k
y = [1, 2]
xiR = 1
dxiR = 2
ls = 10**np.linspace(-10,-6,100)
[alpha, q] = [1.2, 0.5]
[tau, Cnm] = [1.3, 1.2]
print(Bz(r, q, alpha) - TBz(r, n, mm, Cnm, tau))
print(dBz(r, q, alpha) - TdBz(r, n, mm, Cnm, tau))
print(Btheta(r, q, alpha) - TBtheta(r, n, mm, Cnm, tau))
print(dBtheta(r, q, alpha) - TdBtheta(r, n, mm, Cnm, tau))
print(F(r, m, k, q, alpha) - TF(r, n, mm, m, k, Cnm, tau))
print(f(r, m, k, l, q, alpha) - Tf(r, n, mm, m, k, l, Cnm, tau))
print(df(r, m, k, l, q, alpha) - Tdf(r, n, mm, m, k, l, Cnm, tau))
print(g(r, m, k, l, q, alpha) - Tg(r, n, mm, m, k, l, Cnm, tau))
print(F_euler(r, y, m, k, l, q, alpha) - TF_euler(r, y, n, mm, m, k, l, Cnm, tau))
print(bound_cond_D(m, k, l, xiR, dxiR, q, alpha) - Tbound_cond_D(n, mm, m, k, l, xiR, dxiR, Cnm, tau))
print(disp_F(l, m, k, q, alpha, printsol = False) - Tdisp_F(l, n, mm, m, k, Cnm, tau, printsol = False))
model_dispF_comparison(l1, l2, n, mm, m, k1, k2, Cnm, tau, q, alpha, norm = True, info = False)
plot_dispersion_function(alpha, q, m, k, ls, normalized = True)
Tplot_dispersion_function(n, mm, Cnm, tau, m, k, ls, normalized = True)
