import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

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

def line_colorplot(xi, yi, Z, label, ticks, cmap, x, ys, prelabel, labels, axes): 
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (12, 4)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    # Contourf:
    xi = np.array(xi); yi = np.array(yi); Z = np.array(Z)
    X, Y = np.meshgrid(xi, yi)
    im = plt.contourf(X, Y, Z, 100, cmap=plt.get_cmap(cmap))
    cb = plt.colorbar(im , ax = [ax], location = 'top', label=label)
    cb.set_ticks(ticks) 
    
    # Lines:
    #colors = [u'#2ca02c', u'#ff7f0e', u'#1f77b4'] # inverted color cycle, to match the green line in the previous
    pe1 = [mpe.Stroke(linewidth=3, foreground='black'), mpe.Stroke(foreground='black',alpha=1),mpe.Normal()]
    for i in range(len(ys)):
        ax.plot(x[i],ys[i],label=prelabel + labels[i], linewidth = 2.3, path_effects=pe1, color='white')
    ax.legend(fancybox=True, shadow=True)
    plt.legend(frameon = 1).get_frame().set_edgecolor('black')
    #ax.set_ylim([0.25, 0.58])
    plt.show()

def Lorentz(a, q, r):
    mu0 = 1
    return (2*q**2*r)/(mu0*(q**2*r**2 + 1)**3) - (q*r**a*((q*r**a)/(q**2*r**2 + 1) - (2*q**3*r**a*r**2)/(q**2*r**2 + 1)**2 + (a*q*r*r**(a - 1))/(q**2*r**2 + 1)))/(mu0*r*(q**2*r**2 + 1))

def absLorentz(a,q,r):
    return np.abs(Lorentz(a,q,r))

def Twist(a, q, r):
    return q*r**(a-1)

def avg(f, param1, param2):
    rr = np.linspace(0.01,1,100)
    return np.mean(2*np.multiply(rr, f(param1, param2, rr)))