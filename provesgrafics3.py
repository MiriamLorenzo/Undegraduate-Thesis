import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import math
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['text.usetex'] = True

N = 100
obs_B0 = np.linspace(0,5,N)
obs_np = np.linspace(-1,5,N)
Au = 1.496*1e11
obs_R = [0.001*Au, 0.01*Au, 0.1*Au, 1*Au]

mp = 1.673e-27 # proton mass in kg
ratio = 1.16 # ratio between proton density and mass density (see equation 1.13 in thesis)
mu0 = 4*math.pi*1e-7 # magnetic permitivity
sec = 3600 # number of seconds in an hour

def T_hours(l, obs_B0, obs_R, obs_np):
    sigma = np.sqrt(l)
    T = 1/sigma
    obs_rho = ratio*mp*obs_np*1e6
    factor = (obs_R/obs_B0)*np.sqrt(mu0*obs_rho)
    T_d = T*factor/sec
    return T_d

l = 10

plt.rcParams.update({'font.size': 12})

colors = ['#54278f', '#54278f', '#54278f', '#d73027', '#d73027', '#d73027', 
          '#fee08b', '#fee08b', '#fee08b', 
          '#4575b4', '#4575b4', '#4575b4', '#313695', '#313695', '#313695']
colors = ['#d73027'] * 9 + ['#fee08b'] * 3 + ['#4575b4'] * 8

cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=256)

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 11/19, 12/19, 13/19, 1]  # Ensure the 12th and 13th colors are the desired ones

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 16/63, 34/63, 35/63, 1]  # Ensure the 12th and 13th colors are the desired ones

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 24/63, 25/63, 34/63, 35/63, 1]  # Ensure the 12th and 13th colors are the desired ones

# Create the gradient colormap
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Function to format the tick labels as 10^x
def log_formatter(x, pos):
    return r'$10^{%d}$' % int(x)

# Set up the figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Adjust the space between the plots and make room for the colorbar
plt.subplots_adjust(right=0.85)

# Titles for the subplots
titles = [r'$R=10^{-3}\, AU$', r'$R=10^{-2}\, AU$', r'$R=0.1\, AU$', r'$R=1\, AU$']

for i, ax in enumerate(axes.flat):

    M = np.zeros(shape=(N, N))
    for j in range(N):
        for k in range(N):
            #M[i][k] = np.log10(T_hours(1, obs_B0[i]*1e-9, R_value, obs_np[k])*sec)
            M[j][k] = np.log10(T_hours(l, 10**(obs_B0[j]-9), obs_R[i], 10**obs_np[k]))

    X, Z = np.meshgrid(obs_B0, obs_np)


    im = ax.imshow(M, cmap=cmap_custom, extent=[X.min(), X.max(), Z.min(), Z.max()],
                       origin='lower', aspect='auto', vmin=-7, vmax=9)

    # Set tick formatters to display 10^x for the axis labels
    ax.set_xticks(np.linspace(obs_B0.min(), obs_B0.max(), 6))  # Set positions of ticks
    ax.set_yticks(np.linspace(obs_np.min(), obs_np.max(), 7))  # Set positions of ticks
    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    # Set title for each subplot
    ax.set_title(titles[i])

    # Only show x-axis labels on the bottom plots
    if i >= 2:  # Bottom row
        ax.set_xlabel(r'$B_0\, (nT)$', fontsize=16)
    else:
        ax.set_xticklabels([])  # Remove x-tick labels on the top row

    # Only show y-axis labels on the left plots
    if i % 2 == 0:  # Left column
        ax.set_ylabel(r'$n_p\, (cm^{-3})$', fontsize=16)
    else:
        ax.set_yticklabels([])  # Remove y-tick labels on the right column

cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Position for the colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\log_{10}(T_{hours})$', fontsize=16)

# Add a title to the figure
fig.suptitle(r'Characteristic Timescales of an instability for $\lambda = 10$', fontsize=18)

l = 1e-5

plt.rcParams.update({'font.size': 12})

colors = ['#54278f', '#54278f', '#54278f', '#d73027', '#d73027', '#d73027', 
          '#fee08b', '#fee08b', '#fee08b', 
          '#4575b4', '#4575b4', '#4575b4', '#313695', '#313695', '#313695']
colors = ['#d73027'] * 9 + ['#fee08b'] * 3 + ['#4575b4'] * 8

cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=256)

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 11/19, 12/19, 13/19, 1]  # Ensure the 12th and 13th colors are the desired ones

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 16/63, 34/63, 35/63, 1]  # Ensure the 12th and 13th colors are the desired ones

# Define the key colors for the gradient
colors = ['#54278f', '#d73027', '#fee08b', '#fee08b', '#4575b4', '#313695']
# Define the color positions along the gradient (we want to control the position of the key colors)
positions = [0, 24/63, 25/63, 34/63, 35/63, 1]  # Ensure the 12th and 13th colors are the desired ones

# Create the gradient colormap
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Function to format the tick labels as 10^x
def log_formatter(x, pos):
    return r'$10^{%d}$' % int(x)

# Set up the figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Adjust the space between the plots and make room for the colorbar
plt.subplots_adjust(right=0.85)

# Titles for the subplots
titles = [r'$R=10^{-3}\, AU$', r'$R=10^{-2}\, AU$', r'$R=0.1\, AU$', r'$R=1\, AU$']

for i, ax in enumerate(axes.flat):

    M = np.zeros(shape=(N, N))
    for j in range(N):
        for k in range(N):
            #M[i][k] = np.log10(T_hours(1, obs_B0[i]*1e-9, R_value, obs_np[k])*sec)
            M[j][k] = np.log10(T_hours(l, 10**(obs_B0[j]-9), obs_R[i], 10**obs_np[k]))

    X, Z = np.meshgrid(obs_B0, obs_np)


    im = ax.imshow(M, cmap=cmap_custom, extent=[X.min(), X.max(), Z.min(), Z.max()],
                       origin='lower', aspect='auto', vmin=-7, vmax=9)

    # Set tick formatters to display 10^x for the axis labels
    ax.set_xticks(np.linspace(obs_B0.min(), obs_B0.max(), 6))  # Set positions of ticks
    ax.set_yticks(np.linspace(obs_np.min(), obs_np.max(), 7))  # Set positions of ticks
    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    # Set title for each subplot
    ax.set_title(titles[i])

    # Only show x-axis labels on the bottom plots
    if i >= 2:  # Bottom row
        ax.set_xlabel(r'$B_0\, (nT)$', fontsize=16)
    else:
        ax.set_xticklabels([])  # Remove x-tick labels on the top row

    # Only show y-axis labels on the left plots
    if i % 2 == 0:  # Left column
        ax.set_ylabel(r'$n_p\, (cm^{-3})$', fontsize=16)
    else:
        ax.set_yticklabels([])  # Remove y-tick labels on the right column

cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Position for the colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\log_{10}(T_{hours})$', fontsize=16)

# Add a title to the figure
fig.suptitle(r'Characteristic Timescales of an instability for $\lambda = 10^{-5}$', fontsize=18)

plt.show()
