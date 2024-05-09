import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

# Constants
B0 = 8  # nT
n0 = 5   # cm^-3
kappa = 0.23
vb = 53000  # m/s
d0 = 0.009  # AU
a = 3  # m/s^2
sigma = 0.1
t0 = 0  # assuming t0 is 0 for the integration starting point
AU_to_m = 149597870700

plt.rcParams.update({'font.size': 12})

# Functions to integrate
def integrand_f1(tau, t):
    return (d0 + vb * (tau - t0) / AU_to_m + 0.5 * a / AU_to_m * (tau - t0)**2)**(-1.5)

def integrand_f2(tau, t):
    return (d0 + vb * (tau - t0) / AU_to_m)**(-1.5)

def integrand_f3(tau, t):
    return d0**(-1.5)

# Functions for f1, f2, and f3
def f1(t):
    integral, _ = quad(integrand_f1, t0, t, args=(t,))
    return np.exp(1.35e-7 * (B0 * sigma) / (kappa * np.sqrt(n0)) * integral)

def f2(t):
    integral, _ = quad(integrand_f2, t0, t, args=(t,))
    return np.exp(1.35e-7 * (B0 * sigma) / (kappa * np.sqrt(n0)) * integral)

def f3(t):
    integral, _ = quad(integrand_f3, t0, t, args=(t,))
    return np.exp(1.35e-7 * (B0 * sigma) / (kappa * np.sqrt(n0)) * integral)

N = 16

# Time and distance arrays
time_hours = np.linspace(0, N*50, N*1000)  # in hours
time_seconds = time_hours * 3600  # convert time to seconds for the function

# Calculate values for each function
f1_values = np.array([f1(t) for t in time_seconds])
f2_values = np.array([f2(t) for t in time_seconds])
f3_values = np.array([f3(t) for t in time_seconds])

def create_adjusted_plots1(f1_values, f2_values, f3_values, time):
    # Create figure and axes with specific size to match the provided image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Plot on the first axes
    ax1.plot(time[:1000], f1_values[:1000], 'g-', label=r'Expanding Flux Rope ($a > 0$)')
    ax1.plot(time[:1000], f2_values[:1000], 'b-', label=r'Expanding Flux Rope ($a = 0$)')
    ax1.plot(time[:1000], f3_values[:1000], 'r-', label='Static Flux Rope')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel(r'$\left\|\frac{{\bf\xi (r}, t)}{{\bf \xi (r}, 0)}\right\|$ bound')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylim(0.5, 1e20)

    # Plot on the second axes
    distance = d0 + vb * (time*3600) / AU_to_m + 0.5 * a / AU_to_m * (time*3600)**2
    ax2.plot(distance, f1_values, 'g-', label=r'Expanding Flux Rope ($a > 0$)')
    distance = d0 + vb * (time*3600) / AU_to_m
    ax2.plot(distance, f2_values, 'b-', label=r'Expanding Flux Rope ($a = 0$)')
    ax2.set_xlabel('Distance from the Sun (AU)')
    ax2.legend(loc='lower right')
    ax2.set_yscale('log')
    ax2.set_ylim(0.8, 1.1e5)
    ax2.set_xlim(0, 1.01)

    # Adjust layout to prevent overlapping and set the supertitle
    plt.tight_layout()
    plt.suptitle('Maximum Growth of the Perturbation', fontsize=14)
    plt.subplots_adjust(top=0.87)

    # Remove gridlines from both plots
    ax1.grid(True)
    ax2.grid(True)

def create_adjusted_plots2(f1_values, f2_values, f3_values, time):
    # Create figure and axes with specific size to match the provided image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Plot on the first axes
    distance = d0 + vb * (time*3600) / AU_to_m + 0.5 * a / AU_to_m * (time*3600)**2
    ax1.plot(time[:1000], f1_values[:1000]/distance[:1000]*d0/1e3, 'g-', label=r'Expanding Flux Rope ($a > 0$)')
    distance = d0 + vb * (time*3600) / AU_to_m
    ax1.plot(time[:1000], f2_values[:1000]/distance[:1000]*d0/1e3, 'b-', label=r'Expanding Flux Rope ($a = 0$)')
    ax1.plot(time[:1000], f3_values[:1000]/1e3, 'r-', label='Static Flux Rope')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel(r'$\frac{\left\|{\bf\xi (r}, t)\right\|}{R(t)}$ bound')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 1e17)

    # Plot on the second axes
    distance = d0 + vb * (time*3600) / AU_to_m + 0.5 * a / AU_to_m * (time*3600)**2
    ax2.plot(distance, f1_values/distance*d0/1e3, 'g-', label=r'Expanding Flux Rope ($a > 0$)')
    distance = d0 + vb * (time*3600) / AU_to_m
    ax2.plot(distance, f2_values/distance*d0/1e3, 'b-', label=r'Expanding Flux Rope ($a = 0$)')
    ax2.set_xlabel('Distance from the Sun (AU)')
    ax2.legend(loc='right')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4,10)
    ax2.set_xlim(0, 1.01)

    # Adjust layout to prevent overlapping and set the supertitle
    plt.tight_layout()
    plt.suptitle('Maximum Relative Growth of the Perturbation', fontsize=14)
    plt.subplots_adjust(top=0.87)

    # Remove gridlines from both plots
    ax1.grid(True)
    ax2.grid(True)


# Now we will create the adjusted plots using the actual data
create_adjusted_plots1(f1_values, f2_values, f3_values, time_hours)
create_adjusted_plots2(f1_values, f2_values, f3_values, time_hours)

plt.show()