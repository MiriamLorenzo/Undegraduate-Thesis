# # Import necessary scientific and data handling libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm, lognorm, gamma
# from scipy.interpolate import interp1d
# from scipy.optimize import curve_fit
# from matplotlib.ticker import FuncFormatter

# # Set a constant threshold value for later use in filtering data
# Cnm_cut = 1.75

# # Configure Matplotlib to use LaTeX for text rendering and set default figure size and font size
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 12})
# plt.rcParams["figure.figsize"] = (7, 3)

# # Load data from a CSV file into a pandas DataFrame
# data = pd.read_csv('results_modelcc_teresaN.csv', delimiter=',')

# # Display specific data columns for the year 2010
# print(data.loc[data['RYEAR']==2010][['RC10','RRR']])

# # Convert all values in 'RC10' column to their absolute values to ensure non-negativity
# data['RC10'] = abs(data['RC10'])

# # Filter the data to include only rows where 'RCORREL' is greater than 0.5
# data_WF = data.loc[data['RCORREL']>0.5]

# # Extract 'RC10' values from the filtered data where values are less than 3
# rc10_data = data_WF.loc[data_WF['RC10']<3]['RC10']
# raw_rc10_data = data.loc[data['RC10']<3]['RC10']

# # Define custom bins for plotting histograms
# bins = np.linspace(0, 3, 31)

# # Create a histogram plot of 'RC10' data
# plt.figure()
# plt.hist(rc10_data, bins=bins, density=False, alpha=0.6, color='r')

# # Fit a normal distribution to the histogram data and plot the distribution
# mu, std = norm.fit(rc10_data)
# x = np.linspace(0, 3, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p*0.1*len(rc10_data), color='k', linestyle='--', lw=0.8, label=f'Normal Fit:\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

# # Add labels and title to the histogram
# plt.xlabel('$|C_{10}|$')
# plt.ylabel('Number of ICMEs')
# plt.title('Histogram of $C_{10}$', fontsize=16)
# plt.legend()

# # Define a custom formatting function for y-axis ticks
# def custom_formatter(x, pos):
#     return '{%1.0f}' % (x * 0.1 * len(rc10_data))
# plt.yticks(np.linspace(0, 15, 6) / 0.1 / len(rc10_data))
# formatter = FuncFormatter(custom_formatter)
# plt.gca().yaxis.set_major_formatter(formatter)

# plt.tight_layout()

# # Print the number of data points meeting the earlier criteria
# print(len(rc10_data))
# rc10_data2 = data_WF.loc[data_WF['RC10']<Cnm_cut]['RC10']
# print(len(rc10_data2))

# # Load additional datasets for interpolation
# datax = np.loadtxt('C10_tau10_v5.csv', delimiter=',')
# datay = np.loadtxt('lambda_tau10_v5.csv', delimiter=',')

# # Create an interpolation function from the loaded data
# interpolator = interp1d(datax, np.log10(datay), kind='linear', fill_value="extrapolate")

# # Interpolate the values for the filtered 'RC10' data
# interpolated_sigmas = interpolator(rc10_data2)/2
# # print(max(interpolated_sigmas))

# # Create and plot a histogram for the interpolated values
# plt.figure()
# bins = np.linspace(-11, 4, 31) / 2
# plt.hist(interpolated_sigmas, bins=bins, density=False, alpha=0.6, color='r')
# def custom_formatter2(x, pos):
#     return r'10$^{%1.0f}$' % x
# plt.xticks(2 * np.linspace(-5, 2, 8) / 2)
# formatter = FuncFormatter(custom_formatter2)
# plt.gca().xaxis.set_major_formatter(formatter)

# d = bins[1] - bins[0]
# def custom_formatter3(x, pos):
#     return '{%1.0f}' % (x * d * len(rc10_data2))
# formatter = FuncFormatter(custom_formatter3)
# # plt.gca().yaxis.set_major_formatter(formatter)

# # Fit and plot a normal distribution to the histogram of interpolated sigmas
# mu, std = norm.fit(interpolated_sigmas)
# xmin, xmax = -11, 4
# x = np.linspace(xmin, xmax, 100) / 2
# p = norm.pdf(x, mu, std)
# plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw=0.8, label=r'Lognormal Fit:\newline$\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))
# plt.xlabel(r'$\sigma$')
# plt.ylabel('Number of ICMEs')
# plt.title(r'Histogram of $\sigma$', fontsize=16)
# plt.legend()

# plt.tight_layout()

# plt.figure()
# bins = np.linspace(-11,4,31)
# plt.hist(interpolated_sigmas*2, bins=bins, density=False, alpha=0.6, color='r')

# def custom_formatter2(x, pos):
#     return r'10$^{%1.0f}$' %x
# plt.xticks(2*np.linspace(-5,2,8))
# formatter = FuncFormatter(custom_formatter2)
# plt.gca().xaxis.set_major_formatter(formatter)

# d = bins[1]-bins[0]
# def custom_formatter3(x, pos):
#     return '{%1.0f}' %(x * d * len(rc10_data2))
# #plt.yticks(np.linspace(0,12,5)/(d * len(rc10_data2)))
# formatter = FuncFormatter(custom_formatter3)
# #plt.gca().yaxis.set_major_formatter(formatter)

# # Fit a normal distribution
# mu, std = norm.fit(interpolated_sigmas*2)
# xmin, xmax = -11, 4
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw = 0.8, label=r'Lognormal Fit:\newline$\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

# plt.xlabel(r'$\lambda$')
# plt.ylabel('Number of ICMEs')
# plt.title(r'Histogram of $\lambda$', fontsize=16)
# plt.legend()

# plt.tight_layout()

# T_char = 1/data_WF.loc[data_WF['RC10']<Cnm_cut]['RBY0']/1e-9/np.sqrt(10**(interpolated_sigmas*2))*data_WF.loc[data_WF['RC10']<Cnm_cut]['RRR']*1.5e11*np.sqrt(4*3.14*1e-7*1.16*data_WF.loc[data_WF['RC10']<Cnm_cut]['np']*1e6*1.6*1e-27)
# bins = np.linspace(0,5,21)
# bins[:8] = np.linspace(0,np.log10(2.5*24),8)
# plt.figure()
# plt.hist(np.log10(T_char/3600), bins=bins, density=False, alpha=0.6, color='b')

# def custom_formatter4(x, pos):
#     return r'10$^{%1.0f}$' %x
# plt.xticks(np.linspace(0,6,7))
# formatter = FuncFormatter(custom_formatter4)
# plt.gca().xaxis.set_major_formatter(formatter)

# d = bins[1]-bins[0]
# def custom_formatter5(x, pos):
#     return '{%1.0f}' %(x * d * len(rc10_data2))
# plt.yticks(np.linspace(0,15,6))
# formatter = FuncFormatter(custom_formatter5)
# #plt.gca().yaxis.set_major_formatter(formatter)

# # Fit a normal distribution
# mu, std = norm.fit(np.log10(T_char/3600))
# xmin, xmax = 0, 6
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw = 0.8, label=r'Lognormal Fit:\newline$\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

# # print(np.sort(T_char/3600))

# plt.xlabel(r'$T_c$ (h)')
# plt.ylabel('Number of ICMEs')
# plt.title(r'Histogram of $T_c$', fontsize=16)
# plt.legend()
# plt.tight_layout()
# plt.axvline(np.log10(2.5*24),linestyle=':', color='k', lw=1)
# plt.show()

# Import necessary scientific and data handling libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gamma
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

# Set a constant threshold value for later use in filtering data
Cnm_cut = 1.75

# Configure Matplotlib to use LaTeX for text rendering and set default figure size and font size
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (7.3, 3.2)

# Load data from a CSV file into a pandas DataFrame
data = pd.read_csv('results_modelcc_teresaNDiP.csv', delimiter=',')

plt.figure()
plt.scatter(abs(data.loc[data['DiP']>0]['RCORREL']), data.loc[data['DiP']>0]['DiP'])
plt.xlabel('Correlation')
plt.ylabel('DiP')
plt.tight_layout()

plt.figure()
plt.scatter(abs(data.loc[data['DiP']>0]['RCHI']), data.loc[data['DiP']>0]['DiP'])
plt.xlabel('Chi Sq')
plt.ylabel('DiP')
plt.tight_layout()

plt.figure()
plt.scatter(abs(data.loc[data['DiP']>0]['RCHI']), abs(data.loc[data['DiP']>0]['RCORREL']))
plt.xlabel('Chi Sq')
plt.ylabel('Correlation')
plt.tight_layout()

plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(abs(data.loc[data['DiP']>0]['RCORREL']), data.loc[data['DiP']>0]['DiP'], data.loc[data['DiP']>0]['RCHI'])
plt.xlabel('Correlation')
plt.ylabel('DiP')
ax.set_zlabel('Chi Sq')
plt.tight_layout()

# Display specific data columns for the year 2010
print(data.loc[data['RYEAR']==2010][['RC10','RRR']])

# Convert all values in 'RC10' column to their absolute values to ensure non-negativity
data['RC10'] = abs(data['RC10'])

# Filter the data to include only rows where 'RCORREL' is greater than 0.5
data_WF = data.loc[data['RCORREL']>0.5]
data_WF = data.loc[(data['RCHI']<0.4) & (data['RCORREL']>0.5)]

# Extract 'RC10' values from the filtered data where values are less than 3
rc10_data = data_WF.loc[data_WF['RC10']<3]['RC10']
raw_rc10_data = data.loc[data['RC10']<3]['RC10']

# Define custom bins for plotting histograms
bins = np.linspace(0, 3, 31)

# Create a histogram plot of 'RC10' data
plt.figure()
plt.hist(raw_rc10_data, bins=bins, density=False, alpha=0.6, color='g', label = 'All events')

# Fit a normal distribution to the histogram data and plot the distribution
mu, std = norm.fit(raw_rc10_data)
x = np.linspace(0, 3, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p*0.1*len(raw_rc10_data), color='k', linestyle=':', lw=1, label=f'Normal Fit:\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

plt.hist(rc10_data, bins=bins, density=False, alpha=0.6, color='r', label = "'Good' events")

# Fit a normal distribution to the histogram data and plot the distribution
mu, std = norm.fit(rc10_data)
x = np.linspace(0, 3, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p*0.1*len(rc10_data), color='k', linestyle='--', lw=0.8, label=f'Normal Fit:\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

# Add labels and title to the histogram
plt.xlabel('$|C_{10}|$')
plt.ylabel('Number of ICMEs')
plt.title('Histogram of $C_{10}$', fontsize=16)
plt.legend()

plt.tight_layout()

# Print the number of data points meeting the earlier criteria
print(len(rc10_data))
print(len(raw_rc10_data))
rc10_data2 = data_WF.loc[data_WF['RC10']<Cnm_cut]['RC10']
raw_rc10_data2 = data.loc[data['RC10']<Cnm_cut]['RC10']
print(len(rc10_data2))
print(len(raw_rc10_data2))

# Load additional datasets for interpolation
datax = np.loadtxt('C10_tau10_v5.csv', delimiter=',')
datay = np.loadtxt('lambda_tau10_v5.csv', delimiter=',')

# Create an interpolation function from the loaded data
interpolator = interp1d(datax, np.log10(datay), kind='linear', fill_value="extrapolate")

# Interpolate the values for the filtered 'RC10' data
interpolated_sigmas = interpolator(rc10_data2)/2
raw_interpolated_sigmas = interpolator(raw_rc10_data2)/2
# print(max(interpolated_sigmas))

# Create and plot a histogram for the interpolated values
plt.figure()
bins = np.linspace(-11, 4, 31) / 2
plt.hist(raw_interpolated_sigmas, bins=bins, density=False, alpha=0.6, color='g', label = 'All events')

def custom_formatter2(x, pos):
    return r'10$^{%1.0f}$' % x
plt.xticks(2 * np.linspace(-5, 2, 8) / 2)
formatter = FuncFormatter(custom_formatter2)
plt.gca().xaxis.set_major_formatter(formatter)

d = bins[1] - bins[0]

mu, std = norm.fit(raw_interpolated_sigmas)
xmin, xmax = -11, 4
x = np.linspace(xmin, xmax, 100) / 2
p = norm.pdf(x, mu, std)
plt.plot(x, p * d * len(raw_rc10_data2), color='k', linestyle=':', lw=1, label=r'Fit: $\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

plt.hist(interpolated_sigmas, bins=bins, density=False, alpha=0.6, color='r', label = "'Good' events")

# def custom_formatter3(x, pos):
#     return '{%1.0f}' % (x * d * len(rc10_data2))
# formatter = FuncFormatter(custom_formatter3)
# plt.gca().yaxis.set_major_formatter(formatter)

# Fit and plot a normal distribution to the histogram of interpolated sigmas
mu, std = norm.fit(interpolated_sigmas)
xmin, xmax = -11, 4
x = np.linspace(xmin, xmax, 100) / 2
p = norm.pdf(x, mu, std)
plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw=0.8, label=r'Fit: $\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

plt.xlabel(r'$\sigma$')
plt.ylabel('Number of ICMEs')
plt.title(r'Histogram of $\sigma$', fontsize=16)
plt.legend()

plt.tight_layout()

plt.figure()
bins = np.linspace(-11,4,31)
plt.hist(interpolated_sigmas*2, bins=bins, density=False, alpha=0.6, color='r')

def custom_formatter2(x, pos):
    return r'10$^{%1.0f}$' %x
plt.xticks(2*np.linspace(-5,2,8))
formatter = FuncFormatter(custom_formatter2)
plt.gca().xaxis.set_major_formatter(formatter)

d = bins[1]-bins[0]
def custom_formatter3(x, pos):
    return '{%1.0f}' %(x * d * len(rc10_data2))
#plt.yticks(np.linspace(0,12,5)/(d * len(rc10_data2)))
formatter = FuncFormatter(custom_formatter3)
#plt.gca().yaxis.set_major_formatter(formatter)

# Fit a normal distribution
mu, std = norm.fit(interpolated_sigmas*2)
xmin, xmax = -11, 4
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw = 0.8, label=r'Lognormal Fit:\newline$\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

plt.xlabel(r'$\lambda$')
plt.ylabel('Number of ICMEs')
plt.title(r'Histogram of $\lambda$', fontsize=16)
plt.legend()

plt.tight_layout()

T_char = 1/data_WF.loc[data_WF['RC10']<Cnm_cut]['RBY0']/1e-9/np.sqrt(10**(interpolated_sigmas*2))*data_WF.loc[data_WF['RC10']<Cnm_cut]['RRR']*1.5e11*np.sqrt(4*3.14*1e-7*1.16*data_WF.loc[data_WF['RC10']<Cnm_cut]['np']*1e6*1.6*1e-27)
raw_T_char = 1/data.loc[data['RC10']<Cnm_cut]['RBY0']/1e-9/np.sqrt(10**(raw_interpolated_sigmas*2))*data.loc[data['RC10']<Cnm_cut]['RRR']*1.5e11*np.sqrt(4*3.14*1e-7*1.16*data.loc[data['RC10']<Cnm_cut]['np']*1e6*1.6*1e-27)

bins = np.linspace(0,6,25)
d = bins[1]-bins[0]
bins[:8] = np.linspace(0,np.log10(2.5*24),8)
plt.figure()
plt.hist(np.log10(raw_T_char/3600), bins=bins, density=False, alpha=0.6, color='g', label = 'All events')

mu, std = norm.fit(np.log10(raw_T_char/3600))
xmin, xmax = 0, 6.1
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p * d * len(raw_rc10_data2), color='k', linestyle='--', lw = 0.8, label=r'Fit: $\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

plt.hist(np.log10(T_char/3600), bins=bins, density=False, alpha=0.6, color='r', label = "'Good' events")

def custom_formatter4(x, pos):
    return r'10$^{%1.0f}$' %x
plt.xticks(np.linspace(0,6,7))
formatter = FuncFormatter(custom_formatter4)
plt.gca().xaxis.set_major_formatter(formatter)

# Fit a normal distribution
mu, std = norm.fit(np.log10(T_char/3600))
xmin, xmax = 0, 6.1
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p * d * len(rc10_data2), color='k', linestyle='--', lw = 0.8, label=r'Fit: $\mu=10^{%1.2f}$, scale=$10^{%1.2f}$' %(mu,std))

# print(np.sort(T_char/3600))

plt.xlabel(r'$T_c$ (h)')
plt.ylabel('Number of ICMEs')
plt.title(r'Histogram of $T_c$', fontsize=16)
plt.legend()
plt.tight_layout()
plt.axvline(np.log10(2.5*24),linestyle=':', color='k', lw=1)
plt.show()