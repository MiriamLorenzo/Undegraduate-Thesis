import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gamma
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True
# plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (6, 3)

data = pd.read_csv('results_modelcc_teresa.csv', delimiter=',')
print(data.describe())

data['RC10'] = abs(data['RC10'])

print(data.loc[data['RYEAR']==2010][['RC10','RRR']])

data_WF = data.loc[data['RCORREL']>0.6]
# print(data_WF.describe())
# print(data.loc[data['RCORREL']>0.9])

# Extract RC10 data
rc10_data = data_WF.loc[data_WF['RC10']<10]['RC10']
# Define personalized bins
bins = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 5, 9]

# Plot histogram
plt.figure()
plt.hist(rc10_data, bins=bins, density=True, alpha=0.6, color='g', label='Histogram')

# Fit a normal distribution
mu, std = norm.fit(rc10_data)
xmin, xmax = 0, 9
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, color='k', linestyle='--', lw = 0.8, label=f'Normal Fit:\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')

# Fit a lognormal distribution
shape, loc, scale = lognorm.fit(rc10_data)
lognorm_fit = lognorm(shape, loc, scale)
lognorm_pdf = lognorm_fit.pdf(x)
plt.plot(x, lognorm_pdf, color='k', linestyle='-', lw = 0.8, label=f'Lognormal Fit:\nshape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f}')

# # Fit a gamma distribution
# shape, loc, scale = gamma.fit(rc10_data)
# gamma_fit = gamma(shape, loc, scale)
# gamma_pdf = gamma_fit.pdf(x)
# plt.plot(x, gamma_pdf, color='k', linestyle=':', label='Gamma Fit')

# Labels and title
plt.xlabel('$|C_{10}|$')
plt.ylabel('Number of ICMEs')
plt.title('Histogram of $C_{10}$', fontsize=16)
plt.legend()

from matplotlib.ticker import FuncFormatter
# Define your custom formatting function
def custom_formatter(x, pos):
    return '{:.0f}'.format(x * np.diff(bins)[0] * len(rc10_data))

# print(np.diff(bins)[0] * len(rc10_data))
plt.yticks([2/17.4, 6/17.4, 10/17.4, 14/17.4, 18/17.4])
formatter = FuncFormatter(custom_formatter)
plt.gca().yaxis.set_major_formatter(formatter)

ticks = np.linspace(0, 9, 10)
plt.xticks(ticks, ['{:.0f}'.format(tick) for tick in ticks])

print(len(rc10_data))

plt.tight_layout()
plt.show()