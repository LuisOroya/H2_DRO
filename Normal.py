import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rc('font', family='serif', serif='cmr10')
plt.rcParams['axes.unicode_minus'] = False

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb} \usepackage{newtxtext, newtxmath}'

# Ajustar el tamaño de fuente globalmente
plt.rcParams.update({'font.size': 20})

# Generate data
mean = 0
std_devs = [1, 0.5, 2]
x = np.linspace(-10, 10, 1000)

# Create the plots
plt.figure(figsize=(10, 6))

for std in std_devs:
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.plot(x, y, label=f'Standard Deviation = {std}')

# Adding titles and labels
plt.title(r'Normal Distribution with Different Standard Deviations')
plt.xlabel(r'X')
plt.ylabel(r'Probability Density')
# plt.legend()
plt.grid(True)
plt.show()
