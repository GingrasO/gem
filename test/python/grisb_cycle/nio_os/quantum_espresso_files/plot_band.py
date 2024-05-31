import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('nio_bands.dat.gnu').T
data_w = np.loadtxt('nio_band.dat').T

plt.plot(data[0], data[1], 'o', ms=2)
plt.plot(data_w[0]*0.47, data_w[1], 'b+', ms=1)
plt.show()
