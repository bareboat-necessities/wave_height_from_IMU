import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

Data = np.loadtxt(fname="test-data.txt", delimiter=",", skiprows=0)

xs = Data[:, [1]]
x = xs.reshape(-1)

peaks, _ = find_peaks(x, width=30)
valleys, _ = find_peaks(-x, width=30)

plt.plot(peaks, x[peaks], "vg")
plt.plot(valleys, x[valleys], "vg")
plt.plot(x)
plt.legend(['width'])
plt.show()
