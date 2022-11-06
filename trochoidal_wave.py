import numpy as np

import matplotlib.pyplot as plt

a = 0
b = -20
L = 30
g = 9.8
k = 2 * np.pi / L
c = np.sqrt(g / k)
u = np.sqrt(g * L / (2 * np.pi))

n_timesteps = 700
dt = 0.01

times = np.zeros((n_timesteps))
y_val = np.zeros((n_timesteps))

for ii in range(n_timesteps):
    t = ii * dt
    x = a + (np.exp(k * b) / k * np.sin(k * (a + c * t)))
    # x = ii * dt
    # t = ((np.arcsin((x - a) * k / np.exp(k * b)) / k) - a) / c
    y = b - (np.exp(k * b) / k * np.cos(k * (a + c * t)))
    times[ii] = x + t * u
    y_val[ii] = y - b

plt.plot(times, y_val, "r-")
plt.grid()
plt.show()
