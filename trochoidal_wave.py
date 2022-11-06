import numpy as np

import matplotlib.pyplot as plt

g = 9.806

a = 0
b = -1
L = 10

k = 2 * np.pi / L
c = np.sqrt(g / k)
u = np.sqrt(g * L / (2 * np.pi))

n_timesteps = 1000
dt = 0.01

times = np.zeros((n_timesteps))
y_val = np.zeros((n_timesteps))

for ii in range(n_timesteps):
    t = ii * dt
    x = a + (np.exp(k * b) / k * np.sin(k * (a + c * t)))
    # x = ii * dt
    # t = ((np.arcsin((x - a) * k / np.exp(k * b)) / k) - a) / c
    y = - np.exp(k * b) / k * np.cos(k * (a + c * t))
    times[ii] = x + t * u
    y_val[ii] = y

plt.plot(times, y_val, "r-")
plt.grid()
plt.show()
