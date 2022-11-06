import numpy as np

import matplotlib.pyplot as plt

a = 0
b = -20
L = 200
g = 9.8
pi = 3.14
k = 2 * pi / L
c = 4 #np.sqrt(g / k)
u = c * np.exp(k * b)

n_timesteps = 14000
dt = 0.01

times = np.zeros((n_timesteps))
y_val = np.zeros((n_timesteps))

for ii in range(n_timesteps):
    t = ii * dt
    x = a + (np.exp(k * b) / k * np.sin(k * (a + c * t)))
    #x = ii * dt
    #t = ((np.arcsin((x - a) * k / np.exp(k * b)) / k) - a) / c
    y = b - (np.exp(k * b) / k * np.cos(k * (a + c * t)))
    times[ii] = x + t * c
    y_val[ii] = y - b

plt.plot(times, y_val, "r-")
plt.grid()
plt.show()
