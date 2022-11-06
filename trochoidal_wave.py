import numpy as np

import matplotlib.pyplot as plt

g = 9.806

b = -1
L = 10

k = 2 * np.pi / L
c = np.sqrt(g / k)
u = np.sqrt(g * L / (2 * np.pi))
H = np.exp(k * b) / k
T = L / c

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {u}')

n_timesteps = 1000
dt = 0.01

x_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)

for ii in range(n_timesteps):
    t = ii * dt
    x = H * np.sin(k * c * t)
    #x = ii * dt
    #t = np.arcsin(x * k / np.exp(k * b)) / (k * c) + x
    y = - H * np.cos(k * c * t)
    x_val[ii] = x + t * u
    y_val[ii] = y

plt.plot(x_val, y_val, "r-")
plt.grid()
plt.show()
