import numpy as np

import matplotlib.pyplot as plt

g = 9.806  # Gravitational G (m/s^2)

b = -1.5    # Rotation center in Y axis (m)
L = 15       # Wave length (m)

k = 2 * np.pi / L                  # Wave number (1/m)
c = np.sqrt(g / k)                 # Speed in X direction  m/s
H = np.exp(k * b) / k              # Wave height (m)
T = L / c                          # Wave period (s)

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}')

n_timesteps = 2000
dt = 0.01

x_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)


def triangle_wave(w, a, p):
    return 2 * a / np.pi * np.arcsin(np.sin(2 * np.pi * w / p))


for ii in range(n_timesteps):
    t = ii * dt
    x = H * np.sin(k * c * t)
    y = - H * np.cos(k * c * t)
    x_val[ii] = x/c + t
    y_val[ii] = y

plt.plot(x_val, y_val, "r-")
plt.grid()
plt.show()
