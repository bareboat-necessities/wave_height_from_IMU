import numpy as np

import matplotlib.pyplot as plt

g = 9.806  # Gravitational G (m/s^2)

b = -1  # Rotation center in Y axis (m)
L = 10  # Wave length (m)

k = 2 * np.pi / L  # Wave number (1/m)
c = np.sqrt(g / k)  # Speed in X direction  m/s
H = np.exp(k * b) / k  # Wave height (m)
T = L / c  # Wave period (s)

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}')

n_timesteps = 4000
dt = 0.01

time_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)


def triangle_wave(w, a, p):
    return 2 * a / np.pi * np.arcsin(np.sin(2 * np.pi * w / p))


for ii in range(n_timesteps):
    tx = ii * dt
    tw = triangle_wave(tx, H / 2, L * 4)
    t = np.arcsin(tw) / (k * c)
    y = - H * np.cos(k * c * t)
    time_val[ii] = tx
    y_val[ii] = y
    print(f'{time_val[ii]}   {y_val[ii]}')

plt.plot(time_val, y_val, "r-")
plt.grid()
plt.show()
