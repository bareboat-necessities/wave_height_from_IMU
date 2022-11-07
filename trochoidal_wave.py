import numpy as np

import matplotlib.pyplot as plt

g = 9.806   # Gravitational G (m/s^2)

b = -1.5    # Rotation center in Y axis (m)
L = 15      # Wave length (m)

k = 2 * np.pi / L                  # Wave number (1/m)
c = np.sqrt(g / k)                 # Speed in X direction  m/s
H = np.exp(k * b) / k              # Wave height (m)
T = L / c                          # Wave period (s)

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}')

dt = 0.01
n_timesteps = int(7 * T / dt)

x_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)
velY_val = np.zeros(n_timesteps)
accY_val = np.zeros(n_timesteps)

for ii in range(n_timesteps):
    t = ii * dt
    x = H * np.sin(k * c * t)
    y = - H * np.cos(k * c * t)
    x_val[ii] = x/c + t
    y_val[ii] = y
    if ii > 0:
        velY_val[ii] = (y - y_val[ii-1]) / (x_val[ii] - x_val[ii - 1])
        accY_val[ii] = (velY_val[ii] - velY_val[ii-1]) / (x_val[ii] - x_val[ii - 1])

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x_val, y_val, label="Reference Pos")
axarr[0].set_title('Position')
axarr[0].grid()
axarr[0].legend()

axarr[1].plot(x_val, velY_val, label="Reference Vertical Velocity")
axarr[1].set_title('Vertical Velocity')
axarr[1].grid()
axarr[1].legend()

axarr[2].plot(x_val, accY_val, label="Reference Vertical Accel")
axarr[2].set_title('Vertical Accel')
axarr[2].grid()
axarr[2].legend()

plt.show()
