import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Wind_wave

g = 9.806  # Gravitational G (m/s^2)

b = -1.5  # Rotation center in Y axis (m)
L = 15    # Wave length (m)
d = 300   # Depth (m)

k = 2 * np.pi / L  # Wave number (1/m)
c = np.sqrt(g / k * np.tanh(d * k))  # Speed in X direction  m/s
H = np.exp(k * b) / k  # Wave height (m)
T = L / c  # Wave period (s)

# Approx formula to estimate acceleration on top of wave
a_min_est = - g * np.exp(b * 2 * np.pi / L) / (1 - 5./3. * np.exp(b * 2 * np.pi / L))
# bottom of wave
a_max_est = g * np.exp(b * 2 * np.pi / L) / (1 + 7./3. * np.exp(b * 2 * np.pi / L))

# or (reverse)
# b_est = (L / (2 * np.pi)) * np.log(a_min / ((5.0 * a_min / 3.0) - g))
# b_est = (L / (2 * np.pi)) * np.log(a_max / (g - (7.0 * a_max / 3.0)))

# Also
# L = g * T * T / (2 * np.pi)

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}')

dt = 0.01
n_timesteps = int(20 * T / dt)

x_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)
velY_val = np.zeros(n_timesteps)
accY_val = np.zeros(n_timesteps)

for ii in range(n_timesteps):
    t = ii * dt
    x = H * np.sin(k * c * t)
    y = - H * np.cos(k * c * t)
    x_val[ii] = x / c + t
    y_val[ii] = y
    if ii > 0:
        velY_val[ii] = (y - y_val[ii - 1]) / (x_val[ii] - x_val[ii - 1])
        accY_val[ii] = (velY_val[ii] - velY_val[ii - 1]) / (x_val[ii] - x_val[ii - 1])

n_off = int(T / dt / 10)
interp_steps = n_timesteps - 2 * n_off
time_val = np.zeros(interp_steps)
y_val_A = np.zeros(interp_steps)
velY_val_A = np.zeros(interp_steps)
accY_val_A = np.zeros(interp_steps)

f_Y = interp1d(x_val, y_val)
f_velY = interp1d(x_val, velY_val)
f_accY = interp1d(x_val, accY_val)

for ii in range(interp_steps):
    t = (ii + n_off) * dt
    time_val[ii] = t
    if ii > 0:
        y_val_A[ii] = f_Y(t)
        velY_val_A[ii] = f_velY(t)
        accY_val_A[ii] = f_accY(t)

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x_val, y_val, label="Reference Pos")
axarr[0].plot(time_val, y_val_A, label="Reference Pos Extrap")
axarr[0].grid()
axarr[0].legend()

axarr[1].plot(x_val, velY_val, label="Reference Vertical Velocity")
axarr[1].plot(time_val, velY_val_A, label="Reference Vertical Velocity Extrap")
axarr[1].grid()
axarr[1].legend()

axarr[2].plot(x_val, accY_val, label="Reference Vertical Accel")
axarr[2].plot(time_val, accY_val_A, label="Reference Vertical Accel Extrap")
axarr[2].grid()
axarr[2].legend()

file = open("trochoidal_wave.txt", "w+")
for ii in range(interp_steps):
    if ii > 0:
        file.write(f'{time_val[ii]:,.4f}, {accY_val_A[ii]:,.8f}, {y_val_A[ii]:,.8f}, {velY_val_A[ii]:,.8f}\n')
file.close()

plt.show()
