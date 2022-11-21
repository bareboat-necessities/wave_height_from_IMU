import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Wind_wave

g = 9.81  # Gravitational G (m/s^2)

n1_iter = 20
n2_iter = 15
db = 0.02
dL = 20.0

L_arr = np.zeros(n1_iter * n2_iter)
H_arr = np.zeros(n1_iter * n2_iter)

min_a_arr = np.zeros(n1_iter * n2_iter)
max_a_arr = np.zeros(n1_iter * n2_iter)

min_a_est_arr = np.zeros(n1_iter * n2_iter)
max_a_est_arr = np.zeros(n1_iter * n2_iter)

H_est1_arr = np.zeros(n1_iter * n2_iter)
H_est2_arr = np.zeros(n1_iter * n2_iter)

iii = 0
for n1 in range(n1_iter):
    for n2 in range(n2_iter):
        L = dL*(n1+1)    # Wave length (m)
        b = -db*L*(n2+8)  # Rotation center in Y axis (m)
        d = 8000   # Depth (m)

        k = 2 * np.pi / L  # Wave number (1/m)
        c = np.sqrt(g / k * np.tanh(d * k))  # Speed in X direction  m/s
        H = np.exp(k * b) / k  # Wave height (m)
        T = L / c  # Wave period (s)

        #print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}')

        dt = 0.01
        n_timesteps = int(2 * T / dt)

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

        L_arr[iii] = L
        H_arr[iii] = H
        min_a = min(accY_val_A)
        max_a = max(accY_val_A)
        min_a_arr[iii] = min_a
        max_a_arr[iii] = max_a

        a_min_est = - g * np.exp(b * 2 * np.pi / L) / (1 - 1.6 * np.exp(b * 2 * np.pi / L))
        a_max_est = g * np.exp(b * 2 * np.pi / L) / (1 + 2.1 * np.exp(b * 2 * np.pi / L))
        b_est1 = (L / (2 * np.pi)) * np.log(a_min_est / ((1.6 * a_min_est) - g))
        b_est2 = (L / (2 * np.pi)) * np.log(a_max_est / (g - (2.1 * a_max_est)))
        H_est1 = np.exp(k * b_est1) / k
        H_est2 = np.exp(k * b_est2) / k
        min_a_est_arr[iii] = a_min_est
        max_a_est_arr[iii] = a_max_est
        H_est1_arr[iii] = H_est1
        H_est2_arr[iii] = H_est2

        print(f'L={L:,.8f} H={H:,.8f} T={T:,.8f} b={b:,.8f} min_a={min_a:,.8f} max_a={max_a:,.8f} a_min_est={a_min_est:,.8f} a_max_est={a_max_est:,.8f}')
        iii = iii + 1

fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter3D(min_a_arr, L_arr, H_arr, color='blue')
ax1.scatter3D(min_a_est_arr, L_arr, H_est1_arr, color='red')
ax1.set_title('Height/Height Estimated')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter3D(max_a_arr, L_arr, H_arr, color='blue')
ax2.scatter3D(max_a_est_arr, L_arr, H_est2_arr, color='red')

plt.show()
