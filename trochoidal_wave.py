import numpy as np
from scipy import linalg
from scipy import signal
from scipy import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Wind_wave

g = 9.806  # Gravitational G (m/s^2)

b = -2  # Rotation center in Y axis (m)
L = 15  # Wave length (m)
d = 300  # Depth (m)

k = 2 * np.pi / L  # Wave number (1/m)
c = np.sqrt(g / k * np.tanh(d * k))  # Speed in X direction  m/s
H = np.exp(k * b) / k  # Wave height (m)
T = L / c  # Wave period (s)

# Approx formula to estimate vertical acceleration on top of wave
a_min_est = - g * np.exp(b * 2 * np.pi / L) / (1 - 1.6 * np.exp(b * 2 * np.pi / L))
# bottom of wave
a_max_est = g * np.exp(b * 2 * np.pi / L) / (1 + 2.1 * np.exp(b * 2 * np.pi / L))

# or (reverse)
# b_est = (L / (2 * np.pi)) * np.log(a_min / ((1.6 * a_min) - g))
# b_est = (L / (2 * np.pi)) * np.log(a_max / (g - (2.1 * a_max)))

# Also
# L = g * T * T / (2 * np.pi) if depth is infinite

# Doppler effect
# f_observed - observed frequency (1/s)
# L_source - wave length (m)
# L_source = (np.sign(delta_v) * np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (4 * np.pi * (f_observed ** 2))

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}, B: {b}')

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

# Finding frequency

SAMPLE_RATE = 1.0 / dt
N_SAMP = interp_steps

w = fft.rfft(accY_val_A)
freqs = fft.rfftfreq(N_SAMP, 1 / SAMPLE_RATE)

# Find the peak in the coefficients
idx = np.argmax(np.abs(w))
freq = freqs[idx]
freq_in_hertz = abs(freq)
period = 1 / freq_in_hertz
print(freq_in_hertz, 1 / freq_in_hertz)

# Doppler effect
upwind_speed = 2.0  # m/s (relative to waves direction, STW * cos(TWA))
f_observed = (1 + upwind_speed / c) * (1 / T)
print(f'observed_freq upwind (Hz): {f_observed:,.4f}')
delta_v = upwind_speed
L_source1 = (np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (
            4 * np.pi * (f_observed ** 2))
print(f'L_source upwind (m): {L_source1:,.4f}')

downwind_speed = - 4  # m/s (relative to waves direction, STW * cos(TWA))
f_observed = (1 + downwind_speed / c) * (1 / T)
print(f'observed_freq downwind (Hz): {f_observed:,.4f}')
delta_v = downwind_speed
L_source2 = (- np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (
            4 * np.pi * (f_observed ** 2))
print(f'L_source downwind (m): {L_source2:,.4f}')


# TWA, SWT empirical formulas
heel = 15.0  # (deg)
SPD = 5.0  # speed through water (kt)
K = 10.0  # boat and load specific constant (kt^2), about 10.0
leeway = heel * K / (SPD ** 2)  # leeway - (deg)

#
# AWS = Apparent Wind Speed (relative to the boat heading)
# AWA = Apparent Wind Angle (relative to the bow heading, 0 to 180, starboard plus, port minus)
# AWD = Apparent Wind Direction (relative to true north)
#
# AGWS = Apparent Ground Wind Speed (relative to the boat course over the ground)
# AGWA = Apparent Ground Wind Angle (relative to the boat course over the ground, 0 to 180, starboard plus, port minus)
# AGWD = Apparent Ground Wind Direction (relative to true north)
#
# SPD = Knotmeter speed (relative to the water)
# HDT = Heading (relative to true north)
#
# DFT = Current Drift (speed of current, relative to fixed earth)
# SET = Current Set (direction current flows toward, relative to fixed earth true north)
#
# SOG = Speed Over Ground (relative to the fixed earth)
# COG = Course Over Ground (relative to the fixed earth)
#
# GWS = Ground Wind Speed (relative to the fixed earth)
# GWD = Ground Wind Direction (relative to true north)
#
# TWA = True Wind Angle (relative to the heading, 0 = upwind, 180deg = downwind, (+ starboard, - port))
# TWS = True Wind Speed (relative to the water)
# TWD = True Wind Direction (relative to true north)
#
# AWA = + for Starboard, – for Port
# AWD = H + AWA ( 0 < AWD < 360 )
#
# u = SOG * Sin (COG) – AGWS * Sin (AGWD)
# v = SOG * Cos (COG) – AGWS * Cos (AGWD)
#
# GWS = SQRT ( u*u + v*v )
#
# GWD = ATAN ( u / v )
#
#
# From true to apparent:
#
# AWS = sqrt(TWS ** 2 + SPD ** 2 + 2 * TWS * SPD * cos(TWA))
#
# AWA = arccos((TWS * cos(TWA) + SPD) / AWS)
#
# From apparent to true:
#
# TWS = sqrt(AWS ** 2 + SPD ** 2 - 2 * AWS * SPD * cos(AWA))
#
# for starboard:
# TWA = arccos((AWS * cos(AWA) - SPD) / TWS)
#
# for port:
# TWA = - arccos((AWS * cos(AWA) - SPD) / TWS)
#
# There are other factors, boat heel, mast twist, upwash from the sails, wind shear

# Low pass filter (Butterworth)
sos = signal.butter(2, freq_in_hertz * 8, 'low', fs=SAMPLE_RATE, output='sos')
low_pass_filtered = signal.sosfilt(sos, accY_val_A)

w_low = fft.rfft(low_pass_filtered)
freqs_low = fft.rfftfreq(N_SAMP, 1 / SAMPLE_RATE)

# Calc min/max accel
low_pass_filtered_min_a = min(low_pass_filtered)
low_pass_filtered_max_a = max(low_pass_filtered)

b_from_min_a = (L_source1 / (2 * np.pi)) * np.log(low_pass_filtered_min_a / ((1.6 * low_pass_filtered_min_a) - g))
b_from_max_a = (L_source1 / (2 * np.pi)) * np.log(low_pass_filtered_max_a / (g - (2.1 * low_pass_filtered_max_a)))

H_from_min_a = np.exp(2 * np.pi * b_from_min_a / L_source1) * L_source1 / 2 / np.pi
print(f'H_from_min_a upwind (m): {H_from_min_a:,.4f}  b_from_min_a={b_from_min_a:,.4f}')
H_from_max_a = np.exp(2 * np.pi * b_from_max_a / L_source1) * L_source1 / 2 / np.pi
print(f'H_from_max_a upwind (m): {H_from_max_a:,.4f}  b_from_max_a={b_from_max_a:,.4f}')
H_avg = (H_from_min_a + H_from_max_a) / 2
print(f'H_avg downwind (m): {H_avg:,.4f}')

b_from_min_a = (L_source2 / (2 * np.pi)) * np.log(low_pass_filtered_min_a / ((1.6 * low_pass_filtered_min_a) - g))
b_from_max_a = (L_source2 / (2 * np.pi)) * np.log(low_pass_filtered_max_a / (g - (2.1 * low_pass_filtered_max_a)))

H_from_min_a = np.exp(2 * np.pi * b_from_min_a / L_source2) * L_source2 / 2 / np.pi
print(f'H_from_min_a downwind (m): {H_from_min_a:,.4f}  b_from_min_a={b_from_min_a:,.4f}')
H_from_max_a = np.exp(2 * np.pi * b_from_max_a / L_source2) * L_source2 / 2 / np.pi
print(f'H_from_max_a downwind (m): {H_from_max_a:,.4f}  b_from_max_a={b_from_max_a:,.4f}')
H_avg = (H_from_min_a + H_from_max_a) / 2
print(f'H_avg downwind (m): {H_avg:,.4f}')

f, axarr = plt.subplots(4)

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
axarr[2].plot(time_val, low_pass_filtered, label="Low Pass Filtered Vertical Accel")
axarr[2].grid()
axarr[2].legend()

axarr[3].plot(freqs, np.abs(w), label=f'Freq Hz: {freq_in_hertz:,.4f}, Period (s): {period:,.4f}')
axarr[3].plot(freqs_low, np.abs(w_low), label=f'Low Passed Freq Hz')
axarr[3].set_xlim([0, 5])
axarr[3].legend()

file = open("trochoidal_wave.txt", "w+")
for ii in range(interp_steps):
    if ii > 0:
        file.write(f'{time_val[ii]:,.4f}, {accY_val_A[ii]:,.8f}, {y_val_A[ii]:,.8f}, {velY_val_A[ii]:,.8f}\n')
file.close()

plt.show()
