import numpy as np
from scipy import linalg
from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Wind_wave

# More here: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html

g = 9.81  # Gravitational G (m/s^2)

b = -2  # Rotation center in Y axis (m)
L = 15  # Wave length (m)
d = 3000000  # Depth (m)

k = 2 * np.pi / L  # Wave number (1/m)
c = np.sqrt(g / k)  # Speed in X direction  m/s
H = np.exp(k * b) / k  # Wave height (m)
T = L / c  # Wave period (s)

# formula for  vertical acceleration on top of wave
a_min_est = g / (1 - np.exp(- b * 2 * np.pi / L))
# bottom of wave
a_max_est = g / (1 + np.exp(- b * 2 * np.pi / L))

# or (reverse)
# b_est = - (L / (2 * np.pi)) * np.log(1 - g / a_min)
# b_est = - (L / (2 * np.pi)) * np.log(g / a_max - 1)

# Also
# L = g * T * T / (2 * np.pi) if depth is infinite

# Doppler effect
# f_observed - observed frequency (1/s)
# L_source - wave length (m)
# L_source = (np.sign(delta_v) * np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (4 * np.pi * (f_observed ** 2))

print(f'Length: {L}, Height: {H}, Period: {T}, Speed: {c}, B: {b}')

dt = 0.01
n_timesteps = int(8 * T / dt)

t_val = np.zeros(n_timesteps)
y_val = np.zeros(n_timesteps)
velY_val = np.zeros(n_timesteps)
accY_val = np.zeros(n_timesteps)

x = np.zeros(n_timesteps)
z = np.zeros(n_timesteps)

for ii in range(n_timesteps):
    t = ii * dt
    t_val[ii] = t
    x[ii] = (H * np.sin(c * k * t) + c * t) / c
    z[ii] = - H * np.cos(c * k * t)
    if ii > 0:
        x_ = x[ii]
        y_ = z[ii]
        y_val[ii] = y_
        velY_val[ii] = (y_ - z[ii - 1]) / (x[ii] - x[ii - 1])
        accY_val[ii] = (velY_val[ii] - velY_val[ii - 1]) / (x[ii] - x[ii - 1])

print(f'max_y_val min_y_val H (m): {max(y_val):,.4f} {min(y_val):,.4f} {(max(y_val) - min(y_val))/2:,.4f}')

# Finding frequency

SAMPLE_RATE = 1.0 / dt
N_SAMP = n_timesteps

w = fft.rfft(accY_val)
freqs = fft.rfftfreq(N_SAMP, 1 / SAMPLE_RATE)

# Find the peak in the coefficients
idx = np.argmax(np.abs(w))
freq = freqs[idx]
freq_in_hertz = abs(freq)
period = 1 / freq_in_hertz
print(freq_in_hertz, 1 / freq_in_hertz)

# Doppler effect
upwind_speed = 2.0  # m/s (relative to waves direction, SPD * cos(TWA))
f_observed = (1 + upwind_speed / c) * (1 / T)
print(f'observed_freq upwind (Hz): {f_observed:,.4f}')
delta_v = upwind_speed
L_source1 = (np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (
            4 * np.pi * (f_observed ** 2))
print(f'L_source upwind (m): {L_source1:,.4f}')

downwind_speed = - 4  # m/s (relative to waves direction, SPD * cos(TWA))
f_observed = (1 + downwind_speed / c) * (1 / T)
print(f'observed_freq downwind (Hz): {f_observed:,.4f}')
delta_v = downwind_speed
L_source2 = (- np.sqrt(8 * f_observed * g * np.pi * delta_v + g ** 2) + 4 * f_observed * np.pi * delta_v + g) / (
            4 * np.pi * (f_observed ** 2))
print(f'L_source downwind (m): {L_source2:,.4f}')

# Low pass filter (Butterworth)
sos = signal.butter(2, freq_in_hertz * 8, 'low', fs=SAMPLE_RATE, output='sos')
low_pass_filtered = signal.sosfilt(sos, accY_val)

w_low = fft.rfft(low_pass_filtered)
freqs_low = fft.rfftfreq(N_SAMP, 1 / SAMPLE_RATE)

# Calc min/max accel
low_pass_filtered_min_a = min(low_pass_filtered)
low_pass_filtered_max_a = max(low_pass_filtered)

print(f'low_pass_filtered_min_a (m/s^2): {low_pass_filtered_min_a:,.4f}')
print(f'low_pass_filtered_max_a (m/s^2): {low_pass_filtered_max_a:,.4f}')

b_from_min_a = - (L_source1 / (2 * np.pi)) * np.log(1 - g / low_pass_filtered_min_a)
b_from_max_a = - (L_source1 / (2 * np.pi)) * np.log(g / low_pass_filtered_max_a - 1)

H_from_min_a = np.exp(2 * np.pi * b_from_min_a / L_source1) * L_source1 / 2 / np.pi
print(f'H_from_min_a upwind (m): {H_from_min_a:,.4f}  b_from_min_a={b_from_min_a:,.4f}')
H_from_max_a = np.exp(2 * np.pi * b_from_max_a / L_source1) * L_source1 / 2 / np.pi
print(f'H_from_max_a upwind (m): {H_from_max_a:,.4f}  b_from_max_a={b_from_max_a:,.4f}')
H_avg = (H_from_min_a + H_from_max_a) / 2
print(f'H_avg upwind (m): {H_avg:,.4f}')

b_from_min_a = - (L_source2 / (2 * np.pi)) * np.log(1 - g / low_pass_filtered_min_a)
b_from_max_a = - (L_source2 / (2 * np.pi)) * np.log(g / low_pass_filtered_max_a - 1)

H_from_min_a = np.exp(2 * np.pi * b_from_min_a / L_source2) * L_source2 / 2 / np.pi
print(f'H_from_min_a downwind (m): {H_from_min_a:,.4f}  b_from_min_a={b_from_min_a:,.4f}')
H_from_max_a = np.exp(2 * np.pi * b_from_max_a / L_source2) * L_source2 / 2 / np.pi
print(f'H_from_max_a downwind (m): {H_from_max_a:,.4f}  b_from_max_a={b_from_max_a:,.4f}')
H_avg = (H_from_min_a + H_from_max_a) / 2
print(f'H_avg downwind (m): {H_avg:,.4f}')

f, axarr = plt.subplots(4)

axarr[0].plot(x, z, label="Reference Pos")
axarr[0].grid()
axarr[0].legend()

axarr[1].plot(x, velY_val, label="Reference Vertical Velocity")
axarr[1].grid()
axarr[1].legend()

axarr[2].plot(x, accY_val, label="Reference Vertical Accel")
axarr[2].plot(t_val, low_pass_filtered, label="Low Pass Filtered Vertical Accel")
axarr[2].grid()
axarr[2].legend()

axarr[3].plot(freqs, np.abs(w), label=f'Freq Hz: {freq_in_hertz:,.4f}, Period (s): {period:,.4f}')
axarr[3].plot(freqs_low, np.abs(w_low), label=f'Low Passed Freq Hz')
axarr[3].set_xlim([0, 5])
axarr[3].legend()

file = open("trochoidal_wave.txt", "w+")
for ii in range(n_timesteps):
    if ii > 0:
        file.write(f'{t_val[ii]:,.4f}, {accY_val[ii]:,.8f}, {y_val[ii]:,.8f}, {velY_val[ii]:,.8f}\n')
file.close()

plt.show()

# leeway empirical formula
heel = 15.0  # (deg)
SPD = 5.0  # speed through water (kt)
K = 10.0  # boat and load specific constant (kt^2), about 10.0
leeway = heel * K / (SPD ** 2)  # leeway - (deg) angle to adjust heading to maintain constant COG (assuming no current)

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
# HDT = Heading true (relative to true north)
# HDM = Heading magnetic (relative to magnetic north)
#
# DFT = Current Drift (speed of current, relative to fixed earth)
# SET = Current Set (direction current flows toward relative to fixed earth true north)
#
# SOG = Speed Over Ground (relative to the fixed earth)
# COGT = Course Over Ground true (relative to the fixed earth true north)
# COGM = Course Over Ground magnetic (relative to the fixed earth magnetic north)
#
# GWS = Ground Wind Speed (relative to the fixed earth)
# GWD = Ground Wind Direction (relative to true north)
#
# TWA = True Wind Angle (relative to the heading, 0 = upwind, 180deg = downwind, (+ starboard, - port))
# TWS = True Wind Speed (relative to the water)
# TWD = True Wind Direction (relative to true north)
#
# POS = position LAT, LON (latitude, longitude)
# TB(POS1, POS2) = Bearing true (true north angle to maintain in course to reach from POS1 to POS2)
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
# From apparent to true:
#
# TWS = sqrt(AWS ** 2 + SPD ** 2 - 2 * AWS * SPD * cos(AWA - leeway(heel, SPD)))
#
# for starboard:
# TWA = arccos((AWS * cos(AWA - leeway(heel, SPD)) - SPD) / TWS)
#
# for port:
# TWA = - arccos((AWS * cos(AWA - leeway(heel, SPD)) - SPD) / TWS)
#
# There are other factors, boat heel, mast twist, upwash from the sails, wind shear


# Measurable input parameters
# t_start, t_end - time interval of measurements (about 5 mins)
# POS(t) as LAT(t), LON(t)
# AWA(t) AWS(t)
# COG(t) SOG(t)
# HDM(t) + mag_variation -> HDT(t)
# DFT(t) SET(t) - (possibly from current/tide stations harmonics data)
# heel(t), pitch(t)
# SPD(t) - possibly (might be missing) => leeway(heel(t), SPD(t))
# accel(t, x, y, z), vertical_accel(t) via pitch,roll,heel
# ROT(t) - rate of turn


# Calculation steps:
# FFT to get observed wave frequency from acceleration (f_observed)
# Speed toward wave fronts (delta_v for Doppler frequency) from wind and speed data
#  COGT as true bearing from POS1 to POS2
#  Convert HDM to HDT using position and local mag declination, Use avg(HDT) vs COG and coordinates to calculate SPD
#  SPD = (DIST(POS1, POS2)/(t_end - t_start) - (DFT * cos(COGT - SET))) * cos(COGT - avg(HDT))
#  avg(leeway(heel(t), SPD))
#  use avg(AWA), AVG(AWS) and SPD to calculate TWS/TWA
#  TWS = sqrt(AVG(AWS) ** 2 + SPD ** 2 - 2 * AVG(AWS) * SPD * cos(avg(AWA)))
#  TWA = +- arccos((AVG(AWS) * cos(avg(AWA)) - SPD) / TWS)
#  calculate delta_v as SPD * cos(TWA)
# Calculate L_source (source wave length) for trochoidal wave model from f_observed and delta_v using Doppler formulas
# Low pass filter for accel data
# min/max accel after low pass
# Calculate b value for trochoidal wave model from known L_source and min/max accel after low pass
# Calculate wave height from b and L_source


# Assumptions:
#   No tacks, jibes during sample
#   Heading is mostly steady
#   Check validity of accel (against g)
#   Trochoidal wave model
#   Approx formula for b (in trochoidal wave model)

