import sys, getopt

sys.path.append('.')

from pykalman import KalmanFilter
from scipy import linalg
import numpy as np

import RTIMU
import os.path
import time
import math
import quaternion

SETTINGS_FILE = "RTIMULib"

print("Using settings file " + SETTINGS_FILE + ".ini")
if not os.path.exists(SETTINGS_FILE + ".ini"):
    print("Settings file does not exist, will be created")

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)

print("IMU Name: " + imu.IMUName())

if not imu.IMUInit():
    print("IMU Init Failed")
    sys.exit(1)
else:
    print("IMU Init Succeeded")

# this is a good time to set any fusion parameters

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

poll_interval = imu.IMUGetPollInterval()
print("Recommended Poll Interval: %dmS\n" % poll_interval)

PosIntegral_Variance = 100  # TODO: ???
PosIntegral_Trans_Variance = 100  # TODO: ???

# time step
dt = 0.01  # TODO: get it from two sequential data["timestamp"]

# transition_matrix
F = [[1, dt, 0.5 * dt ** 2],
     [0, 1, dt],
     [0, 0, 1]]

# for transition offset
B = [(1.0 / 6) * dt ** 3,
     0.5 * dt ** 2,
     dt]

# observation_matrix
H = [1, 0, 0]

# transition_covariance
Q = [[PosIntegral_Trans_Variance, 0, 0],
     [0, 0.2, 0],
     [0, 0, 0.1]]

# observation_covariance
R = [[PosIntegral_Variance]]

# initial_state_mean
X0 = [0,  # height integral
      0,  # height
      0]  # velocity

# initial_state_covariance
P0 = [[PosIntegral_Variance, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]

# Evaluate signal offset from 0
Acc_Mean = [[0.0]]  # TODO: calculate using running avg over long periods 1 mins or so.

filtered_state_means = X0
filtered_state_covariances = P0

kf = KalmanFilter(transition_matrices=F,
                  observation_matrices=H,
                  transition_covariance=Q,
                  observation_covariance=R,
                  initial_state_mean=X0,
                  initial_state_covariance=P0)

while True:
    if imu.IMURead():
        data = imu.getIMUData()
        fusionPose = data["fusionPose"]
        fusionQPose = data["fusionQPose"]
        accel = data["accel"]
        timestamp = data["timestamp"]
        timestamp_ms = timestamp / 1000
        alignedAccel = quaternion.rotvecquat(accel, data['fusionQPose'])

        observation = 0
        acc = [[alignedAccel[2]]]
        transition_offset = B * (acc - Acc_Mean)
        filtered_state_means, filtered_state_covariances = kf.filter_update(
            filtered_state_mean=filtered_state_means,
            filtered_state_covariance=filtered_state_covariances,
            transition_offset=transition_offset,
            observation=observation
        )

        print("t: %f, r: %f p: %f y: %f ax: %f ay: %f az: %f  aax: %f aay: %f aaz: %f" % (
            timestamp_ms,
            math.degrees(fusionPose[0]),
            math.degrees(fusionPose[1]),
            math.degrees(fusionPose[2]),
            accel[0],
            accel[1],
            accel[2],
            alignedAccel[0],
            alignedAccel[1],
            alignedAccel[2]), flush=False)
        print("vel: %f pos %f" % (
            filtered_state_means[2],
            filtered_state_means[1]), flush=True)

        time.sleep(poll_interval * 1.0 / 1000.0)
        print("\033[A", end="", flush=False)
        print("\033[A", end="", flush=False)
