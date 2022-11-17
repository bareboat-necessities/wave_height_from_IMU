import sys, getopt

sys.path.append('.')

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

if (not imu.IMUInit()):
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

while True:
    if imu.IMURead():
        # x, y, z = imu.getFusionData()
        # print("%f %f %f" % (x,y,z))
        data = imu.getIMUData()
        fusionPose = data["fusionPose"]
        fusionQPose = data["fusionQPose"]
        accel = data["accel"]
        timestamp = data["timestamp"]
        timestamp_ms = timestamp / 1000
        alignedAccel = quaternion.multiply(data['fusionQPose'], accel)
        print("t: %f, r: %f p: %f y: %f ax: %f ay: %f az: %f  aax: %f aay: %f aaz: %f" % (timestamp_ms,
                                     math.degrees(fusionPose[0]),
                                     math.degrees(fusionPose[1]),
                                     math.degrees(fusionPose[2]),
                                     accel[0],
                                     accel[1],
                                     accel[2],
                                     alignedAccel[0],
                                     alignedAccel[1],
                                     alignedAccel[2]), flush=True)
        time.sleep(poll_interval*1.0/1000.0)
        print("\033[A", end="", flush=False)
