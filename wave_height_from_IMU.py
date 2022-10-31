# Based on: https://github.com/juangallostra/AltitudeEstimation/tree/master/extras
#
# Vertical acceleration
# estimation from IMU readings via a Kalman filter
#
# Author: Juan Gallostra
# Date: 16-05-2018

import serial
import time
import numpy as np
import numpy.linalg as la

# desired sampling period
DESIRED_SAMPLING = 0.02

# standard deviation of sensors - educated guess
sigma_accel = 0.2
sigma_gyro = 0.2
sigma_baro = 5

# gravity in m/s^2 
g = 9.81

# more guesses - it has to be less that one for sure (I think)
ca = 0.5

# gain of complementary filter
Kc = np.array([np.sqrt(2*(sigma_accel/sigma_baro)), sigma_accel/sigma_baro]) 

# Serial parameters
PORT = '/dev/ttyACM0'
BAUDRATE = 9600


def skew(v):
	"""
	Returns the skew symmetric matrix of a vector
	"""
	return np.array([[0,   -v[2],  v[1]],
					 [v[2],   0,  -v[0]],
					 [-v[1], v[0],  0]])


def vector_angle(v1, v2):
    """ 
    Returns the angle in degrees between vectors 'v1' and 'v2'    
    """
    return np.arccos(np.dot(v1, v2) / (la.norm(v1) * la.norm(v2)))*180/np.pi


def millibars_to_meters(mb, ground_height=0):
	"""
	Relate pressure to height
	"""
	return 44330 * (1 - (mb/1013.25)**0.19) - ground_height


def calibrate(baro, ground_pressure, HISTORY, count):
	"""
	Compute the ground pressure and altitude from a series od readings
	"""
	ground_pressure -= ground_pressure/8
	if len(HISTORY) > count % 48:
		del HISTORY[count % 48]
	HISTORY.insert(count % 48, baro)
	ground_pressure += sum(HISTORY)/48
	if count == 200:
		return True, millibars_to_meters(ground_pressure/8), ground_pressure, HISTORY
	return False, millibars_to_meters(ground_pressure/8), ground_pressure, HISTORY


def get_sensor_data(serial_obj):
	"""
	Get accel, gyro and barometer data from serial
	"""
	raw_data = serial_obj.readline().rstrip().split(",")
	data = map(float, raw_data)
	# split into gyro and accel readings
	accel = np.array(data[:3])*g
	# account for gyro bias
	gyro = np.array(data[3:6])
	# pressure
	baro = data[-2]
	return accel, gyro, baro


def get_prediction_covariance(state_prev, t, sigma_gyro):
	"""
	Get the prediction covariance matrix
	"""
	Sigma = np.power(sigma_gyro, 2)*np.identity(3) 
	return -np.power(t, 2)*skew(state_prev).dot(Sigma).dot(skew(state_prev))


def get_measurement_covariance(ca, a_sensor_prev, sigma_accel):
	"""
	Get the measurement covariance matrix
	"""
	Sigma = np.power(sigma_accel, 2)*np.identity(3)
	return Sigma + (1.0/3)*np.power(ca, 2)*la.norm(a_sensor_prev)*np.identity(3)


def predict_state(gyro_prev, z, T):
	"""
	Predict the state evolution of the system one step ahead
	"""
	return (np.identity(3) - T*skew(gyro_prev)).dot(z)


def predict_error_covariance(gyro_prev, z_prev, T, P, sigma_gyro):
	"""
	Predict the covariance matrix from the data we have 
	"""
	Q = get_prediction_covariance(z_prev, T, sigma_gyro) # Prediction covariance matrix
	return (np.identity(3) - T*skew(gyro_prev)).dot(P).dot((np.identity(3) - T*skew(gyro_prev)).T) + Q 


def update_kalman_gain(P, H, ca, a_sensor_prev, sigma_accel):
	"""
	Compute the gain from the predicted error covariance matrix
	and the measurement covariance matrix
	"""
	R = get_measurement_covariance(ca, a_sensor_prev, sigma_accel)
	return P.dot(H.T).dot(la.inv(H.dot(P).dot(H.T) + R))

def update_state_with_measurement(predicted_state, K, measurement, H):
	"""
	Update the state estimate with the measurement
	"""
	return predicted_state + K.dot(measurement - H.dot(predicted_state))

def update_error_covariance(P, H, K):
	"""
	Update the error covariance with the calculated gain
	"""
	return (np.identity(3) - K.dot(H)).dot(P) 

def ZUPT(a_earth, vertical_vel, zupt_history, zupt_counter):
	"""
	Apply zero-velocity update to limit drift error. When the
	zero speed is detected then the speed is set to zero in 
	preference to the complementary filter integral.
	The zero speed is detected when during the last 12 readings
	the estimated acceleration is lower than the threshold  
	"""
	THRESHOLD = 0.3
	if len(zupt_history) > zupt_counter % 12:
		del zupt_history[zupt_counter % 12]
	zupt_history.insert(zupt_counter % 12, a_earth)
	 
	if sum([la.norm(val) > THRESHOLD for val in zupt_history]) == 0:
		return 0, zupt_history, zupt_counter
	return vertical_vel, zupt_history, zupt_counter

def interpolate(curr_x, x_init, x_end, y_init, y_end):
	"""
	Compute an intermediate value between two points by
	linear interpolation
	"""
	m = (y_end - y_init)/(x_end - x_init)
	n = y_end - x_end * m
	return m * curr_x + n

def interpolate_array(current_t, t_init, t_end, vector_init, vector_end):
	"""
	Compute an intermediate vector between two vectors by linear interpolation
	of each of its components
	"""
	return [interpolate(current_t, t_init, t_end, values[0], values[1]) for values in zip(vector_init, vector_end)]




# Serial communication object
serial_com = serial.Serial(PORT, BAUDRATE)

# Initialise needed variables
prev_time = time.time()
ZUPT_counter = 0
z = np.array([0, 0, 1]) # assume earth and body frame have same orientation
a_sensor = np.zeros(3) # the components of the acceleration are all 0
gyro_prev = np.zeros(3)
P = np.array([[100, 0, 0],[0, 100, 0],[0, 0, 100]]) # initial error covariance matrix
H = g*np.identity(3) # observation transition matrix
v = 0 # vertical velocity

# This paramaters and variables will be calculated and used 
# during baro calibration and later used for altitude estimation
ground_pressure = 0
ground_height = 0
HISTORY = []
calibrated = False
count = 0

# for complementary filter
zupt_history = []
zupt_counter = 0
baro_prev = 0
a_earth_prev = 0

# for kalman filter
z_prev = z

# for interpolation
i_gyro_prev = np.zeros(3)
i_baro_prev = np.zeros(3)

# This is where the magic happens,
# so pay close attention
while True:

	# get new sensor data
	accel, gyro, baro = get_sensor_data(serial_com)
	if not calibrated:
		calibrated, ground_height, ground_pressure, HISTORY = calibrate(baro, ground_pressure, HISTORY, count)
		h = 0
		count += 1
	# Calculate sampling period
	curr_time = time.time()
	T = curr_time - prev_time

	# oversample via linear interpolation
	for delta_t in np.linspace(DESIRED_SAMPLING, T, T/DESIRED_SAMPLING):

		# interpolate values
		i_accel = np.array(interpolate_array(delta_t, 0, T, accel_prev, accel))
		i_gyro = np.array(interpolate_array(delta_t, 0, T, gyro_prev, gyro))
		i_baro = interpolate(delta_t, 0, T, baro_prev, baro)

		# Kalman filter for vertical acceleration estimation

		# Prediction update with data from previous iteration and sensorss
		z = predict_state(i_gyro_prev, z_prev, DESIRED_SAMPLING) # State prediction
		z /= la.norm(z)
		P = predict_error_covariance(i_gyro_prev, z_prev, DESIRED_SAMPLING, P, sigma_gyro)
		# Measurement update
		K = update_kalman_gain(P, H, ca, a_sensor, sigma_accel)
		measurement = accel - ca*a_sensor
		z = update_state_with_measurement(z, K, measurement, H)
		z /= la.norm(z)
		P = update_error_covariance(P, H, K)

		# compute the acceleration from the estimated value of z
		a_sensor = i_accel - g*z
		# Acceleration in earth reference frame
		a_earth = np.vdot(a_sensor, z)

		# Complementary filter for altitude and vertical velocity estimation

		state = np.array([h, v])
		if baro_prev and a_earth_prev:
			state = np.array([[1, DESIRED_SAMPLING],[0, 1]]).dot(state) + \
		        	np.array([[1, DESIRED_SAMPLING/2],[0, 1]]).dot(Kc)*DESIRED_SAMPLING*(millibars_to_meters(i_baro_prev, ground_height) - h) + \
		        	np.array([DESIRED_SAMPLING/2, 1])*DESIRED_SAMPLING*a_earth_prev
		h, v = state

		# ZUPT
		v, zupt_history, zupt_counter = ZUPT(a_earth, v, zupt_history, zupt_counter)
		zupt_counter += 1

		# to see what's going on
		print a_earth, v, h, millibars_to_meters(i_baro_prev, ground_height)

		# complementary filter estimates from values of previous measurements
		i_baro_prev = i_baro
		a_earth_prev = a_earth
		# for next kalman iteration
		i_gyro_prev = i_gyro
		z_prev = z

	# for next interpolation
	baro_prev = baro
	gyro_prev = gyro
	accel_prev = accel
	# Update time of last measurement
	prev_time = curr_time

serial_com.close()
