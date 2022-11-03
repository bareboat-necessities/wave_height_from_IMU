from pykalman import KalmanFilter
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt(fname="test-data.txt", delimiter=",", skiprows=0)

# Data description
#  Time
#  AccX - acceleration signal
#  RefPosX - real position (ground truth)
#  RefVelX - real velocity (ground truth)

Time = Data[:, [0]]
AccX = Data[:, [1]]
RefPosX = Data[:, [2]]
RefVelX = Data[:, [3]]

AccX_Value = AccX
AccX_Variance = 0.0007
PosIntegral_Variance = 1 # TODO: ???


# time step
dt = 0.01

# transition_matrix  
F = [[1, dt, 0.5*dt**2],
     [0,  1,       dt],
     [0,  0,        1]]


B = [ (1.0/6)*dt**3,
          0.5*dt**2,
                 dt]

# observation_matrix
H = [1, 0, 0]

# transition_covariance
Q = [[1,     0,    0],
     [0,   0.2,    0],
     [0,     0,  0.1]]

# observation_covariance
R = [[PosIntegral_Variance]]

# initial_state_mean
X0 = [0,                  # height integral
      0,                  # height
      0]                  # velocity

# initial_state_covariance
P0 = [[PosIntegral_Variance, 0,    0],
      [0,                    0,    0],
      [0,                    0,    0]]

n_timesteps = AccX_Value.shape[0]
n_dim_state = 3
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

kf = KalmanFilter(transition_matrices = F,
                  observation_matrices = H,
                  transition_covariance = Q,
                  observation_covariance = R,
                  initial_state_mean = X0,
                  initial_state_covariance = P0)

# iterative estimation for each new measurement
for t in range(n_timesteps):
    if t == 0:
        filtered_state_means[t] = X0
        filtered_state_covariances[t] = P0
    else:
        observation = 0
        transition_offset = B * AccX_Value[t]
        filtered_state_means[t], filtered_state_covariances[t] = (
            kf.filter_update(
                filtered_state_mean = filtered_state_means[t-1],
                filtered_state_covariance = filtered_state_covariances[t-1],
                transition_offset = transition_offset,
                observation = observation
            )
        )


f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(Time, AccX_Value, label="Input AccX")
#xarr[0].plot(Time, filtered_state_means[:, 2], "r-", label="Estimated AccX")
axarr[0].set_title('Acceleration X')
axarr[0].grid()
axarr[0].legend()
axarr[0].set_ylim([-0.3, 0.3])

axarr[1].plot(Time, RefVelX, label="Reference VelX")
axarr[1].plot(Time, filtered_state_means[:, 2], "r-", label="Estimated VelX")
axarr[1].set_title('Velocity X')
axarr[1].grid()
axarr[1].legend()
axarr[1].set_ylim([-0.6, 0.4])

axarr[2].plot(Time, RefPosX, label="Reference PosX")
axarr[2].plot(Time, filtered_state_means[:, 1], "r-", label="Estimated PosX")
axarr[2].set_title('Position X')
axarr[2].grid()
axarr[2].legend()
axarr[2].set_ylim([-2, 2])

plt.show()
