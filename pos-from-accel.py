from pykalman import KalmanFilter
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt(fname="test-data-no-noise.txt", delimiter=",", skiprows=0)

# Data description
#  Time
#  AccX - acceleration signal
#  RefPosX - real position (ground truth)
#  RefVelX - real velocity (ground truth)

Time = Data[:, [0]]
AccX = Data[:, [1]]
RefPosX = Data[:, [2]]
RefVelX = Data[:, [3]]


plt.plot(AccX, RefPosX, "r-")
plt.grid()
plt.show()
