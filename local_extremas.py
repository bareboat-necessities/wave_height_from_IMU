import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

Data = np.loadtxt(fname="test-data.txt", delimiter=",", skiprows=0)

xs = Data[:, [1]]

df = pd.DataFrame(xs, columns=['data'])

n = 1000  # number of points to be checked before and after

# Find local peaks

df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                  order=n)[0]]['data']
df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                  order=n)[0]]['data']

# Plot results

plt.scatter(df.index, df['min'], c='r')
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['data'])
plt.show()
