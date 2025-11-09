
import numpy as np
import matplotlib.pyplot as plt

# Solution to ODE x(t) = x(0)e^(-0.5t) = E^(-0.5t)
# where x(0) = 1

#linspace() function in NumPy returns an array of evenly spaced numbers over a specified range. 

t = np.linspace(0, 10, 1000) # the start time to end time seconds. 100 is the amount of points 

#reshape the array to be a column vector (rows, columns) (-1 means unspecified number of rows, 1 means one column)
x = np.exp(-0.5 * t).reshape(-1, 1)

plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()

import pysindy as ps

# Build SINDy model
model = ps.SINDy()

model.fit(x, t=0.01) # t is the time step between measurements (the more frequent the measurements, the smaller t is)
model.print()