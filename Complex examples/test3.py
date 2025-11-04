import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt


# x(t) = x0 * exp(-2*t) | x0 = 3
# y(t) = y0 * exp(t) | y0 = 0.5

t = np.linspace(0, 1, 100)

x = 3 * np.exp(-2 * t)
y= 0.5 * np.exp(t)

x = np.stack((x,y), axis=-1)

model = ps.SINDy()
model.fit(x, t=t, feature_names=["x", "y"])
model.print()

