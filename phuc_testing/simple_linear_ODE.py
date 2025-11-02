import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

# Parameters
a = -2.0      # linear coefficient
x0 = 3.0      # initial condition
t = np.linspace(0, 5, 100)  # time vector

# Generate linear ODE data
x = x0 * np.exp(a * t)
X = x.reshape(-1, 1)  # SINDy expects (n_samples, n_features)

# Fit SINDy model
model = ps.SINDy()
model.fit(X, t=t, feature_names=["x"])
model.print()
