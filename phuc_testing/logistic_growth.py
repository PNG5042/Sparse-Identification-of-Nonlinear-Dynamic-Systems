import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

# Parameters
r = 1.0       # growth rate
K = 10.0      # carrying capacity
x0 = 0.5      # initial population

# Time vector
t = np.linspace(0, 5, 100)

# Logistic growth solution
x = K * x0 * np.exp(r*t) / (K + x0*(np.exp(r*t)-1))
X = x.reshape(-1, 1)  # shape (n_samples, n_features)

# Fit SINDy
model = ps.SINDy()
model.fit(X, t=t, feature_names=["x"])
model.print()

# Plot
plt.plot(t, x, label="Testing logistic growth")
x_sim = model.simulate(X[0], t)
plt.plot(t, x_sim, '--', label="SINDy prediction")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.show()
