import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint

# Define the logistic growth model
def logistic_growth(x, t):
    return x * (1 - x)

t = np.linspace(0, 10, 1000)  # Time vector
x0 = 0.1  # Initial condition

x = odeint(logistic_growth, x0, t)  # Integrate the ODE

plt.plot(t, x, label='True Logistic Growth')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.show()

# 5. Build SINDy model with polynomial library
poly_library = ps.PolynomialLibrary(degree=2)  # logistic is quadratic
model = ps.SINDy(feature_library=poly_library)

model.fit(x, t=0.01)  # t is the time step between measurements
model.print()

# Simulate the model
x_sim = model.simulate(x0, t)


plt.plot(t, x, 'b', label='True')
plt.plot(t, x_sim, 'r--', label='SINDy Prediction')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.show()