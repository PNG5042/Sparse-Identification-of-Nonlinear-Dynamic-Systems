import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
g = 9.81
k = 0.02

# Dynamics
def ball_dynamics(state, t):
    y, v = state
    dydt = v
    dvdt = -g - k * v * abs(v)
    return [dydt, dvdt]

# Time grid
t = np.linspace(0, 5, 500)
y0 = [50, 0]
sol = odeint(ball_dynamics, y0, t)
y, v = sol[:, 0], sol[:, 1]

# Combine and add small noise
X = np.vstack((y, v)).T
X_noisy = X + 0.001 * np.random.randn(*X.shape)

# Custom library for the drag term
def v_abs_v(x):
    # x is a 1D array representing one variable (velocity in this case)
    # Return v * |v|
    return x * np.abs(x)

# Combine polynomial library with custom library
poly_library = ps.PolynomialLibrary(degree=1, include_bias=True)
custom_library = ps.CustomLibrary(
    library_functions=[v_abs_v],
    function_names=[lambda x: "v|v|"]
)
feature_library = poly_library + custom_library

# Fit SINDy model with lower threshold
model = ps.SINDy(
    feature_library=feature_library,
    optimizer=ps.STLSQ(threshold=0.01)  # Lower threshold for better fitting
)
model.fit(X_noisy, t=t)
model.print()

# Plot comparison
X_pred = model.simulate(X_noisy[0], t)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, X[:, 0], 'b-', label='True', linewidth=2)
plt.plot(t, X_pred[:, 0], 'r--', label='SINDy', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, X[:, 1], 'b-', label='True', linewidth=2)
plt.plot(t, X_pred[:, 1], 'r--', label='SINDy', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()