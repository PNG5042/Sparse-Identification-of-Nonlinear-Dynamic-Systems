import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Damped harmonic oscillator

# State vector: z = [x, v] where v = ẋ

# ż₁ = v                           (dx/dt = v)
# ż₂ = -ωₙ²·x - 2ζωₙ·v            (dv/dt from original equation)

zeta = 0.1          #damping Ratio
omega_n = 2*np.pi   #natural frequency (1 Hz)

def damped_oscillator(t, z):
    x, v = z
    dxdt = v
    dvdt = -omega_n**2 * x - 2*zeta*omega_n * v
    return[dxdt, dvdt]

# Initital condidtions
z0 = [1.0, 0.0]     #start at x = 1, v = 0

# time points
t_span = (0,5)
t_eval = np.linspace(0, 5, 500) # you can increase the amount of data points

# solve
sol = solve_ivp(damped_oscillator, t_span, z0, t_eval=t_eval, method='RK45')

# extract the solution
x = sol.y[0]  # position
v = sol.y[1]  # velocity
t = sol.t     # time points


# ----
# Choose library - polynomial up to degree 2 should work
library = ps.PolynomialLibrary(degree=2)
# This creates: [1, x, v, x², xv, v²]
# Degree 0: [1]                    # constant term
# Degree 1: [x, v]                 # linear terms
# Degree 2: [x², xv, v²]           # quadratic terms

# Optimizer with sparsity threshold
optimizer = ps.STLSQ(threshold=0.01) # You can increase for more accture model
# STLSQ = Sequential Thresholded Least Squares
# This is the sparse regression optimizer that finds which terms are actually needed.
# How it works:

# Fits all 6 candidate functions using least squares
# Removes coefficients smaller than 0.01 (threshold)
# Re-fits with remaining terms
# Repeats until converged

# Result: Only keeps important terms, giving you sparse, interpretable equations!

#build model
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
    differentiation_method=ps.FiniteDifference(order=2))

    # Higher order = smaller errors in derivatives = better chance 
    # SINDy discovers the correct equations.

    # order=1  →  max error ≈ 0.01      (1%)
    # order=2  →  max error ≈ 0.0001    (0.01%)
    # order=4  →  max error ≈ 0.00000001 (0.000001%)

# pass both x and it's derivatives:
X = np.column_stack([x,v])
x_dot = np.column_stack([v, -omega_n**2 * x - 2*zeta*omega_n * v])

model.fit(X, t=t) # SINDy will use FiniteDifference internally
model.print()


# extract Coefficients
# ωₙ² = 39.48
# 2ζωₙ = 1.26

# Simulate with discovered model
X_sim = model.simulate(z0,t=t)

# plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(t, X[:, 0], 'b-', label='True x')
plt.plot(t, X_sim[:, 0], 'r--', label='SINDy x')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.subplot(1,2,2)
plt.plot(t, X[:, 1], 'b-', label='True v')
plt.plot(t, X_sim[:, 1], 'r--', label='SINDy v')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity')

plt.show()