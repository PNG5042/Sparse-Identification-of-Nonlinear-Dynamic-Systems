import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pysindy as ps

# Physical parameters
g = 9.81  # gravity
L = 1.0   # pendulum length
b = 0.1   # damping coefficient

# Define the true dynamics
def pendulum(t, state):
    theta, omega = state
    dtheta = omega
    domega = -(g/L) * np.sin(theta) - b * omega
    return [dtheta, domega]

# Generate data for SMALL angle (linearized regime)
theta0_small = 0.2  # ~11 degrees
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
sol_small = solve_ivp(pendulum, t_span, [theta0_small, 0], t_eval=t_eval)

# Generate data for LARGE angle (nonlinear regime)
theta0_large = 2.5  # ~143 degrees
sol_large = solve_ivp(pendulum, t_span, [theta0_large, 0], t_eval=t_eval)

# Extract state variables
X_small = sol_small.y.T  # shape: (n_samples, 2) = [theta, omega]
X_large = sol_large.y.T

print("="*60)
print("SMALL ANGLE CASE (theta_0 = 0.2 rad ~= 11 degrees)")
print("="*60)

# Option A: Polynomial library (will find theta_dot_dot = -omega_0^2*theta - b*omega)
model_poly = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3, include_bias=False),
    optimizer=ps.STLSQ(threshold=0.01)
)

model_poly.fit(X_small, t=t_eval)
print("\nDiscovered Equations (Polynomial Library):")
model_poly.print()

# Check coefficients
print("\nCoefficients matrix:")
print(model_poly.coefficients())
print("\nExpected for small angle:")
print(f"  d(theta)/dt = omega")
print(f"  d(omega)/dt ~= -{g/L:.3f}*theta - {b:.3f}*omega")

print("\n" + "="*60)
print("LARGE ANGLE CASE (theta_0 = 2.5 rad ~= 143 degrees)")
print("="*60)

# Option B: Combined library with trig functions
poly_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
fourier_lib = ps.FourierLibrary(n_frequencies=1)  # includes sin, cos

model_trig = ps.SINDy(
    feature_library=poly_lib + fourier_lib,
    optimizer=ps.STLSQ(threshold=0.05)
)

model_trig.fit(X_large, t=t_eval)
print("\nDiscovered Equations (Polynomial + Fourier Library):")
model_trig.print()

# Check coefficients
print("\nCoefficients matrix:")
print(model_trig.coefficients())
print("\nExpected for large angle:")
print(f"  d(theta)/dt = omega")
print(f"  d(omega)/dt = -{g/L:.3f}*sin(theta) - {b:.3f}*omega")

# Simulate both models and compare
print("\n" + "="*60)
print("VALIDATION: Simulating Discovered Models")
print("="*60)

# Simulate small angle model
X_pred_small = model_poly.simulate(X_small[0], t_eval)
error_small = np.mean((X_small - X_pred_small)**2)
print(f"\nSmall angle MSE: {error_small:.6e}")

# Simulate large angle model
X_pred_large = model_trig.simulate(X_large[0], t_eval)
error_large = np.mean((X_large - X_pred_large)**2)
print(f"Large angle MSE: {error_large:.6e}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Small angle case
axes[0, 0].plot(t_eval, X_small[:, 0], 'b-', label='True', linewidth=2)
axes[0, 0].plot(t_eval, X_pred_small[:, 0], 'r--', label='SINDy', linewidth=2)
axes[0, 0].set_ylabel('theta (rad)', fontsize=12)
axes[0, 0].set_title('Small Angle: Position', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_eval, X_small[:, 1], 'b-', label='True', linewidth=2)
axes[1, 0].plot(t_eval, X_pred_small[:, 1], 'r--', label='SINDy', linewidth=2)
axes[1, 0].set_ylabel('omega (rad/s)', fontsize=12)
axes[1, 0].set_xlabel('Time (s)', fontsize=12)
axes[1, 0].set_title('Small Angle: Velocity', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Large angle case
axes[0, 1].plot(t_eval, X_large[:, 0], 'b-', label='True', linewidth=2)
axes[0, 1].plot(t_eval, X_pred_large[:, 0], 'r--', label='SINDy', linewidth=2)
axes[0, 1].set_ylabel('theta (rad)', fontsize=12)
axes[0, 1].set_title('Large Angle: Position', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(t_eval, X_large[:, 1], 'b-', label='True', linewidth=2)
axes[1, 1].plot(t_eval, X_pred_large[:, 1], 'r--', label='SINDy', linewidth=2)
axes[1, 1].set_ylabel('omega (rad/s)', fontsize=12)
axes[1, 1].set_xlabel('Time (s)', fontsize=12)
axes[1, 1].set_title('Large Angle: Velocity', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('pendulum_sindy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Phase portrait comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(X_small[:, 0], X_small[:, 1], 'b-', label='True', linewidth=2)
axes[0].plot(X_pred_small[:, 0], X_pred_small[:, 1], 'r--', label='SINDy', linewidth=2)
axes[0].set_xlabel('theta (rad)', fontsize=12)
axes[0].set_ylabel('omega (rad/s)', fontsize=12)
axes[0].set_title('Small Angle: Phase Portrait', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(X_large[:, 0], X_large[:, 1], 'b-', label='True', linewidth=2)
axes[1].plot(X_pred_large[:, 0], X_pred_large[:, 1], 'r--', label='SINDy', linewidth=2)
axes[1].set_xlabel('theta (rad)', fontsize=12)
axes[1].set_ylabel('omega (rad/s)', fontsize=12)
axes[1].set_title('Large Angle: Phase Portrait', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('pendulum_phase_portraits.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Analysis complete! Plots saved.")
print("="*60)