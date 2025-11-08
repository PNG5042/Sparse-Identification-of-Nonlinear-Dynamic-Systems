import numpy as np
import pandas as pd

# giải hệ phương trình vi phân ODE
from scipy.integrate import solve_ivp

# vẽ đồ thị 2D và 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# dùng Sparse Identification of Nonlinear Dynamical Systems để tìm các phương trình chi phối dữ liệu
import pysindy as ps

# tạo các đặc trưng đa thức để SINDy xây dựng mô hình
from pysindy.feature_library import PolynomialLibrary
from sklearn.metrics import r2_score

# ====================================================================
# 1. Define the coupled creep ODE system
# ====================================================================
class Alloy617Creep:
    def __init__(self, temp=950, sigma_0=75, E=170000):
        """
        temp: Temperature in Celsius
        sigma_0: Threshold stress (MPa)
        E: Young's modulus (MPa)
        """
        self.temp = temp + 273.15  # Convert to Kelvin
        self.sigma_0 = sigma_0 if temp < 900 else 0
        self.E = E
        self.R = 8.314  # Gas constant J/mol·K
        self.Q = 300000  # Activation energy (J/mol)

    def ode_system(self, t, y, stress_applied):
        """
        y = [strain, effective_stress, damage]
        """
        strain, stress_eff, damage = y

        # Norton creep constants (simplified)
        A = 5e-15
        n = 3.0

        # Damage parameters
        k_damage = 1e-4
        alpha = 2.0

        # Effective stress considering damage
        stress_actual = stress_eff * (1 - damage)
        sigma_eff = max(stress_actual - self.sigma_0, 1.0)

        # Strain rate: primary + secondary + tertiary creep
        primary_rate = 1.0 / (1 + t + 1e-6)
        secondary_rate = A * (sigma_eff ** n) * np.exp(-self.Q / (self.R * self.temp))
        tertiary_factor = 1 + 2 * damage
        d_strain = primary_rate + secondary_rate * tertiary_factor

        # Stress evolution (minimal relaxation)
        d_stress = -0.1 * d_strain

        # Damage evolution
        d_damage = k_damage * (sigma_eff / 100) ** alpha

        return [d_strain, d_stress, d_damage]

# ====================================================================
# 2. Generate synthetic creep dataset
# ====================================================================
def generate_creep_data(stress_levels, temp=950, t_max=1000, n_points=500):
    model = Alloy617Creep(temp=temp)
    all_data = []

    for stress in stress_levels:
        y0 = [0.0, stress, 0.0]  # initial strain, stress, damage
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, n_points)

        sol = solve_ivp(
            lambda t, y: model.ode_system(t, y, stress),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6
        )

        df = pd.DataFrame({
            'time': sol.t,
            'strain': sol.y[0],
            'stress': sol.y[1],
            'damage': sol.y[2],
            'applied_stress': stress
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

# Generate data
stress_levels = [100, 150, 200]
creep_data = generate_creep_data(stress_levels, temp=950)
print("Synthetic creep data generated.")

# ====================================================================
# 3. Fit SINDy model
# ====================================================================
# Select data for a single stress level (200 MPa)
sample_data = creep_data[creep_data['applied_stress'] == 200]
X = sample_data[['strain', 'stress', 'damage']].values
t = sample_data['time'].values
dt = t[1] - t[0]

# Polynomial library (degree 2)
poly_lib = PolynomialLibrary(degree=2, include_bias=True)

# SINDy model
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.001),
    feature_library=poly_lib,
    differentiation_method=ps.FiniteDifference(order=2)
)

# Fit model
model.fit(X, t=dt, feature_names=['strain', 'stress', 'damage'])
print("\nDiscovered governing equations:")
model.print()

# Simulate SINDy model
X_sim = model.simulate(X[0], t)

# ====================================================================
# 4. Visualization
# ====================================================================
plt.figure(figsize=(12, 8))

# Strain
plt.subplot(2, 2, 1)
plt.plot(t, X[:, 0], 'b', label='True')
plt.plot(t, X_sim[:, 0], 'r--', label='SINDy')
plt.xlabel('Time')
plt.ylabel('Strain')
plt.title('Strain Evolution')
plt.legend()
plt.grid(True)

# Stress
plt.subplot(2, 2, 2)
plt.plot(t, X[:, 1], 'b', label='True')
plt.plot(t, X_sim[:, 1], 'r--', label='SINDy')
plt.xlabel('Time')
plt.ylabel('Stress')
plt.title('Stress Evolution')
plt.legend()
plt.grid(True)

# Damage
plt.subplot(2, 2, 3)
plt.plot(t, X[:, 2], 'b', label='True')
plt.plot(t, X_sim[:, 2], 'r--', label='SINDy')
plt.xlabel('Time')
plt.ylabel('Damage')
plt.title('Damage Evolution')
plt.legend()
plt.grid(True)

# 3D Phase plot
ax = plt.subplot(2, 2, 4, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b', label='True')
ax.plot(X_sim[:, 0], X_sim[:, 1], X_sim[:, 2], 'r--', label='SINDy')
ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.set_zlabel('Damage')
ax.set_title('3D Phase Space')
ax.legend()

plt.tight_layout()
plt.show()

# ====================================================================
# 5. Evaluate Model
# ====================================================================
r2_strain = r2_score(X[:, 0], X_sim[:, 0])
r2_stress = r2_score(X[:, 1], X_sim[:, 1])
r2_damage = r2_score(X[:, 2], X_sim[:, 2])
print(f"\nR² Scores:\n Strain: {r2_strain:.4f}\n Stress: {r2_stress:.4f}\n Damage: {r2_damage:.4f}")
