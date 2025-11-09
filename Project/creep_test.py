import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from mpl_toolkits.mplot3d import Axes3D
from pysindy.feature_library import CustomLibrary, PolynomialLibrary
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

# ============================================================================
# CREEP MODEL: Strain, Stress, and Damage Coupling for Alloy 617
# ============================================================================


class CreepModel:
    """
    Coupled creep model incorporating:
    - Strain evolution (primary, secondary, tertiary)
    - Stress redistribution (multiaxial loading)
    - Damage accumulation (cavitation, grain boundary cracking)
    """

    def __init__(self, temp=950, sigma_0=75, E=170000):
        """
        Parameters:
        - temp: Temperature (°C)
        - sigma_0: Threshold stress (MPa) - 75 MPa at 750°C, 0 at 950°C
        - E: Young's modulus (MPa)
        """
        self.temp = temp + 273.15  # Convert to Kelvin
        self.sigma_0 = sigma_0 if temp < 900 else 0
        self.E = E
        self.R = 8.314  # Gas constant J/(mol·K)
        self.Q = 300000  # Activation energy (J/mol)

    def creep_damage_ode(self, t, y, stress_applied):
        """
        Coupled ODEs for strain, stress, and damage

        State vector y = [strain, effective_stress, damage]
        """
        strain, stress_eff, damage = y

        # Material constants for Alloy 617
        A = 5e-15  # Norton constant (adjusted for better behavior)
        n = 3.0  # Stress exponent (reduced for stability)

        # Damage evolution parameters
        k_damage = 0.0001  # Damage rate constant (reduced)
        alpha = 2.0  # Damage stress sensitivity

        # Effective stress accounting for damage
        stress_actual = stress_eff * (1 - damage)
        sigma_eff = max(stress_actual - self.sigma_0, 1.0)

        # Strain rate (Norton-Bailey with damage)
        # Primary + Secondary + Tertiary creep
        primary_rate = 1.0 / (1 + t + 1e-6)  # Decaying primary

        secondary_rate = A * (sigma_eff**n) * np.exp(-self.Q / (self.R * self.temp))

        tertiary_factor = 1 + 2 * damage  # Linear damage effect
        tertiary_rate = secondary_rate * tertiary_factor

        d_strain = primary_rate + tertiary_rate

        # Stress evolution (minimal relaxation)
        d_stress = -0.1 * d_strain

        # Damage evolution (simplified)
        damage_rate = k_damage * (sigma_eff / 100) ** alpha
        d_damage = damage_rate

        return [d_strain, d_stress, d_damage]


# ============================================================================
# GENERATE SYNTHETIC CREEP DATA
# ============================================================================


def generate_creep_dataset(stress_levels, temp=950, time_max=1000, n_points=500):
    """
    Generate synthetic creep data for multiple stress levels
    """
    model = CreepModel(temp=temp)
    datasets = []

    for stress in stress_levels:
        t_span = (0, time_max)
        t_eval = np.linspace(0, time_max, n_points)

        # Initial conditions: [strain, stress, damage]
        y0 = [0.0, stress, 0.0]

        # Solve ODE system
        sol = solve_ivp(lambda t, y: model.creep_damage_ode(t, y, stress), t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-6)

        df = pd.DataFrame({"time": sol.t, "strain": sol.y[0], "stress": sol.y[1], "damage": sol.y[2], "applied_stress": stress, "temperature": temp})

        # Calculate strain rate
        df["strain_rate"] = np.gradient(df["strain"], df["time"])

        # Calculate damage rate
        df["damage_rate"] = np.gradient(df["damage"], df["time"])

        datasets.append(df)

    return pd.concat(datasets, ignore_index=True)


# Generate data for multiple stress conditions
stress_levels = [100, 150, 200, 250]  # MPa
temperature = 950  # °C

print("Generating synthetic creep data...")
creep_data = generate_creep_dataset(stress_levels, temp=temperature)

# ============================================================================
# PYSINDY MODEL FITTING
# ============================================================================

print("\nFitting PySINDy models...\n")

# Select one stress condition for detailed analysis
sample_data = creep_data[creep_data["applied_stress"] == 200].copy()
t = sample_data["time"].values
dt = t[1] - t[0]

# State variables: [strain, stress, damage]
X = sample_data[["strain", "stress", "damage"]].values

# Use standard polynomial library with lower degree
poly_lib = PolynomialLibrary(degree=2, include_bias=True)

# Initialize PySINDy model with STLSQ - very low threshold
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.001, alpha=0.001),
    feature_library=poly_lib,
    differentiation_method=ps.FiniteDifference(order=2),
)

# Fit the model
print("Training PySINDy model on strain-stress-damage dynamics...")
print(f"Data shapes - X: {X.shape}, time points: {len(t)}")
print(f"Data ranges - Strain: [{X[:, 0].min():.6f}, {X[:, 0].max():.6f}]")
print(f"             Stress: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
print(f"             Damage: [{X[:, 2].min():.6f}, {X[:, 2].max():.6f}]")
model.fit(X, t=dt, feature_names=["strain", "stress", "damage"])

print("\n" + "=" * 70)
print("DISCOVERED GOVERNING EQUATIONS:")
print("=" * 70)
model.print()
print("=" * 70 + "\n")

# Simulate using discovered model
X_sim = model.simulate(X[0], t)

# Denormalize for comparison
# X_sim = X_sim_normalized * X_std + X_mean

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(16, 12))

# Plot 1: Strain evolution
ax1 = plt.subplot(3, 3, 1)
ax1.plot(t, X[:, 0], "b-", linewidth=2, label="True Data")
ax1.plot(t, X_sim[:, 0], "r--", linewidth=2, label="PySINDy")
ax1.set_xlabel("Time (hours)", fontsize=11)
ax1.set_ylabel("Strain", fontsize=11)
ax1.set_title("Strain Evolution", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Stress evolution
ax2 = plt.subplot(3, 3, 2)
ax2.plot(t, X[:, 1], "b-", linewidth=2, label="True Data")
ax2.plot(t, X_sim[:, 1], "r--", linewidth=2, label="PySINDy")
ax2.set_xlabel("Time (hours)", fontsize=11)
ax2.set_ylabel("Stress (MPa)", fontsize=11)
ax2.set_title("Stress Relaxation", fontsize=12, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Damage evolution
ax3 = plt.subplot(3, 3, 3)
ax3.plot(t, X[:, 2], "b-", linewidth=2, label="True Data")
ax3.plot(t, X_sim[:, 2], "r--", linewidth=2, label="PySINDy")
ax3.set_xlabel("Time (hours)", fontsize=11)
ax3.set_ylabel("Damage", fontsize=11)
ax3.set_title("Damage Accumulation", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Strain rate
ax4 = plt.subplot(3, 3, 4)
strain_rate_true = np.gradient(X[:, 0], dt)
strain_rate_sim = np.gradient(X_sim[:, 0], dt)
ax4.plot(t, strain_rate_true, "b-", linewidth=2, label="True Data")
ax4.plot(t, strain_rate_sim, "r--", linewidth=2, label="PySINDy")
ax4.set_xlabel("Time (hours)", fontsize=11)
ax4.set_ylabel("Strain Rate (1/hr)", fontsize=11)
ax4.set_title("Strain Rate (dε/dt)", fontsize=12, fontweight="bold")
ax4.set_yscale("log")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Damage rate
ax5 = plt.subplot(3, 3, 5)
damage_rate_true = np.gradient(X[:, 2], dt)
damage_rate_sim = np.gradient(X_sim[:, 2], dt)
ax5.plot(t, damage_rate_true, "b-", linewidth=2, label="True Data")
ax5.plot(t, damage_rate_sim, "r--", linewidth=2, label="PySINDy")
ax5.set_xlabel("Time (hours)", fontsize=11)
ax5.set_ylabel("Damage Rate (1/hr)", fontsize=11)
ax5.set_title("Damage Rate (dD/dt)", fontsize=12, fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Strain vs Damage (coupling)
ax6 = plt.subplot(3, 3, 6)
ax6.plot(X[:, 0], X[:, 2], "b-", linewidth=2, label="True Data")
ax6.plot(X_sim[:, 0], X_sim[:, 2], "r--", linewidth=2, label="PySINDy")
ax6.set_xlabel("Strain", fontsize=11)
ax6.set_ylabel("Damage", fontsize=11)
ax6.set_title("Strain-Damage Coupling", fontsize=12, fontweight="bold")
ax6.legend(loc="best")
ax6.grid(True, alpha=0.3)

# Plot 7: 3D Phase Space
ax7 = fig.add_subplot(3, 3, 7, projection="3d")
ax7.plot(X[:, 0], X[:, 1], X[:, 2], "b-", linewidth=2, label="True")
ax7.plot(X_sim[:, 0], X_sim[:, 1], X_sim[:, 2], "r--", linewidth=2, label="PySINDy")
ax7.set_xlabel("Strain", fontsize=10)
ax7.set_ylabel("Stress (MPa)", fontsize=10)
ax7.set_zlabel("Damage", fontsize=10)
ax7.set_title("3D Phase Space", fontsize=12, fontweight="bold")
ax7.legend(loc="best")

# Plot 8: Multi-stress comparison (Strain)
ax8 = plt.subplot(3, 3, 8)
for stress in stress_levels:
    data_subset = creep_data[creep_data["applied_stress"] == stress]
    ax8.plot(data_subset["time"], data_subset["strain"], linewidth=2, label=f"{stress} MPa")
ax8.set_xlabel("Time (hours)", fontsize=11)
ax8.set_ylabel("Strain", fontsize=11)
ax8.set_title("Multi-Stress Strain Evolution", fontsize=12, fontweight="bold")
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Multi-stress comparison (Damage)
ax9 = plt.subplot(3, 3, 9)
for stress in stress_levels:
    data_subset = creep_data[creep_data["applied_stress"] == stress]
    ax9.plot(data_subset["time"], data_subset["damage"], linewidth=2, label=f"{stress} MPa")
ax9.set_xlabel("Time (hours)", fontsize=11)
ax9.set_ylabel("Damage", fontsize=11)
ax9.set_title("Multi-Stress Damage Evolution", fontsize=12, fontweight="bold")
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# MODEL PERFORMANCE METRICS
# ============================================================================

r2_strain = r2_score(X[:, 0], X_sim[:, 0])
r2_stress = r2_score(X[:, 1], X_sim[:, 1])
r2_damage = r2_score(X[:, 2], X_sim[:, 2])

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"R² Score (Strain):  {r2_strain:.6f}")
print(f"R² Score (Stress):  {r2_stress:.6f}")
print(f"R² Score (Damage):  {r2_damage:.6f}")
print(f"Average R²:         {np.mean([r2_strain, r2_stress, r2_damage]):.6f}")
print("=" * 70 + "\n")

# ============================================================================
# ANALYSIS: MONKMAN-GRANT & LARSON-MILLER
# ============================================================================

print("=" * 70)
print("CREEP LIFE PREDICTION ANALYSIS")
print("=" * 70)

for stress in stress_levels:
    data_subset = creep_data[creep_data["applied_stress"] == stress]

    # Minimum creep rate
    min_creep_rate = data_subset["strain_rate"].min()

    # Time to 5% strain (rupture criterion from paper)
    rupture_idx = data_subset[data_subset["strain"] >= 0.05].index
    if len(rupture_idx) > 0:
        t_rupture = data_subset.loc[rupture_idx[0], "time"]

        # Monkman-Grant constant
        mg_constant = t_rupture * min_creep_rate

        # Larson-Miller Parameter: LMP = T(C + log(t))
        T = temperature + 273.15  # Kelvin
        C = 20  # Typical constant for nickel alloys
        LMP = T * (C + np.log10(t_rupture))

        print(f"\n{stress} MPa:")
        print(f"  Min Creep Rate:      {min_creep_rate:.6e} /hr")
        print(f"  Time to Rupture:     {t_rupture:.2f} hours")
        print(f"  Monkman-Grant (C):   {mg_constant:.6e}")
        print(f"  Larson-Miller (LMP): {LMP:.2f}")

print("\n" + "=" * 70 + "\n")

# Export results
print("Exporting results to CSV...")
creep_data.to_csv("alloy617_creep_data.csv", index=False)
print("Data saved to: alloy617_creep_data.csv")
print("\nAnalysis complete!")
