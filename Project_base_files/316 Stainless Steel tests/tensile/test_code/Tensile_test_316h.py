import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ============================================================================
# TENSILE TEST MODEL: Stress-Strain Behavior for 316H Stainless Steel
# ============================================================================


class TensileTestModel316H:
    """
    Tensile test model for 316H Stainless Steel incorporating:
    - Elastic region (Hooke's Law)
    - Yield transition (0.2% offset yield strength)
    - Plastic hardening (power law and Voce hardening)
    - Necking and failure
    """

    def __init__(self, temp=25, strain_rate=1e-3):
        """
        Parameters:
        - temp: Temperature (°C)
        - strain_rate: Strain rate (1/s)
        """
        self.temp = temp
        self.strain_rate = strain_rate

        # Temperature-dependent properties for 316H Stainless Steel
        if temp <= 100:
            self.E = 195000  # Young's modulus (MPa) at room temp
            self.yield_stress = 240  # Yield strength (MPa)
            self.K = 1050  # Strength coefficient (MPa)
            self.n = 0.45  # Strain hardening exponent
            self.UTS = 580  # Ultimate tensile strength (MPa)
        elif temp <= 400:
            self.E = 185000
            self.yield_stress = 195
            self.K = 950
            self.n = 0.42
            self.UTS = 520
        elif temp <= 600:
            self.E = 170000
            self.yield_stress = 160
            self.K = 800
            self.n = 0.38
            self.UTS = 450
        else:  # High temperature (650-800°C)
            self.E = 155000
            self.yield_stress = 130
            self.K = 650
            self.n = 0.32
            self.UTS = 350

        # Calculate yield strain
        self.yield_strain = self.yield_stress / self.E

        # Necking parameters
        self.necking_strain = self.n  # Necking occurs at strain = n (Considère criterion)
        self.fracture_strain = self.necking_strain + 0.20  # Additional elongation in neck

    def stress_strain_response(self, strain):
        """
        Complete stress-strain curve with elastic, plastic, and necking regions
        """
        stress = np.zeros_like(strain)

        for i, eps in enumerate(strain):
            if eps <= self.yield_strain:
                # Elastic region (Hooke's Law)
                stress[i] = self.E * eps

            elif eps <= self.necking_strain:
                # Plastic region (Hollomon power law with smooth transition)
                eps_plastic = eps - self.yield_strain
                stress[i] = self.K * (eps_plastic + self.yield_strain) ** self.n

            elif eps <= self.fracture_strain:
                # Necking region (stress decreases)
                progress = (eps - self.necking_strain) / (
                    self.fracture_strain - self.necking_strain
                )
                failure_stress = 0.65 * self.UTS  # Stress at fracture
                stress[i] = self.UTS - (self.UTS - failure_stress) * progress

            else:
                # Fracture
                stress[i] = 0

        return stress

    def add_noise(self, data, noise_level=0.02):
        """Add realistic measurement noise"""
        noise = np.random.normal(0, noise_level * np.std(data), len(data))
        return data + noise


# ============================================================================
# GENERATE SYNTHETIC TENSILE TEST DATA
# ============================================================================


def generate_tensile_dataset(temperatures, strain_rate=1e-3, n_points=1000):
    """
    Generate synthetic tensile test data for multiple temperatures
    """
    datasets = []

    for temp in temperatures:
        model = TensileTestModel316H(temp=temp, strain_rate=strain_rate)

        # Generate strain array up to fracture
        strain = np.linspace(0, model.fracture_strain, n_points)

        # Calculate stress response
        stress = model.stress_strain_response(strain)

        # Add realistic noise
        stress_noisy = model.add_noise(stress, noise_level=0.01)

        # Create dataframe
        df = pd.DataFrame(
            {
                "strain": strain,
                "stress": stress_noisy,
                "true_stress": stress,
                "temperature": temp,
                "strain_rate": strain_rate,
            }
        )

        # Calculate engineering properties
        df["elastic_modulus"] = np.gradient(df["stress"], df["strain"])

        # Calculate true stress and true strain (for plastic region)
        df["true_strain"] = np.log(1 + df["strain"])
        df["true_stress_corrected"] = df["stress"] * (1 + df["strain"])

        datasets.append(df)

    return pd.concat(datasets, ignore_index=True)


# Generate data for multiple temperatures
temperatures = [25, 300, 550, 750]  # °C
strain_rate = 1e-3  # 1/s (quasi-static)

print("Generating synthetic tensile test data for 316H Stainless Steel...")
tensile_data = generate_tensile_dataset(temperatures, strain_rate=strain_rate)

# ============================================================================
# MATERIALS PROPERTY EXTRACTION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTED MECHANICAL PROPERTIES - 316H STAINLESS STEEL")
print("=" * 70)

properties_summary = []

for temp in temperatures:
    data_subset = tensile_data[tensile_data["temperature"] == temp].copy()

    # Young's Modulus (initial slope)
    elastic_region = data_subset[data_subset["strain"] < 0.005]
    E_measured = np.polyfit(elastic_region["strain"], elastic_region["stress"], 1)[0]

    # Yield Strength (0.2% offset)
    offset_strain = 0.002
    offset_line = E_measured * (data_subset["strain"] - offset_strain)
    yield_idx = np.where(data_subset["stress"] > offset_line)[0]
    if len(yield_idx) > 0:
        yield_stress = data_subset.iloc[yield_idx[0]]["stress"]
        yield_strain = data_subset.iloc[yield_idx[0]]["strain"]
    else:
        yield_stress = np.nan
        yield_strain = np.nan

    # Ultimate Tensile Strength
    UTS = data_subset["stress"].max()
    UTS_strain = data_subset.loc[data_subset["stress"].idxmax(), "strain"]

    # Fracture properties
    fracture_strain = data_subset["strain"].max()
    fracture_stress = data_subset.iloc[-1]["stress"]

    # Ductility metrics
    elongation = fracture_strain * 100  # Percent elongation

    # Toughness (area under stress-strain curve)
    toughness = np.trapz(data_subset["stress"], data_subset["strain"])

    properties_summary.append(
        {
            "Temperature (°C)": temp,
            "Young's Modulus (GPa)": E_measured / 1000,
            "Yield Strength (MPa)": yield_stress,
            "UTS (MPa)": UTS,
            "Fracture Strain": fracture_strain,
            "Elongation (%)": elongation,
            "Toughness (MJ/m³)": toughness,
        }
    )

    print(f"\nTemperature: {temp}°C")
    print(f"  Young's Modulus:    {E_measured/1000:.1f} GPa")
    print(f"  Yield Strength:     {yield_stress:.1f} MPa")
    print(f"  UTS:                {UTS:.1f} MPa")
    print(f"  Fracture Strain:    {fracture_strain:.3f}")
    print(f"  Elongation:         {elongation:.1f}%")
    print(f"  Toughness:          {toughness:.2f} MJ/m³")

print("=" * 70 + "\n")

# ============================================================================
# CONSTITUTIVE MODEL FITTING (POWER LAW HARDENING)
# ============================================================================

print("=" * 70)
print("POWER LAW HARDENING MODEL FITTING")
print("=" * 70)


def power_law_hardening(strain, K, n):
    """Hollomon equation: σ = K * ε^n"""
    return K * strain**n


for temp in temperatures:
    data_subset = tensile_data[tensile_data["temperature"] == temp].copy()

    # Fit only plastic region (after yield, before necking)
    plastic_region = data_subset[
        (data_subset["strain"] > 0.005)
        & (data_subset["strain"] < data_subset["stress"].idxmax() / len(data_subset))
    ]

    if len(plastic_region) > 10:
        try:
            popt, _ = curve_fit(
                power_law_hardening,
                plastic_region["strain"],
                plastic_region["stress"],
                p0=[1000, 0.3],
                maxfev=5000,
            )
            K_fit, n_fit = popt

            # Calculate R² for fit quality
            stress_pred = power_law_hardening(plastic_region["strain"], K_fit, n_fit)
            r2 = r2_score(plastic_region["stress"], stress_pred)

            print(f"\n{temp}°C: σ = {K_fit:.1f} * ε^{n_fit:.3f}  (R² = {r2:.4f})")
        except:
            print(f"\n{temp}°C: Fitting failed")

print("\n" + "=" * 70 + "\n")

# ============================================================================
# PYSINDY MODEL FITTING
# ============================================================================

print("=" * 70)
print("PYSINDY CONSTITUTIVE MODEL DISCOVERY - 316H STAINLESS STEEL")
print("=" * 70)

# Use room temperature data for SINDy analysis
sample_data = tensile_data[tensile_data["temperature"] == 25].copy().reset_index(drop=True)

# Prepare data: stress as state variable, strain as time
X = sample_data["stress"].values.reshape(-1, 1)
t = sample_data["strain"].values

# Ensure strictly increasing time
mask = np.concatenate([[True], np.diff(t) > 0])
X = X[mask]
t = t[mask]

# Initialize SINDy with polynomial library
poly_lib = PolynomialLibrary(degree=3, include_bias=True)

model_sindy = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.5, alpha=0.01),
    feature_library=poly_lib,
    differentiation_method=ps.FiniteDifference(order=2),
)

print("\nFitting SINDy to stress-strain relationship...")
print("Modeling: dσ/dε = f(σ)")

try:
    model_sindy.fit(X, t=t, feature_names=["stress"])

    print("\nDiscovered equation for dσ/dε:")
    print("=" * 70)
    model_sindy.print()
    print("=" * 70)

    # Get the coefficients
    coefficients = model_sindy.coefficients()
    feature_names = model_sindy.get_feature_names()

    print("\nCoefficients:")
    for i, (coef, name) in enumerate(zip(coefficients[0], feature_names)):
        if abs(coef) > 0.01:
            print(f"  {name}: {coef:.4f}")

except Exception as e:
    print(f"\nSINDy fitting encountered an issue: {e}")
    print("This is common with material stress-strain data due to complex nonlinearity.")

print("\n" + "=" * 70 + "\n")

# Alternative: Fit SINDy to plastic region only
print("=" * 70)
print("ALTERNATIVE: PYSINDY ON PLASTIC REGION ONLY")
print("=" * 70)

# Focus on plastic region where behavior is more regular
plastic_data = (
    sample_data[
        (sample_data["strain"] > 0.01)
        & (sample_data["strain"] < sample_data["strain"].quantile(0.8))
    ]
    .copy()
    .reset_index(drop=True)
)

if len(plastic_data) > 50:
    X_plastic = plastic_data["stress"].values.reshape(-1, 1)
    t_plastic = plastic_data["strain"].values

    # Ensure strictly increasing
    mask = np.concatenate([[True], np.diff(t_plastic) > 0])
    X_plastic = X_plastic[mask]
    t_plastic = t_plastic[mask]

    # Use simpler polynomial library for plastic region
    poly_lib_plastic = PolynomialLibrary(degree=2, include_bias=True)

    model_plastic = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.1, alpha=0.001),
        feature_library=poly_lib_plastic,
        differentiation_method=ps.FiniteDifference(order=2),
    )

    print("\nFitting SINDy to plastic deformation region...")
    print("Modeling: dσ/dε = f(σ) for plastic strain")

    try:
        model_plastic.fit(X_plastic, t=t_plastic, feature_names=["stress"])

        print("\nDiscovered equation for plastic region:")
        print("=" * 70)
        model_plastic.print()
        print("=" * 70)

        # Evaluate model quality
        X_pred = model_plastic.simulate(X_plastic[0], t_plastic)
        r2 = r2_score(X_plastic, X_pred)
        print(f"\nModel R² score: {r2:.4f}")

    except Exception as e:
        print(f"\nPlastic region SINDy fitting issue: {e}")

print("\n" + "=" * 70 + "\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('316H Stainless Steel - Tensile Test Analysis', fontsize=16, fontweight='bold')

# Plot 1: Stress-Strain curves at different temperatures
ax1 = axes[0, 0]
for temp in temperatures:
    data_temp = tensile_data[tensile_data["temperature"] == temp]
    ax1.plot(data_temp["strain"], data_temp["stress"], label=f'{temp}°C', linewidth=2)
ax1.set_xlabel('Engineering Strain', fontsize=11)
ax1.set_ylabel('Engineering Stress (MPa)', fontsize=11)
ax1.set_title('Stress-Strain Curves', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Temperature dependence of properties
ax2 = axes[0, 1]
props_df = pd.DataFrame(properties_summary)
ax2_twin = ax2.twinx()
ax2.plot(props_df["Temperature (°C)"], props_df["Yield Strength (MPa)"], 
         'o-', label='Yield Strength', color='blue', linewidth=2, markersize=8)
ax2.plot(props_df["Temperature (°C)"], props_df["UTS (MPa)"], 
         's-', label='UTS', color='red', linewidth=2, markersize=8)
ax2_twin.plot(props_df["Temperature (°C)"], props_df["Young's Modulus (GPa)"], 
              '^-', label="Young's Modulus", color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Temperature (°C)', fontsize=11)
ax2.set_ylabel('Strength (MPa)', fontsize=11)
ax2_twin.set_ylabel("Young's Modulus (GPa)", fontsize=11)
ax2.set_title('Temperature Dependence', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2_twin.legend(loc='center right')
ax2.grid(True, alpha=0.3)

# Plot 3: Ductility vs Temperature
ax3 = axes[1, 0]
ax3.plot(props_df["Temperature (°C)"], props_df["Elongation (%)"], 
         'o-', color='purple', linewidth=2, markersize=8)
ax3.set_xlabel('Temperature (°C)', fontsize=11)
ax3.set_ylabel('Elongation (%)', fontsize=11)
ax3.set_title('Ductility vs Temperature', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Toughness vs Temperature
ax4 = axes[1, 1]
ax4.plot(props_df["Temperature (°C)"], props_df["Toughness (MJ/m³)"], 
         'o-', color='orange', linewidth=2, markersize=8)
ax4.set_xlabel('Temperature (°C)', fontsize=11)
ax4.set_ylabel('Toughness (MJ/m³)', fontsize=11)
ax4.set_title('Toughness vs Temperature', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
