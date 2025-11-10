import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ============================================================================
# TENSILE TEST MODEL: Stress-Strain Behavior for Alloy 617
# ============================================================================


class TensileTestModel:
    """
    Tensile test model for Alloy 617 incorporating:
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

        # Temperature-dependent properties for Alloy 617
        if temp <= 100:
            self.E = 211000  # Young's modulus (MPa) at room temp
            self.yield_stress = 380  # Yield strength (MPa)
            self.K = 1200  # Strength coefficient (MPa)
            self.n = 0.35  # Strain hardening exponent
            self.UTS = 750  # Ultimate tensile strength (MPa)
        elif temp <= 500:
            self.E = 195000
            self.yield_stress = 320
            self.K = 1000
            self.n = 0.30
            self.UTS = 650
        elif temp <= 700:
            self.E = 180000
            self.yield_stress = 280
            self.K = 850
            self.n = 0.25
            self.UTS = 550
        else:  # High temperature (750-950°C)
            self.E = 170000
            self.yield_stress = 200
            self.K = 650
            self.n = 0.20
            self.UTS = 400

        # Calculate yield strain
        self.yield_strain = self.yield_stress / self.E

        # Necking parameters
        self.necking_strain = self.n  # Necking occurs at strain = n (Considère criterion)
        self.fracture_strain = self.necking_strain + 0.15  # Additional elongation in neck

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
                # Use Ramberg-Osgood type transition
                eps_plastic = eps - self.yield_strain
                stress[i] = self.K * (eps_plastic + self.yield_strain) ** self.n

            elif eps <= self.fracture_strain:
                # Necking region (stress decreases)
                # Linear decrease from UTS to failure stress
                progress = (eps - self.necking_strain) / (
                    self.fracture_strain - self.necking_strain
                )
                failure_stress = 0.7 * self.UTS  # Stress at fracture
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
        model = TensileTestModel(temp=temp, strain_rate=strain_rate)

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
temperatures = [25, 400, 650, 850]  # °C
strain_rate = 1e-3  # 1/s (quasi-static)

print("Generating synthetic tensile test data...")
tensile_data = generate_tensile_dataset(temperatures, strain_rate=strain_rate)

# ============================================================================
# MATERIALS PROPERTY EXTRACTION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTED MECHANICAL PROPERTIES")
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
print("PYSINDY CONSTITUTIVE MODEL DISCOVERY")
print("=" * 70)

# Use room temperature data for SINDy analysis
sample_data = tensile_data[tensile_data["temperature"] == 25].copy().reset_index(drop=True)

# For SINDy, we need to model the relationship between strain and stress
# We'll use strain as the "time" variable (it's monotonically increasing)
# and stress as the state variable

# Prepare data: stress as state variable, strain as time
X = sample_data["stress"].values.reshape(-1, 1)  # State variable
t = sample_data["strain"].values  # "Time" variable (monotonically increasing)

# Ensure strictly increasing time
# Remove any duplicate or non-increasing strain values
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

# Fit model: modeling dσ/dε (stress rate with respect to strain)
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
        if abs(coef) > 0.01:  # Only print significant coefficients
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
