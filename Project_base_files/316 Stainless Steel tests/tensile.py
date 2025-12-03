import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Generate Synthetic Tensile Test Data for 316H Stainless Steel
# =============================================================================

def generate_tensile_data(strain, temp=20, noise_level=0.5):
    """
    Generate synthetic tensile stress-strain data for 316H stainless steel:
    - Elastic region: Linear (E ~ 193 GPa)
    - Yield point: ~240 MPa
    - Strain hardening: Power law
    - Necking: Exponential softening
    
    Temperature effect on properties included.
    """
    # Temperature-dependent properties for 316H
    E_20 = 193000  # Young's modulus at 20°C (MPa)
    sigma_y_20 = 240  # Yield strength at 20°C (MPa)
    
    # Temperature correction factors (approximate)
    temp_factor = 1 - (temp - 20) / 1000
    E = E_20 * temp_factor
    sigma_y = sigma_y_20 * temp_factor
    
    # Material constants for 316H
    K = 1400 * temp_factor  # Strength coefficient (MPa)
    n = 0.45  # Strain hardening exponent
    
    # Strain boundaries
    strain_yield = sigma_y / E
    strain_uts = 0.45  # Ultimate tensile strain
    strain_fracture = 0.6  # Fracture strain
    
    stress = np.zeros_like(strain)
    
    for i, eps in enumerate(strain):
        if eps < strain_yield:  # Elastic region
            stress[i] = E * eps
        elif eps < strain_uts:  # Strain hardening region
            # Ramberg-Osgood type relationship
            eps_plastic = eps - strain_yield
            stress[i] = sigma_y + K * (eps_plastic ** n)
        else:  # Necking region
            # Exponential softening after UTS
            stress_uts = sigma_y + K * ((strain_uts - strain_yield) ** n)
            decay_rate = 5
            stress[i] = stress_uts * np.exp(-decay_rate * (eps - strain_uts))
    
    # Add measurement noise
    stress += noise_level * np.random.randn(len(strain))
    
    return stress

# Create strain array and generate data
np.random.seed(42)
strain = np.linspace(0, 0.65, 600)
stress_raw = generate_tensile_data(strain, temp=20, noise_level=2.0)

print("=" * 60)
print("PYSINDY TENSILE TEST ANALYSIS - 316H STAINLESS STEEL")
print("=" * 60)

# =============================================================================
# 2. Data Preprocessing
# =============================================================================

# Smooth noisy experimental data
stress_smooth = savgol_filter(stress_raw, window_length=51, polyorder=3)

# Estimate derivatives (strain hardening rate)
hardening_rate = np.gradient(stress_smooth, strain)

print("\n[1] Data Preprocessing Complete")
print(f"    Strain range: {strain.min():.4f} - {strain.max():.4f}")
print(f"    Stress range: {stress_smooth.min():.1f} - {stress_smooth.max():.1f} MPa")
print(f"    Max stress: {stress_smooth.max():.1f} MPa at strain {strain[np.argmax(stress_smooth)]:.3f}")
print(f"    Young's modulus (approx): {hardening_rate[10]:.0f} MPa")

# =============================================================================
# 3. Define Custom Library for Tensile Behavior
# =============================================================================

# Polynomial library for strain hardening
poly_library = ps.PolynomialLibrary(degree=3)

# Custom library with physically meaningful functions
custom_library = ps.CustomLibrary(
    library_functions=[
        lambda x: np.sqrt(np.abs(x) + 1e-10),
        lambda x: x ** 0.3,
        lambda x: x ** 0.5,
    ],
    function_names=[
        lambda x: f"sqrt({x})",
        lambda x: f"{x}^0.3",
        lambda x: f"{x}^0.5"
    ]
)

# Combined library
library = poly_library + custom_library

print("\n[2] Feature Library Created")
print(f"    Polynomial degree: 3")
print(f"    Custom functions: sqrt(ε), ε^0.3, ε^0.5 (power law hardening)")

# =============================================================================
# 4. Approach A: Segment by Tensile Regions
# =============================================================================

print("\n[3] Approach A: Region-Segmented Models")
print("-" * 40)

# Define region boundaries based on stress-strain behavior
strain_elastic_end = 0.002  # Approximate yield point
strain_hardening_end = 0.45  # Approximate UTS

# Segment data
mask_elastic = strain < strain_elastic_end
mask_hardening = (strain >= strain_elastic_end) & (strain < strain_hardening_end)
mask_necking = strain >= strain_hardening_end

# Reshape for SINDy (treating stress as state variable, strain as time)
X_elastic = stress_smooth[mask_elastic].reshape(-1, 1)
X_hardening = stress_smooth[mask_hardening].reshape(-1, 1)
X_necking = stress_smooth[mask_necking].reshape(-1, 1)

eps_elastic = strain[mask_elastic]
eps_hardening = strain[mask_hardening]
eps_necking = strain[mask_necking]

# Fit elastic region model
if len(X_elastic) > 10:
    model_elastic = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=1),
        optimizer=ps.STLSQ(threshold=1e-6)
    )
    model_elastic.fit(X_elastic, t=eps_elastic)
    print("\n  Elastic Region Model:")
    model_elastic.print()
    E_estimated = model_elastic.coefficients()[0, 1] if len(model_elastic.coefficients()[0]) > 1 else 0
    print(f"  Estimated Young's Modulus: {1/E_estimated if E_estimated != 0 else 0:.0f} MPa")

# Fit strain hardening model
if len(X_hardening) > 10:
    model_hardening = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=0.01)
    )
    model_hardening.fit(X_hardening, t=eps_hardening)
    print("\n  Strain Hardening Model:")
    model_hardening.print()

# =============================================================================
# 5. Approach B: Unified Model
# =============================================================================

print("\n[4] Approach B: Unified Model (All Regions)")
print("-" * 40)

# Use strain as independent variable, stress as dependent
# We'll model dσ/dε = f(σ, ε)
X_unified = np.column_stack([stress_smooth, strain])

model_unified = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
    optimizer=ps.STLSQ(threshold=1.0, alpha=0.05)
)
model_unified.fit(X_unified, t=strain)

print("\n  Unified Model Equation (x0=stress, x1=strain):")
model_unified.print()

# Simulate and compare
try:
    X_predicted = model_unified.simulate(X_unified[0], t=strain)
    stress_predicted = X_predicted[:, 0]
    r2_unified = r2_score(stress_smooth, stress_predicted)
    print(f"\n  R² Score: {r2_unified:.4f}")
except Exception as e:
    print(f"  Simulation note: {e}")
    stress_predicted = None

# =============================================================================
# 6. Approach C: Multi-Temperature Ensemble
# =============================================================================

print("\n[5] Approach C: Multi-Temperature Ensemble")
print("-" * 40)

# Generate tests at different temperatures
strain_tests = [np.linspace(0, 0.65, 400) for _ in range(3)]
stress_tests = [
    generate_tensile_data(strain_tests[0], temp=20).reshape(-1, 1),
    generate_tensile_data(strain_tests[1], temp=200).reshape(-1, 1),
    generate_tensile_data(strain_tests[2], temp=400).reshape(-1, 1),
]

model_ensemble = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=1.0)
)

model_ensemble.fit(stress_tests, t=strain_tests)

print("\n  Ensemble Model (3 temperature levels: 20°C, 200°C, 400°C):")
model_ensemble.print()

# =============================================================================
# 7. Compare Multiple Optimizers
# =============================================================================

print("\n[6] Optimizer Comparison")
print("-" * 40)

X_for_optimization = stress_smooth.reshape(-1, 1)

optimizers = {
    'STLSQ': ps.STLSQ(threshold=1.0),
    'SSR': ps.SSR(criteria='model_residual'),
}

results = []
for name, opt in optimizers.items():
    try:
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=2),
            optimizer=opt
        )
        model.fit(X_for_optimization, t=strain)
        score = model.score(X_for_optimization, t=strain)
        complexity = np.count_nonzero(model.coefficients())
        results.append({
            'name': name,
            'score': score,
            'complexity': complexity,
            'model': model
        })
        print(f"  {name:8s}: Score={score:.4f}, Non-zero coeffs={complexity}")
    except Exception as e:
        print(f"  {name:8s}: Failed - {e}")

# =============================================================================
# 8. Parametric Model (Including Temperature and Strain Rate)
# =============================================================================

print("\n[7] Parametric Model (Stress, Temp, Strain Rate)")
print("-" * 40)

# Create multi-variable dataset
temp_val = 20  # °C
strain_rate_val = 0.001  # /s (typical quasi-static test)

X_parametric = np.column_stack([
    stress_smooth,
    temp_val * np.ones_like(stress_smooth),
    strain_rate_val * np.ones_like(stress_smooth)
])

library_parametric = ps.PolynomialLibrary(degree=2, include_interaction=True)

model_parametric = ps.SINDy(
    feature_library=library_parametric,
    optimizer=ps.STLSQ(threshold=0.1)
)
model_parametric.fit(X_parametric, t=strain)

print("\n  Parametric Model (x0=stress, x1=temp, x2=strain_rate):")
model_parametric.print()

# =============================================================================
# 9. Validation with Train/Test Split
# =============================================================================

print("\n[8] Model Validation")
print("-" * 40)

# Split data (keeping order)
split_idx = int(0.8 * len(strain))
strain_train, strain_test = strain[:split_idx], strain[split_idx:]
stress_train = stress_smooth[:split_idx].reshape(-1, 1)
stress_test = stress_smooth[split_idx:].reshape(-1, 1)

# Fit on training data
model_val = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=1.0)
)
model_val.fit(stress_train, t=strain_train)

# Predict on test data
try:
    stress_pred_test = model_val.simulate(stress_train[-1], t=strain_test)
    r2 = r2_score(stress_test.flatten(), stress_pred_test.flatten())
    rmse = np.sqrt(mean_squared_error(stress_test.flatten(), stress_pred_test.flatten()))
    print(f"  Test R² Score: {r2:.4f}")
    print(f"  Test RMSE: {rmse:.2f} MPa")
except Exception as e:
    print(f"  Prediction note: {e}")
    stress_pred_test = None

# =============================================================================
# 10. Generate Publication-Ready Figures
# =============================================================================

print("\n[9] Generating Figures...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Stress-Strain Curve
ax1 = axes[0, 0]
ax1.plot(strain, stress_raw, 'o', alpha=0.3, markersize=2, label='Raw Data')
ax1.plot(strain, stress_smooth, '-', linewidth=2, label='Smoothed Data')
ax1.axvline(strain_elastic_end, color='gray', linestyle='--', alpha=0.5, label='Region boundaries')
ax1.axvline(strain_hardening_end, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Strain (mm/mm)')
ax1.set_ylabel('Stress (MPa)')
ax1.set_title('Tensile Test - 316H Stainless Steel')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Strain Hardening Rate
ax2 = axes[0, 1]
ax2.plot(strain, hardening_rate, 'b-', linewidth=1.5)
ax2.axvline(strain_elastic_end, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(strain_hardening_end, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Strain (mm/mm)')
ax2.set_ylabel('dσ/dε (MPa)')
ax2.set_title('Strain Hardening Rate')
ax2.text(strain_elastic_end/2, ax2.get_ylim()[1]*0.9, 'Elastic', ha='center')
ax2.text((strain_elastic_end+strain_hardening_end)/2, ax2.get_ylim()[1]*0.9, 'Hardening', ha='center')
ax2.text((strain_hardening_end+strain.max())/2, ax2.get_ylim()[1]*0.9, 'Necking', ha='center')
ax2.grid(True, alpha=0.3)

# Plot 3: Model Prediction vs Experimental
ax3 = axes[1, 0]
ax3.plot(strain, stress_smooth, 'ko', markersize=3, alpha=0.5, label='Experimental')
if stress_predicted is not None:
    ax3.plot(strain, stress_predicted, 'r-', linewidth=2, label='SINDy Prediction')
ax3.set_xlabel('Strain (mm/mm)')
ax3.set_ylabel('Stress (MPa)')
ax3.set_title('Model Prediction vs Experimental')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals
ax4 = axes[1, 1]
if stress_predicted is not None:
    residuals = stress_smooth - stress_predicted
    ax4.plot(strain, residuals, 'g-', linewidth=1)
    ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax4.fill_between(strain, residuals, 0, alpha=0.3)
    ax4.set_xlabel('Strain (mm/mm)')
    ax4.set_ylabel('Residual (MPa)')
    ax4.set_title(f'Model Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.2f} MPa)')
else:
    ax4.text(0.5, 0.5, 'Residuals not available', ha='center', va='center', transform=ax4.transAxes)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tensile_316h_sindy_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 11. Material Properties Summary
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - 316H STAINLESS STEEL")
print("=" * 60)

# Calculate key properties from the curve
yield_idx = np.where(strain > strain_elastic_end)[0][0]
uts_idx = np.argmax(stress_smooth)
elongation = strain[uts_idx]

print("\nMaterial Properties:")
print(f"  - Young's Modulus: ~{hardening_rate[10]:.0f} MPa")
print(f"  - Yield Strength (0.2% offset): ~{stress_smooth[yield_idx]:.1f} MPa")
print(f"  - Ultimate Tensile Strength: {stress_smooth[uts_idx]:.1f} MPa")
print(f"  - Elongation at UTS: {elongation*100:.1f}%")
print(f"  - Data points: {len(strain)}")

if results:
    best = max(results, key=lambda x: x['score'])
    print(f"  - Best optimizer: {best['name']} (Score: {best['score']:.4f})")

print(f"\nFigure saved to: tensile_316h_sindy_analysis.png")
print("\nNote: This analysis demonstrates PySINDy's capability to identify")
print("      constitutive relationships for strain hardening behavior.")