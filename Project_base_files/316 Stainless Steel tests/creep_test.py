import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Generate Synthetic Creep Data (Primary, Secondary, Tertiary stages)
# =============================================================================

def generate_creep_data(t, stress=100, temp=500, noise_level=0.001):
    """
    Generate synthetic creep strain data with three stages:
    - Primary: decelerating creep (power law)
    - Secondary: steady-state creep (linear)
    - Tertiary: accelerating creep (exponential)
    """
    # Time boundaries for stages
    t1, t2 = 0.15 * t.max(), 0.7 * t.max()
    
    # Stress and temperature factors
    A = 1e-6 * stress**1.5 * np.exp(-5000 / (temp + 273))
    
    strain = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t1:  # Primary creep
            strain[i] = A * 1000 * ti**0.3
        elif ti < t2:  # Secondary creep
            eps_t1 = A * 1000 * t1**0.3
            strain[i] = eps_t1 + A * 50 * (ti - t1)
        else:  # Tertiary creep
            eps_t1 = A * 1000 * t1**0.3
            eps_t2 = eps_t1 + A * 50 * (t2 - t1)
            strain[i] = eps_t2 + A * 10 * (np.exp(0.5 * (ti - t2)) - 1)
    
    # Add noise
    strain += noise_level * np.random.randn(len(t))
    return strain

# Create time array and generate data
np.random.seed(42)
time = np.linspace(0.01, 10, 500)
strain_raw = generate_creep_data(time, stress=100, temp=500, noise_level=0.0005)

print("=" * 60)
print("PYSINDY CREEP TEST ANALYSIS")
print("=" * 60)

# =============================================================================
# 2. Data Preprocessing
# =============================================================================

# Smooth noisy experimental data
strain_smooth = savgol_filter(strain_raw, window_length=51, polyorder=3)

# Estimate derivatives numerically
strain_rate = np.gradient(strain_smooth, time)

print("\n[1] Data Preprocessing Complete")
print(f"    Time range: {time.min():.2f} - {time.max():.2f} hours")
print(f"    Strain range: {strain_smooth.min():.6f} - {strain_smooth.max():.6f}")
print(f"    Max strain rate: {strain_rate.max():.6f} /hr")

# =============================================================================
# 3. Define Custom Library for Creep Phenomena
# =============================================================================

# Polynomial library
poly_library = ps.PolynomialLibrary(degree=3)

# Custom library with physically meaningful functions
custom_library = ps.CustomLibrary(
    library_functions=[
        lambda x: np.exp(-np.clip(x, -10, 10)),
        lambda x: np.sqrt(np.abs(x) + 1e-10),
        lambda x: np.log(np.abs(x) + 1e-6),
    ],
    function_names=[
        lambda x: f"exp(-{x})",
        lambda x: f"sqrt({x})",
        lambda x: f"log({x})"
    ]
)

# Combined library
library = poly_library + custom_library

print("\n[2] Feature Library Created")
print(f"    Polynomial degree: 3")
print(f"    Custom functions: exp(-x), sqrt(x), log(x)")

# =============================================================================
# 4. Approach A: Segment by Creep Stage
# =============================================================================

print("\n[3] Approach A: Stage-Segmented Models")
print("-" * 40)

# Define stage boundaries
t_primary = 1.5
t_secondary = 7.0

# Segment data
mask_primary = time < t_primary
mask_secondary = (time >= t_primary) & (time < t_secondary)
mask_tertiary = time >= t_secondary

# Reshape for SINDy (needs 2D array)
X_primary = strain_smooth[mask_primary].reshape(-1, 1)
X_secondary = strain_smooth[mask_secondary].reshape(-1, 1)
X_tertiary = strain_smooth[mask_tertiary].reshape(-1, 1)

t_prim = time[mask_primary]
t_sec = time[mask_secondary]
t_tert = time[mask_tertiary]

# Fit primary creep model
if len(X_primary) > 10:
    model_primary = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=1e-6)  # Lower threshold for small coefficients
    )
    model_primary.fit(X_primary, t=t_prim)
    print("\n  Primary Creep Model:")
    model_primary.print()

# Fit secondary creep model
if len(X_secondary) > 10:
    model_secondary = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=1e-6)  # Lower threshold for small coefficients
    )
    model_secondary.fit(X_secondary, t=t_sec)
    print("\n  Secondary Creep Model:")
    model_secondary.print()

# =============================================================================
# 5. Approach B: Unified Model
# =============================================================================

print("\n[4] Approach B: Unified Model (All Stages)")
print("-" * 40)

X_unified = strain_smooth.reshape(-1, 1)

model_unified = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=0.0001, alpha=0.01)
)
model_unified.fit(X_unified, t=time)

print("\n  Unified Model Equation:")
model_unified.print()

# Simulate and compare
try:
    strain_predicted = model_unified.simulate(X_unified[0], t=time)
    r2_unified = r2_score(strain_smooth, strain_predicted.flatten())
    print(f"\n  R² Score: {r2_unified:.4f}")
except Exception as e:
    print(f"  Simulation note: {e}")
    strain_predicted = None

# =============================================================================
# 6. Approach C: Multi-Experiment Ensemble
# =============================================================================

print("\n[5] Approach C: Multi-Experiment Ensemble")
print("-" * 40)

# Generate multiple experiments at different conditions
time_tests = [np.linspace(0.01, 10, 300) for _ in range(3)]
strain_tests = [
    generate_creep_data(time_tests[0], stress=80, temp=500).reshape(-1, 1),
    generate_creep_data(time_tests[1], stress=100, temp=500).reshape(-1, 1),
    generate_creep_data(time_tests[2], stress=120, temp=500).reshape(-1, 1),
]

model_ensemble = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.0001)
)

# In newer PySINDy versions, pass lists directly without the keyword
model_ensemble.fit(strain_tests, t=time_tests)

print("\n  Ensemble Model (3 stress levels):")
model_ensemble.print()

# =============================================================================
# 7. Compare Multiple Optimizers
# =============================================================================

print("\n[6] Optimizer Comparison")
print("-" * 40)

optimizers = {
    'STLSQ': ps.STLSQ(threshold=0.0001),
    'SSR': ps.SSR(criteria='model_residual'),
}

results = []
for name, opt in optimizers.items():
    try:
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=2),
            optimizer=opt
        )
        model.fit(X_unified, t=time)
        score = model.score(X_unified, t=time)
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
# 8. Parametric Model (Including Temperature and Stress)
# =============================================================================

print("\n[7] Parametric Model (Strain, Temp, Stress)")
print("-" * 40)

# Create multi-variable dataset
stress_val = 100
temp_val = 500
X_parametric = np.column_stack([
    strain_smooth,
    temp_val * np.ones_like(strain_smooth),
    stress_val * np.ones_like(strain_smooth)
])

library_parametric = ps.PolynomialLibrary(degree=2, include_interaction=True)

model_parametric = ps.SINDy(
    feature_library=library_parametric,
    optimizer=ps.STLSQ(threshold=0.00001)
)
model_parametric.fit(X_parametric, t=time)

print("\n  Parametric Model (x0=strain, x1=temp, x2=stress):")
model_parametric.print()

# =============================================================================
# 9. Validation with Train/Test Split
# =============================================================================

print("\n[8] Model Validation")
print("-" * 40)

# Split time series (keeping temporal order)
split_idx = int(0.8 * len(time))
time_train, time_test = time[:split_idx], time[split_idx:]
strain_train = strain_smooth[:split_idx].reshape(-1, 1)
strain_test = strain_smooth[split_idx:].reshape(-1, 1)

# Fit on training data
model_val = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.0001)
)
model_val.fit(strain_train, t=time_train)

# Predict on test data
try:
    strain_pred_test = model_val.simulate(strain_train[-1], t=time_test)
    r2 = r2_score(strain_test.flatten(), strain_pred_test.flatten())
    rmse = np.sqrt(mean_squared_error(strain_test.flatten(), strain_pred_test.flatten()))
    print(f"  Test R² Score: {r2:.4f}")
    print(f"  Test RMSE: {rmse:.6f}")
except Exception as e:
    print(f"  Prediction note: {e}")
    strain_pred_test = None

# =============================================================================
# 10. Generate Publication-Ready Figures
# =============================================================================

print("\n[9] Generating Figures...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Experimental vs Smoothed Data
ax1 = axes[0, 0]
ax1.plot(time, strain_raw, 'o', alpha=0.3, markersize=2, label='Raw Data')
ax1.plot(time, strain_smooth, '-', linewidth=2, label='Smoothed Data')
ax1.axvline(t_primary, color='gray', linestyle='--', alpha=0.5, label='Stage boundaries')
ax1.axvline(t_secondary, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Strain')
ax1.set_title('Creep Test Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Strain Rate vs Time
ax2 = axes[0, 1]
ax2.plot(time, strain_rate, 'b-', linewidth=1.5)
ax2.axvline(t_primary, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(t_secondary, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Strain Rate (1/h)')
ax2.set_title('Strain Rate Evolution')
ax2.text(t_primary/2, ax2.get_ylim()[1]*0.9, 'Primary', ha='center')
ax2.text((t_primary+t_secondary)/2, ax2.get_ylim()[1]*0.9, 'Secondary', ha='center')
ax2.text((t_secondary+time.max())/2, ax2.get_ylim()[1]*0.9, 'Tertiary', ha='center')
ax2.grid(True, alpha=0.3)

# Plot 3: Model Prediction vs Experimental
ax3 = axes[1, 0]
ax3.plot(time, strain_smooth, 'ko', markersize=3, alpha=0.5, label='Experimental')
if strain_predicted is not None:
    ax3.plot(time, strain_predicted, 'r-', linewidth=2, label='SINDy Prediction')
ax3.set_xlabel('Time (h)')
ax3.set_ylabel('Strain')
ax3.set_title('Model Prediction vs Experimental')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals
ax4 = axes[1, 1]
if strain_predicted is not None:
    residuals = strain_smooth - strain_predicted.flatten()
    ax4.plot(time, residuals, 'g-', linewidth=1)
    ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax4.fill_between(time, residuals, 0, alpha=0.3)
    ax4.set_xlabel('Time (h)')
    ax4.set_ylabel('Residual')
    ax4.set_title(f'Model Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.2e})')
else:
    ax4.text(0.5, 0.5, 'Residuals not available', ha='center', va='center', transform=ax4.transAxes)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('creep_sindy_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 11. Summary
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nKey Results:")
print(f"  - Data points: {len(time)}")
print(f"  - Creep stages identified: Primary (<{t_primary}h), Secondary ({t_primary}-{t_secondary}h), Tertiary (>{t_secondary}h)")
if results:
    best = max(results, key=lambda x: x['score'])
    print(f"  - Best optimizer: {best['name']} (Score: {best['score']:.4f})")
print(f"\nFigure saved to: creep_sindy_analysis.png")

