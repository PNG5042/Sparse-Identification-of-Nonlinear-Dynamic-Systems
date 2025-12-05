# stainless_steel_creep.py
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# 1. Generate Synthetic Creep Data
# -----------------------------
def generate_creep_data(t, stress=100, temp=500, noise_level=0.0005):
    """
    Generate synthetic creep strain for stainless steel:
    3 stages: primary (decelerating), secondary (steady), tertiary (accelerating)
    """
    t1, t2 = 0.15 * t.max(), 0.7 * t.max()
    A = 1e-6 * stress**1.5 * np.exp(-5000 / (temp + 273))
    strain = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t1:
            strain[i] = A * 1000 * ti**0.3
        elif ti < t2:
            eps_t1 = A * 1000 * t1**0.3
            strain[i] = eps_t1 + A * 50 * (ti - t1)
        else:
            eps_t1 = A * 1000 * t1**0.3
            eps_t2 = eps_t1 + A * 50 * (t2 - t1)
            strain[i] = eps_t2 + A * 10 * (np.exp(0.5 * (ti - t2)) - 1)
    strain += noise_level * np.random.randn(len(t))
    return strain

# -----------------------------
# 2. Prepare Data
# -----------------------------
np.random.seed(42)
time = np.linspace(0.01, 10, 500)
strain_raw = generate_creep_data(time)

# Smooth strain
strain_smooth = savgol_filter(strain_raw, window_length=51, polyorder=3)
# Compute strain rate
strain_rate = np.gradient(strain_smooth, time)

# -----------------------------
# 3. Stage Segmentation
# -----------------------------
t_primary = 1.5
t_secondary = 7.0

mask_primary = time < t_primary
mask_secondary = (time >= t_primary) & (time < t_secondary)
mask_tertiary = time >= t_secondary

X_primary = strain_smooth[mask_primary].reshape(-1, 1)
X_secondary = strain_smooth[mask_secondary].reshape(-1, 1)
X_tertiary = strain_smooth[mask_tertiary].reshape(-1, 1)

t_prim = time[mask_primary]
t_sec = time[mask_secondary]
t_tert = time[mask_tertiary]

# -----------------------------
# 4. Stage-Segmented Models
# -----------------------------
if len(X_primary) > 10:
    model_primary = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                             optimizer=ps.STLSQ(threshold=1e-6))
    model_primary.fit(X_primary, t=t_prim)
    print("Primary Creep Model:")
    model_primary.print()

if len(X_secondary) > 10:
    model_secondary = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                               optimizer=ps.STLSQ(threshold=1e-6))
    model_secondary.fit(X_secondary, t=t_sec)
    print("Secondary Creep Model:")
    model_secondary.print()

if len(X_tertiary) > 10:
    model_tertiary = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                               optimizer=ps.STLSQ(threshold=1e-6))
    model_tertiary.fit(X_tertiary, t=t_tert)
    print("Tertiary Creep Model:")
    model_tertiary.print()

# -----------------------------
# 5. Unified Model
# -----------------------------
X_unified = strain_smooth.reshape(-1, 1)
model_unified = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                          optimizer=ps.STLSQ(threshold=1e-4))
model_unified.fit(X_unified, t=time)
print("Unified Model:")
model_unified.print()

try:
    strain_pred = model_unified.simulate(X_unified[0], t=time)
    r2 = r2_score(strain_smooth, strain_pred.flatten())
    rmse = np.sqrt(mean_squared_error(strain_smooth, strain_pred.flatten()))
    print(f"Unified Model R²: {r2:.4f}, RMSE: {rmse:.6e}")
except Exception as e:
    print(f"Simulation failed: {e}")
    strain_pred = None

# -----------------------------
# 6. Train/Test Split Validation
# -----------------------------
X_train, X_test, t_train, t_test = train_test_split(
    X_unified, time, test_size=0.2, shuffle=False
)

model_val = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                     optimizer=ps.STLSQ(threshold=1e-4))
model_val.fit(X_train, t=t_train)

try:
    strain_pred_test = model_val.simulate(X_train[-1], t=t_test - t_train[-1])
    r2_test = r2_score(X_test.flatten(), strain_pred_test.flatten())
    rmse_test = np.sqrt(mean_squared_error(X_test.flatten(), strain_pred_test.flatten()))
    print(f"Test R²: {r2_test:.4f}, Test RMSE: {rmse_test:.6e}")
except Exception as e:
    print(f"Validation simulation failed: {e}")
    strain_pred_test = None

# -----------------------------
# 7. Visualization
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot strain
ax1 = axes[0]
ax1.plot(time, strain_raw, '.', alpha=0.3, label='Raw')
ax1.plot(time, strain_smooth, '-', label='Smoothed')
if strain_pred is not None:
    ax1.plot(time, strain_pred, '--', label='SINDy Pred')
ax1.axvline(t_primary, color='k', linestyle='--', alpha=0.5)
ax1.axvline(t_secondary, color='k', linestyle='--', alpha=0.5)
ax1.set_ylabel('Strain')
ax1.set_xlabel('Time (h)')
ax1.set_title('Stainless Steel Creep Test')
ax1.legend()
ax1.grid(True)

# Plot strain rate
ax2 = axes[1]
ax2.plot(time, strain_rate, 'b-')
ax2.axvline(t_primary, color='k', linestyle='--', alpha=0.5)
ax2.axvline(t_secondary, color='k', linestyle='--', alpha=0.5)
ax2.set_ylabel('Strain Rate (1/h)')
ax2.set_xlabel('Time (h)')
ax2.set_title('Strain Rate Evolution')
ax2.grid(True)

plt.tight_layout()
plt.savefig('stainless_steel_creep.png', dpi=150)
plt.show()

print("Plot saved as: stainless_steel_creep.png")
