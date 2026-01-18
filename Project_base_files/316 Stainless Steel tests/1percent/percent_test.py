# ================================
# ENHANCED ANALYSIS & INSIGHTS
# ================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\Users\phili\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Test_data\SS316H-1percent.csv")

print("="*70)
print("SS316H CREEP MODEL - DETAILED ANALYSIS")
print("="*70)

# Encode Heat
if df["Heat"].dtype == object:
    heat_mapping = {heat: i for i, heat in enumerate(df["Heat"].unique())}
    df["Heat_encoded"] = df["Heat"].map(heat_mapping)
    print("\nHeat Encodings:")
    for heat, code in heat_mapping.items():
        count = (df["Heat"] == heat).sum()
        print(f"  {heat:15s} → {code} ({count} samples)")
else:
    df["Heat_encoded"] = df["Heat"]

# Extract features
Heat = df["Heat_encoded"].values
Temp = df["Temp (K)"].values
Stress = df["Stress (Mpa)"].values
Time = df["Time (h) to 1% strain"].values

print(f"\nDataset Statistics:")
print(f"  Total samples: {len(Time)}")
print(f"  Time:   {Time.min():.1f} - {Time.max():.1f} hours ({Time.max()/8760:.1f} years)")
print(f"  Temp:   {Temp.min():.1f} - {Temp.max():.1f} K ({Temp.min()-273:.1f}°C - {Temp.max()-273:.1f}°C)")
print(f"  Stress: {Stress.min():.1f} - {Stress.max():.1f} MPa")

# Create features
X_features = np.column_stack([
    np.ones_like(Heat), Heat, Heat**2,
    Temp, 1/Temp, Temp**2, 1/(Temp**2),
    Stress, np.log(Stress), Stress**2, Stress**3,
    Stress**(-1), Stress**(-3), Stress**(-5),
    (1/Temp) * Stress, (1/Temp) * np.log(Stress),
    Temp * np.log(Stress), (1/Temp) * (Stress**(-5)),
    Heat * (1/Temp), Heat * np.log(Stress)
])

feature_names = [
    'Const', 'Heat', 'Heat²', 'T', '1/T', 'T²', '1/T²',
    'σ', 'log(σ)', 'σ²', 'σ³', 'σ⁻¹', 'σ⁻³', 'σ⁻⁵',
    '(1/T)σ', '(1/T)log(σ)', 'T·log(σ)', '(1/T)σ⁻⁵',
    'Heat·(1/T)', 'Heat·log(σ)'
]

y_log = np.log(Time)

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.25, random_state=42
)

# Train best model
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

# Predictions
y_pred_log_train = model.predict(X_train)
y_pred_log_test = model.predict(X_test)

Time_train = np.exp(y_train)
Time_pred_train = np.exp(y_pred_log_train)
Time_test = np.exp(y_test)
Time_pred_test = np.exp(y_pred_log_test)

# Calculate detailed metrics
train_r2 = r2_score(y_train, y_pred_log_train)
test_r2 = r2_score(y_test, y_pred_log_test)

train_errors = np.abs((Time_pred_train - Time_train) / Time_train) * 100
test_errors = np.abs((Time_pred_test - Time_test) / Time_test) * 100

print(f"\n{'='*70}")
print("MODEL PERFORMANCE")
print(f"{'='*70}")
print(f"Training R² (log):  {train_r2:.4f}")
print(f"Test R² (log):      {test_r2:.4f}")
print(f"Overfitting check:  {train_r2 - test_r2:.4f} (< 0.1 is good)")

print(f"\nError Distribution (Test Set):")
print(f"  Mean error:       {test_errors.mean():.1f}%")
print(f"  Median error:     {np.median(test_errors):.1f}%")
print(f"  25th percentile:  {np.percentile(test_errors, 25):.1f}%")
print(f"  75th percentile:  {np.percentile(test_errors, 75):.1f}%")
print(f"  90th percentile:  {np.percentile(test_errors, 90):.1f}%")
print(f"  Max error:        {test_errors.max():.1f}%")

# Identify best and worst predictions
best_idx = np.argmin(test_errors)
worst_idx = np.argmax(test_errors)

print(f"\nBest Prediction:")
print(f"  Actual: {Time_test[best_idx]:.1f}h, Predicted: {Time_pred_test[best_idx]:.1f}h, Error: {test_errors[best_idx]:.1f}%")

print(f"\nWorst Prediction:")
print(f"  Actual: {Time_test[worst_idx]:.1f}h, Predicted: {Time_pred_test[worst_idx]:.1f}h, Error: {test_errors[worst_idx]:.1f}%")

# Feature importance analysis
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (Scaled Coefficients)")
print(f"{'='*70}")
coef_abs = np.abs(model.coef_)
sorted_indices = np.argsort(coef_abs)[::-1]

print("\nAll Features (sorted by importance):")
for i, idx in enumerate(sorted_indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:15s}: {model.coef_[idx]:8.4f}")

# Physics interpretation
print(f"\n{'='*70}")
print("PHYSICS INTERPRETATION")
print(f"{'='*70}")

norton_coef = model.coef_[8]  # log(σ)
arrhenius_coef = model.coef_[4]  # 1/T

print(f"Norton stress exponent (n):     ~{-norton_coef:.2f}")
print(f"Arrhenius temperature effect:   {arrhenius_coef:.4f}")
print(f"\nKey creep mechanisms identified:")
if abs(model.coef_[4]) > 0.5:  # 1/T
    print("  ✓ Strong temperature dependence (Arrhenius behavior)")
if abs(model.coef_[8]) > 0.3:  # log(σ)
    print("  ✓ Power-law stress dependence (Norton-Bailey)")
if abs(model.coef_[16]) > 0.5:  # T·log(σ)
    print("  ✓ Temperature-stress interaction (complex creep)")

# =====================================
# COMPREHENSIVE VISUALIZATION
# =====================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Log-space predictions
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_pred_log_train, alpha=0.4, s=30, label='Train', color='blue')
ax1.scatter(y_test, y_pred_log_test, alpha=0.6, s=50, label='Test', 
            edgecolors='black', linewidth=0.5, color='red')
ax1.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()], 
         'k--', linewidth=2, label='Perfect')
ax1.set_xlabel("Measured log(Time)", fontsize=11)
ax1.set_ylabel("Predicted log(Time)", fontsize=11)
ax1.set_title(f"Log-Space Predictions (R²={test_r2:.3f})", fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Real-space (log-log)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(Time_train, Time_pred_train, alpha=0.4, s=30, label='Train', color='blue')
ax2.scatter(Time_test, Time_pred_test, alpha=0.6, s=50, label='Test',
            edgecolors='black', linewidth=0.5, color='red')
ax2.plot([Time.min(), Time.max()], [Time.min(), Time.max()], 
         'k--', linewidth=2, label='Perfect')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Measured Time (h)", fontsize=11)
ax2.set_ylabel("Predicted Time (h)", fontsize=11)
ax2.set_title("Real-Space (Log-Log)", fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Residuals
ax3 = fig.add_subplot(gs[0, 2])
residuals_test = y_test - y_pred_log_test
ax3.scatter(y_pred_log_test, residuals_test, alpha=0.6, s=50, 
            edgecolors='black', linewidth=0.5)
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel("Predicted log(Time)", fontsize=11)
ax3.set_ylabel("Residual", fontsize=11)
ax3.set_title("Residuals (Test Set)", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(test_errors, bins=20, edgecolor='black', alpha=0.7)
ax4.axvline(np.median(test_errors), color='red', linestyle='--', 
            linewidth=2, label=f'Median: {np.median(test_errors):.1f}%')
ax4.set_xlabel("Prediction Error (%)", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title("Error Distribution", fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Feature importance
ax5 = fig.add_subplot(gs[1, 1])
top_n = 10
top_indices = np.argsort(coef_abs)[-top_n:]
ax5.barh(range(top_n), coef_abs[top_indices])
ax5.set_yticks(range(top_n))
ax5.set_yticklabels([feature_names[i] for i in top_indices])
ax5.set_xlabel("Absolute Coefficient", fontsize=11)
ax5.set_title("Top 10 Features", fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Error vs Temperature
ax6 = fig.add_subplot(gs[1, 2])
test_indices = np.arange(len(y_test))
X_test_original = scaler.inverse_transform(X_test)
temps_test = X_test_original[:, 3]  # Temperature column
ax6.scatter(temps_test, test_errors, alpha=0.6, s=50, 
            edgecolors='black', linewidth=0.5)
ax6.set_xlabel("Temperature (K)", fontsize=11)
ax6.set_ylabel("Prediction Error (%)", fontsize=11)
ax6.set_title("Error vs Temperature", fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Error vs Stress
ax7 = fig.add_subplot(gs[2, 0])
stress_test = X_test_original[:, 7]  # Stress column
ax7.scatter(stress_test, test_errors, alpha=0.6, s=50,
            edgecolors='black', linewidth=0.5)
ax7.set_xlabel("Stress (MPa)", fontsize=11)
ax7.set_ylabel("Prediction Error (%)", fontsize=11)
ax7.set_title("Error vs Stress", fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Prediction range visualization
ax8 = fig.add_subplot(gs[2, 1:])
sorted_idx = np.argsort(Time_test)
x_pos = np.arange(len(Time_test))
ax8.plot(x_pos, Time_test[sorted_idx], 'ko-', label='Actual', markersize=6, linewidth=2)
ax8.plot(x_pos, Time_pred_test[sorted_idx], 'rs-', label='Predicted', 
         markersize=6, linewidth=2, alpha=0.7)
ax8.fill_between(x_pos, Time_test[sorted_idx]*0.5, Time_test[sorted_idx]*2,
                  alpha=0.2, color='gray', label='±100% band')
ax8.set_yscale('log')
ax8.set_xlabel("Sample Index (sorted by actual time)", fontsize=11)
ax8.set_ylabel("Time to 1% Strain (h)", fontsize=11)
ax8.set_title("Predictions Sorted by Actual Time", fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, which='both')

plt.suptitle('SS316H Creep Model - Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig("Test_Output/1percent.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("Analysis complete! Plot saved as '1percent_detailed_analysis.png'")
print(f"{'='*70}")