import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ==============================
# LOAD + CLEAN DATA
# ==============================
df = pd.read_csv(
    r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\316 Stainless Steel tests\on-ter-creep\SS316H-on-ter-creep.csv"
)

# Rename columns to easier names
df = df.rename(columns={
    "Count": "count",
    "Heat": "heat",
    "Temp (K)": "temp_k",
    "Stress (Mpa)": "stress_mpa",
    "Time (h) to ter creep": "t_ter_h",
})

# Make sure numeric columns are numeric
for col in ["temp_k", "stress_mpa", "t_ter_h"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Heat is mixed string/numeric -> keep as string for now
df["heat"] = df["heat"].astype(str).str.strip()

# Drop rows with missing key values
df = df.dropna(subset=["temp_k", "stress_mpa", "t_ter_h"]).reset_index(drop=True)

# Optional: add Celsius
df["temp_c"] = df["temp_k"] - 273.15

# ==============================
# FEATURES + TARGET
# ==============================
X = df[["temp_k", "stress_mpa"]].values

# Strongly recommended: model log(time) because target spans orders of magnitude
y = np.log(df["t_ter_h"].values)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# MODEL TRAINING (Ridge)
# ==============================
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict (in log space)
y_pred = model.predict(X_test_scaled)

# Convert back to hours for human-readable metrics/plots
y_test_h = np.exp(y_test)
y_pred_h = np.exp(y_pred)

# ==============================
# METRICS
# ==============================
mae = mean_absolute_error(y_test_h, y_pred_h)
mape = mean_absolute_percentage_error(y_test_h, y_pred_h)
r2_log = r2_score(y_test, y_pred)        # R^2 in log space (recommended)
r2_hours = r2_score(y_test_h, y_pred_h)  # R^2 in hours (can look worse due to scale)

print("===== Metrics (hours) =====")
print(f"MAE  = {mae:.3f} h")
print(f"MAPE = {mape*100:.2f} %")
print("===== R^2 =====")
print(f"R^2 (log-hours space) = {r2_log:.4f}")
print(f"R^2 (hours space)     = {r2_hours:.4f}")

print("\nModel coefficients (scaled inputs):", model.coef_)
print("Intercept (log-hours):", model.intercept_)

# ==============================
# COMPREHENSIVE VISUALIZATION
# ==============================

# 1) Parity plot (hours)
plt.figure(figsize=(6, 6))
plt.scatter(y_test_h, y_pred_h, alpha=0.7)
mn = min(y_test_h.min(), y_pred_h.min())
mx = max(y_test_h.max(), y_pred_h.max())
plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
plt.xlabel("Actual Time to Tertiary Creep (h)")
plt.ylabel("Predicted Time to Tertiary Creep (h)")
plt.title("Parity Plot (Hours): Ridge on log(Time)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Parity plot (log-hours)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
plt.xlabel("Actual log(Time to Tertiary) (log h)")
plt.ylabel("Predicted log(Time to Tertiary) (log h)")
plt.title("Parity Plot (Log-Hours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Residuals vs Predicted (hours)
residuals_h = y_test_h - y_pred_h
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_h, residuals_h, alpha=0.7)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted Time to Tertiary Creep (h)")
plt.ylabel("Residual (Actual - Predicted) (h)")
plt.title("Residuals vs Predicted (Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Residual distribution (hours)
plt.figure(figsize=(6, 4))
plt.hist(residuals_h, bins=30, alpha=0.75)
plt.xlabel("Residual (h)")
plt.ylabel("Count")
plt.title("Residual Distribution (Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) Residuals vs Stress (hours)
plt.figure(figsize=(6, 4))
plt.scatter(X_test[:, 1], residuals_h, alpha=0.7)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Stress (MPa)")
plt.ylabel("Residual (h)")
plt.title("Residuals vs Stress (Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6) Residuals vs Temperature (hours)
plt.figure(figsize=(6, 4))
plt.scatter(X_test[:, 0], residuals_h, alpha=0.7)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Temperature (K)")
plt.ylabel("Residual (h)")
plt.title("Residuals vs Temperature (Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Predicted surface (Temp–Stress -> predicted hours)
temp_range = np.linspace(df["temp_k"].min(), df["temp_k"].max(), 60)
stress_range = np.linspace(df["stress_mpa"].min(), df["stress_mpa"].max(), 60)
TT, SS = np.meshgrid(temp_range, stress_range)

grid = np.column_stack([TT.ravel(), SS.ravel()])
grid_scaled = scaler.transform(grid)

Z_log = model.predict(grid_scaled).reshape(TT.shape)
Z_h = np.exp(Z_log)

plt.figure(figsize=(7, 5))
cp = plt.contourf(TT, SS, Z_h, levels=30)
plt.colorbar(cp, label="Predicted Time to Tertiary Creep (h)")
plt.xlabel("Temperature (K)")
plt.ylabel("Stress (MPa)")
plt.title("Predicted Tertiary Creep Time Surface (Hours)")
plt.tight_layout()
plt.show()

# 8) Show the top/bottom predicted hours on the test set (quick sanity check)
test_results = pd.DataFrame({
    "temp_k": X_test[:, 0],
    "stress_mpa": X_test[:, 1],
    "actual_h": y_test_h,
    "pred_h": y_pred_h,
    "abs_err_h": np.abs(residuals_h),
})

print("\nTop 10 worst absolute errors (test set):")
print(test_results.sort_values("abs_err_h", ascending=False).head(10).to_string(index=False))
