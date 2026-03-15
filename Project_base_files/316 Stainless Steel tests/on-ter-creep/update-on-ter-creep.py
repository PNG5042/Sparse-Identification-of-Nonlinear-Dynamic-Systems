import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

# ==============================
# LOAD + CLEAN DATA
# ==============================
df = pd.read_csv(
    r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\316 Stainless Steel tests\on-ter-creep\SS316H-on-ter-creep.csv"
)

df = df.rename(columns={
    "Count": "count",
    "Heat": "heat",
    "Temp (K)": "temp_k",
    "Stress (Mpa)": "stress_mpa",
    "Time (h) to ter creep": "t_ter_h",
})

for col in ["temp_k", "stress_mpa", "t_ter_h"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["heat"] = df["heat"].astype(str).str.strip()
df = df.dropna(subset=["temp_k", "stress_mpa", "t_ter_h"]).reset_index(drop=True)
df = df[df["t_ter_h"] > 0].reset_index(drop=True)

# =======================================
# OUTLIER REMOVAL
# =======================================
upper_limit = df["t_ter_h"].quantile(0.99)
df = df[df["t_ter_h"] < upper_limit].reset_index(drop=True)

# =======================================
# PHYSICS-INFORMED FEATURES
# =======================================
df["inv_temp"] = 1.0 / df["temp_k"]
df["log_stress"] = np.log(df["stress_mpa"])
df["stress_temp_ratio"] = df["stress_mpa"] / df["temp_k"]
df["stress_sq"] = df["stress_mpa"] ** 2
df["temp_sq"] = df["temp_k"] ** 2

# =======================================
# ONE-HOT ENCODE HEAT
# =======================================
df_encoded = pd.get_dummies(df, columns=["heat"], drop_first=True)

# =======================================
# FINAL FEATURE MATRIX
# =======================================
feature_cols = [
    "temp_k", "stress_mpa",
    "inv_temp", "log_stress",
    "stress_temp_ratio", "stress_sq", "temp_sq"
]
heat_cols = [c for c in df_encoded.columns if c.startswith("heat_")]
feature_cols += heat_cols

X = df_encoded[feature_cols].values
y = np.log(df_encoded["t_ter_h"].values)  # log time

# =======================================
# SPLIT + SCALE (Ridge needs scaling)
# =======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================================
# TRAIN EXTRA TREES (nonlinear, no xgboost)
# =======================================
et_model = ExtraTreesRegressor(
    n_estimators=800,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
y_pred_et_log = et_model.predict(X_test)
y_pred_et = np.exp(y_pred_et_log)

# =======================================
# TRAIN RIDGE (baseline)
# =======================================
ridge_model = Ridge(alpha=0.1, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge_log = ridge_model.predict(X_test_scaled)
y_pred_ridge = np.exp(y_pred_ridge_log)

# =======================================
# EVALUATE
# =======================================
def evaluate_model(name, y_test_log, y_pred_h, y_pred_log):
    y_true_h = np.exp(y_test_log)
    y_pred_log = np.log(y_pred_h)

    print("\n===== Metrics (hours) =====")
    print(f"MAE  = {mean_absolute_error(y_true_h, y_pred_h):.3f} h")
    print(f"MAPE = {mean_absolute_percentage_error(y_true_h, y_pred_h)*100:.2f} %")

    print("\n===== R^2 =====")
    print(f"R^2 (log space) = {r2_score(y_test_log, y_pred_log):.6f}")
    print(f"R^2 (hours)     = {r2_score(y_true_h, y_pred_h):.6f}")

evaluate_model("ExtraTrees", y_test, y_pred_et, y_pred_et_log)
evaluate_model("Ridge Baseline", y_test, y_pred_ridge, y_pred_ridge_log)

# =======================================
# VISUALIZATION
# =======================================
def parity_plot(y_test_log, y_pred_h, title):
    y_test_h = np.exp(y_test_log)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_h, y_pred_h, alpha=0.6)
    mn, mx = min(y_test_h.min(), y_pred_h.min()), max(y_test_h.max(), y_pred_h.max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual (h)")
    plt.ylabel("Predicted (h)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

parity_plot(y_test, y_pred_et, "ExtraTrees Parity Plot (Hours)")
parity_plot(y_test, y_pred_ridge, "Ridge Parity Plot (Hours)")

# Residuals (ExtraTrees)
residuals = np.exp(y_test) - y_pred_et
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_et, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted (h)")
plt.ylabel("Residual (h)")
plt.title("Residuals vs Predictions (ExtraTrees)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Stress vs Time to Tertiary Creep – log scale -> Norton law / stress exponent
plt.figure(figsize=(6,4))
plt.scatter(df["stress_mpa"], np.log(df["t_ter_h"]), alpha=0.6)
plt.xlabel("Stress (MPa)")
plt.ylabel("log(Time to Tertiary Creep) (log h)")
plt.title("Stress vs log(Time to Tertiary Creep)")
plt.grid(True)
plt.show()

# Temperature vs Time – Arrhenius check
plt.figure(figsize=(6,4))
plt.scatter(1.0 / df["temp_k"], np.log(df["t_ter_h"]), alpha=0.6)
plt.xlabel("Temperature (K)")
plt.ylabel("log(Time to Tertiary Creep) (log h)")
plt.title("Temp vs log(Time)")
plt.grid(True)
plt.show()