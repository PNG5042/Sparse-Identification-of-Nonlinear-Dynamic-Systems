import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# =========================================================
# LOAD + CLEAN DATA
# =========================================================
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

# =========================================================
# OUTLIER REMOVAL
# =========================================================
upper_limit = df["t_ter_h"].quantile(0.99)
df = df[df["t_ter_h"] < upper_limit].reset_index(drop=True)

# =========================================================
# PHYSICS-INFORMED FEATURES
# =========================================================
df["inv_temp"] = 1.0 / df["temp_k"]
df["log_stress"] = np.log(df["stress_mpa"])
df["stress_temp_ratio"] = df["stress_mpa"] / df["temp_k"]
df["stress_sq"] = df["stress_mpa"] ** 2
df["temp_sq"] = df["temp_k"] ** 2

# =========================================================
# ONE-HOT ENCODE HEAT
# =========================================================
df_encoded = pd.get_dummies(df, columns=["heat"], drop_first=False)

# =========================================================
# FINAL FEATURE MATRIX
# =========================================================
feature_cols = [
    "temp_k",
    "stress_mpa",
    "inv_temp",
    "log_stress",
    "stress_temp_ratio",
    "stress_sq",
    "temp_sq"
]
heat_cols = [c for c in df_encoded.columns if c.startswith("heat_")]
feature_cols += heat_cols

X = df_encoded[feature_cols].values
y = np.log(df_encoded["t_ter_h"].values)   # log(time to tertiary creep)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# SCALE DATA FOR RIDGE
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# TRAIN MODELS
# =========================================================
et_model = ExtraTreesRegressor(
    n_estimators=800,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=0.1, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

# =========================================================
# PREDICTIONS
# =========================================================
y_pred_et_log = et_model.predict(X_test)
y_pred_et = np.exp(y_pred_et_log)

y_pred_ridge_log = ridge_model.predict(X_test_scaled)
y_pred_ridge = np.exp(y_pred_ridge_log)

# =========================================================
# STANDARD PREDICTION METRICS
# =========================================================
def evaluate_prediction_model(name, y_test_log, y_pred_h):
    y_true_h = np.exp(y_test_log)
    y_pred_log = np.log(np.clip(y_pred_h, 1e-12, None))

    print(f"\n{'='*70}")
    print(f"{name} - PREDICTION METRICS")
    print(f"{'='*70}")
    print(f"MAE (hours)           = {mean_absolute_error(y_true_h, y_pred_h):.6f}")
    print(f"MAPE (%)              = {mean_absolute_percentage_error(y_true_h, y_pred_h)*100:.6f}")
    print(f"R^2 (log space)       = {r2_score(y_test_log, y_pred_log):.6f}")
    print(f"R^2 (hours)           = {r2_score(y_true_h, y_pred_h):.6f}")

evaluate_prediction_model("ExtraTrees", y_test, y_pred_et)
evaluate_prediction_model("Ridge", y_test, y_pred_ridge)

# =========================================================
# PRINT MODEL EQUATION PARAMETERS (RIDGE)
# =========================================================
# Ridge is the interpretable model here. ExtraTrees does not have a simple equation.
# Because Ridge was trained on scaled features, these are coefficients in scaled space.
# If you want a simple equation, print them exactly as the model learned them.

print(f"\n{'='*70}")
print("RIDGE MODEL EQUATION PARAMETERS")
print(f"{'='*70}")

ridge_intercept = ridge_model.intercept_
ridge_coeffs = ridge_model.coef_

print(f"Intercept: {ridge_intercept:.10f}")
for name, coef in zip(feature_cols, ridge_coeffs):
    print(f"{name:20s}: {coef:.10f}")

# =========================================================
# OPTIONAL: PRINT EQUATION STRING
# =========================================================
equation_terms = [f"({ridge_intercept:.10f})"]
for name, coef in zip(feature_cols, ridge_coeffs):
    sign = "+" if coef >= 0 else "-"
    equation_terms.append(f" {sign} ({abs(coef):.10f})*{name}")

equation_str = "log(t_ter_h) = " + "".join(equation_terms)

print(f"\n{'='*70}")
print("RIDGE MODEL EQUATION")
print(f"{'='*70}")
print(equation_str)

# =========================================================
# ANALYTICAL COEFFICIENTS
# =========================================================
# Replace these with the actual coefficients from your analytical equation.
# The keys must match the Ridge model parameter names exactly.
#
# Example:
# log(t_ter_h) = a0 + a1*temp_k + a2*stress_mpa + a3*inv_temp + ...
#
# Then define:
analytical_coeffs = {
    "Intercept": 0.5000,
    "temp_k": 0.0001,
    "stress_mpa": -0.0100,
    "inv_temp": 1000.0000,
    "log_stress": -2.5000,
    "stress_temp_ratio": -0.0500,
    "stress_sq": 0.00001,
    "temp_sq": -0.0000001,
    # Add heat coefficients if your analytical equation includes them
    # Example:
    # "heat_XYZ": 0.1234,
}

# =========================================================
# RELATIVE ERROR + METRIC
# metric = 1 - relative error
# relative error = (Analytical equation coefficient - SINDy equation coefficient)
#                  / Analytical equation coefficient
#
# In this code, Ridge coefficients are used as the model equation coefficients.
# =========================================================
print(f"\n{'='*70}")
print("COEFFICIENT COMPARISON: ANALYTICAL vs MODEL")
print(f"{'='*70}")

model_params = {"Intercept": ridge_intercept}
for name, coef in zip(feature_cols, ridge_coeffs):
    model_params[name] = coef

comparison_rows = []

all_terms = list(analytical_coeffs.keys())

for term in all_terms:
    analytical_value = analytical_coeffs[term]
    model_value = model_params.get(term, np.nan)

    if pd.isna(model_value):
        relative_error = np.nan
        metric = np.nan
        note = "Model parameter not found"
    elif analytical_value == 0:
        relative_error = np.nan
        metric = np.nan
        note = "Analytical coefficient is 0, cannot divide by zero"
    else:
        relative_error = (analytical_value - model_value) / analytical_value
        metric = 1 - relative_error
        note = ""

    comparison_rows.append({
        "Term": term,
        "Analytical equation coefficient": analytical_value,
        "Model equation coefficient": model_value,
        "Relative error": relative_error,
        "Metric": metric,
        "Note": note
    })

comparison_df = pd.DataFrame(comparison_rows)

print(comparison_df.to_string(index=False))

# =========================================================
# OVERALL METRIC
# =========================================================
valid_metric_values = comparison_df["Metric"].dropna()

print(f"\n{'='*70}")
print("OVERALL METRIC")
print(f"{'='*70}")

if len(valid_metric_values) > 0:
    overall_metric = valid_metric_values.mean()
    print(f"Average metric = {overall_metric:.10f}")
else:
    overall_metric = np.nan
    print("No valid metric could be computed.")


# =========================================================
# VISUALIZATION
# =========================================================
def parity_plot(y_test_log, y_pred_h, title):
    y_test_h = np.exp(y_test_log)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_h, y_pred_h, alpha=0.6)
    mn = min(y_test_h.min(), y_pred_h.min())
    mx = max(y_test_h.max(), y_pred_h.max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual (h)")
    plt.ylabel("Predicted (h)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

parity_plot(y_test, y_pred_et, "ExtraTrees Parity Plot (Hours)")
parity_plot(y_test, y_pred_ridge, "Ridge Parity Plot (Hours)")

# Residual plot for ExtraTrees
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

# Stress vs log(Time to Tertiary Creep)
plt.figure(figsize=(6, 4))
plt.scatter(df["stress_mpa"], np.log(df["t_ter_h"]), alpha=0.6)
plt.xlabel("Stress (MPa)")
plt.ylabel("log(Time to Tertiary Creep) (log h)")
plt.title("Stress vs log(Time to Tertiary Creep)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Temperature vs log(Time)
plt.figure(figsize=(6, 4))
plt.scatter(1.0 / df["temp_k"], np.log(df["t_ter_h"]), alpha=0.6)
plt.xlabel("1 / Temperature (1/K)")
plt.ylabel("log(Time to Tertiary Creep) (log h)")
plt.title("1/Temp vs log(Time)")
plt.grid(True)
plt.tight_layout()
plt.show()