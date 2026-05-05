import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# USER SETTINGS
# =========================================================
CSV_FILE = r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\316 Stainless Steel tests\on-ter-creep\SS316H-on-ter-creep.csv"

DASHBOARD_OUT = "SS316H_creep_dashboard.png"

# =========================================================
# NOTE ABOUT REQUIREMENT
# =========================================================
# Required formulas:
#
# relative error =
# (Analytical equation coefficient - SINDy equation coefficient)
# / Analytical equation coefficient
#
# metric = 1 - relative error
#
# This code uses Ridge model coefficients as the model equation coefficients.
# If your instructor requires true SINDy coefficients, this section can be
# modified after your Monday meeting.
# =========================================================

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_FILE)

# =========================================================
# RENAME COLUMNS
# =========================================================
df = df.rename(columns={
    "Count": "count",
    "Heat": "heat",
    "Temp (K)": "temp_k",
    "Stress (Mpa)": "stress_mpa",
    "Time (h) to ter creep": "t_ter_h",
})

# =========================================================
# CLEAN DATA
# =========================================================
for col in ["temp_k", "stress_mpa", "t_ter_h"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["heat"] = df["heat"].astype(str).str.strip()

df = df.dropna(
    subset=["temp_k", "stress_mpa", "t_ter_h"]
).reset_index(drop=True)

df = df[df["t_ter_h"] > 0].reset_index(drop=True)

# =========================================================
# REMOVE OUTLIERS
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
df_encoded = pd.get_dummies(df,columns=["heat"],drop_first=False)

# =========================================================
# FEATURE MATRIX
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

X_df = df_encoded[feature_cols]
X = X_df.values

# Target variable: log time to tertiary creep
y = np.log(df_encoded["t_ter_h"].values)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
(X_train,X_test,y_train,y_test,X_train_df,X_test_df) = train_test_split(
    X,y,X_df,test_size=0.2,random_state=42
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

ridge_model = Ridge(
    alpha=0.1,
    random_state=42
)

ridge_model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# PREDICTIONS
# =========================================================
y_test_h = np.exp(y_test)

# ExtraTrees prediction
y_pred_et_log = et_model.predict(X_test)
y_pred_et_h = np.exp(y_pred_et_log)

# Ridge prediction
y_pred_ridge_log = ridge_model.predict(X_test_scaled)
y_pred_ridge_h = np.exp(y_pred_ridge_log)

# =========================================================
# STANDARD PREDICTION METRICS
# =========================================================
def get_metrics(y_true_log, y_pred_h):
    y_true_h = np.exp(y_true_log)
    y_pred_log = np.log(np.clip(y_pred_h, 1e-12, None))

    mae = mean_absolute_error(y_true_h, y_pred_h)
    mape = mean_absolute_percentage_error(y_true_h, y_pred_h) * 100
    rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    r2_log = r2_score(y_true_log, y_pred_log)
    r2_h = r2_score(y_true_h, y_pred_h)

    return mae, mape, rmse, r2_log, r2_h


(
    et_mae,
    et_mape,
    et_rmse,
    et_r2_log,
    et_r2_h
) = get_metrics(y_test, y_pred_et_h)

(
    ridge_mae,
    ridge_mape,
    ridge_rmse,
    ridge_r2_log,
    ridge_r2_h
) = get_metrics(y_test, y_pred_ridge_h)

# =========================================================
# PRINT PREDICTION METRICS
# =========================================================
print("\n" + "=" * 70)
print("ExtraTrees - PREDICTION METRICS")
print("=" * 70)
print(f"MAE (hours)     = {et_mae:.6f}")
print(f"MAPE (%)        = {et_mape:.6f}")
print(f"RMSE (hours)    = {et_rmse:.6f}")
print(f"R² log space    = {et_r2_log:.6f}")
print(f"R² hours        = {et_r2_h:.6f}")

print("\n" + "=" * 70)
print("Ridge - PREDICTION METRICS")
print("=" * 70)
print(f"MAE (hours)     = {ridge_mae:.6f}")
print(f"MAPE (%)        = {ridge_mape:.6f}")
print(f"RMSE (hours)    = {ridge_rmse:.6f}")
print(f"R² log space    = {ridge_r2_log:.6f}")
print(f"R² hours        = {ridge_r2_h:.6f}")

# =========================================================
# MODEL EQUATION PARAMETERS
# =========================================================
ridge_intercept = ridge_model.intercept_
ridge_coeffs = ridge_model.coef_

print("\n" + "=" * 70)
print("MODEL EQUATION PARAMETERS")
print("=" * 70)

print(f"Intercept: {ridge_intercept:.10f}")

for name, coef in zip(feature_cols, ridge_coeffs):
    print(f"{name:20s}: {coef:.10f}")

# =========================================================
# PRINT MODEL EQUATION
# =========================================================
equation_terms = [f"({ridge_intercept:.10f})"]

for name, coef in zip(feature_cols, ridge_coeffs):
    sign = "+" if coef >= 0 else "-"
    equation_terms.append(
        f" {sign} ({abs(coef):.10f})*{name}"
    )

equation_str = "log(t_ter_h) = " + "".join(equation_terms)

print("\n" + "=" * 70)
print("MODEL EQUATION")
print("=" * 70)
print(equation_str)

# =========================================================
# ANALYTICAL EQUATION COEFFICIENTS
# =========================================================
# Replace these placeholder coefficients with the actual analytical
# coefficients from your equation/literature/model.
#
# The keys must match the model parameter names exactly.
# =========================================================
analytical_coeffs = {
    "Intercept": 0.5000,
    "temp_k": 0.0001,
    "stress_mpa": -0.0100,
    "inv_temp": 1000.0000,
    "log_stress": -2.5000,
    "stress_temp_ratio": -0.0500,
    "stress_sq": 0.00001,
    "temp_sq": -0.0000001,
}

# =========================================================
# REQUIRED RELATIVE ERROR + METRIC
# =========================================================
# relative error =
# (Analytical equation coefficient - SINDy equation coefficient)
# / Analytical equation coefficient
#
# metric = 1 - relative error
#
# Here, "SINDy equation coefficient" is represented by the model coefficient.
# =========================================================
model_params = {
    "Intercept": ridge_intercept
}

for name, coef in zip(feature_cols, ridge_coeffs):
    model_params[name] = coef

comparison_rows = []

for term, analytical_value in analytical_coeffs.items():
    sindy_value = model_params.get(term, np.nan)

    if pd.isna(sindy_value):
        relative_error = np.nan
        metric = np.nan
        note = "SINDy/model coefficient not found"

    elif analytical_value == 0:
        relative_error = np.nan
        metric = np.nan
        note = "Analytical coefficient is 0, cannot divide by zero"

    else:
        relative_error = (analytical_value - sindy_value) / analytical_value
        metric = 1 - relative_error
        note = ""

    comparison_rows.append({
        "Term": term,
        "Analytical equation coefficient": analytical_value,
        "SINDy equation coefficient": sindy_value,
        "Relative error": relative_error,
        "Metric": metric,
        "Note": note
    })

comparison_df = pd.DataFrame(comparison_rows)

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON: ANALYTICAL vs SINDY/MODEL")
print("=" * 70)
print(comparison_df.to_string(index=False))

# =========================================================
# OVERALL REQUIRED METRIC
# =========================================================
valid_metric_values = comparison_df["Metric"].dropna()

print("\n" + "=" * 70)
print("OVERALL REQUIRED METRIC")
print("=" * 70)

if len(valid_metric_values) > 0:
    overall_metric = valid_metric_values.mean()
    print(f"Average metric = {overall_metric:.10f}")
else:
    overall_metric = np.nan
    print("No valid metric could be computed.")

# =========================================================
# SAVE COEFFICIENT COMPARISON TABLE
# =========================================================
output_dir = os.path.dirname(DASHBOARD_OUT)

comparison_csv = os.path.join(
    output_dir,
    "coefficient_comparison_metric.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)

print(f"\nCoefficient comparison saved to:\n{comparison_csv}")

# =========================================================
# RESIDUALS
# =========================================================
et_residuals = y_test_h - y_pred_et_h
ridge_residuals = y_test_h - y_pred_ridge_h

# =========================================================
# DASHBOARD STYLE
# =========================================================
plt.style.use("dark_background")

FIG_BG = "#0f172a"
AX_BG = "#111827"
GRID_C = "#334155"
TXT_C = "#dbeafe"

BLUE = "#60a5fa"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f97316"
PURPLE = "#a855f7"

# =========================================================
# CREATE DASHBOARD
# =========================================================
fig, axes = plt.subplots(
    3,
    3,
    figsize=(18, 14)
)

fig.patch.set_facecolor(FIG_BG)

for ax in axes.flatten():
    ax.set_facecolor(AX_BG)

    ax.grid(
        True,
        color=GRID_C,
        alpha=0.4
    )

    ax.tick_params(colors=TXT_C)
    ax.xaxis.label.set_color(TXT_C)
    ax.yaxis.label.set_color(TXT_C)
    ax.title.set_color("#93c5fd")

# =========================================================
# 1. EXTRATREES PARITY PLOT
# =========================================================
ax = axes[0, 0]

ax.scatter(
    y_test_h,
    y_pred_et_h,
    alpha=0.7,
    color=GREEN
)

mn = min(
    y_test_h.min(),
    y_pred_et_h.min()
)

mx = max(
    y_test_h.max(),
    y_pred_et_h.max()
)

ax.plot(
    [mn, mx],
    [mn, mx],
    "--",
    color=RED
)

ax.set_title("ExtraTrees: Actual vs Predicted")
ax.set_xlabel("Actual Time (h)")
ax.set_ylabel("Predicted Time (h)")

ax.text(
    0.05,
    0.92,
    f"R² = {et_r2_h:.3f}\nRMSE = {et_rmse:.1f}",
    transform=ax.transAxes,
    fontsize=11,
    color=GREEN,
    fontweight="bold"
)

# =========================================================
# 2. RIDGE PARITY PLOT
# =========================================================
ax = axes[0, 1]

ax.scatter(
    y_test_h,
    y_pred_ridge_h,
    alpha=0.7,
    color=BLUE
)

mn = min(
    y_test_h.min(),
    y_pred_ridge_h.min()
)

mx = max(
    y_test_h.max(),
    y_pred_ridge_h.max()
)

ax.plot(
    [mn, mx],
    [mn, mx],
    "--",
    color=RED
)

ax.set_title("Ridge: Actual vs Predicted")
ax.set_xlabel("Actual Time (h)")
ax.set_ylabel("Predicted Time (h)")

ax.text(
    0.05,
    0.92,
    f"R² = {ridge_r2_h:.3f}\nRMSE = {ridge_rmse:.1f}",
    transform=ax.transAxes,
    fontsize=11,
    color=BLUE,
    fontweight="bold"
)

# =========================================================
# 3. MODEL COMPARISON
# =========================================================
ax = axes[0, 2]

models = ["ExtraTrees", "Ridge"]

rmse_values = [
    et_rmse,
    ridge_rmse
]

r2_values = [
    et_r2_h,
    ridge_r2_h
]

ax.scatter(
    rmse_values,
    r2_values,
    s=200,
    color=[GREEN, BLUE]
)

for model_name, xval, yval in zip(
    models,
    rmse_values,
    r2_values
):
    ax.text(
        xval,
        yval,
        f" {model_name}",
        color=TXT_C
    )

ax.set_title("Model Quality: RMSE vs R²")
ax.set_xlabel("RMSE")
ax.set_ylabel("R²")

# =========================================================
# 4. FEATURE IMPORTANCE
# =========================================================
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": ridge_coeffs,
    "AbsCoefficient": np.abs(ridge_coeffs)
}).sort_values("AbsCoefficient", ascending=True)

ax = axes[1, 0]

top_coef_df = coef_df.tail(10)

ax.barh(
    top_coef_df["Feature"],
    top_coef_df["AbsCoefficient"],
    color=GREEN
)

ax.set_title("Feature Importance\n(|Model coefficient|)")
ax.set_xlabel("Absolute coefficient")

# =========================================================
# 5. MAIN MODEL COEFFICIENTS
# =========================================================
ax = axes[1, 1]

main_features = [
    "temp_k",
    "stress_mpa",
    "inv_temp",
    "log_stress",
    "stress_temp_ratio",
    "stress_sq",
    "temp_sq"
]

main_coefs = [
    ridge_coeffs[
        feature_cols.index(f)
    ]
    for f in main_features
]

colors = [
    GREEN if c >= 0 else RED
    for c in main_coefs
]

ax.bar(
    main_features,
    main_coefs,
    color=colors
)

ax.axhline(
    0,
    color=RED,
    linestyle="--"
)

ax.set_title("Main Equation Coefficients")
ax.tick_params(axis="x", rotation=45)

# =========================================================
# 6. REQUIRED METRIC BY TERM
# =========================================================
ax = axes[1, 2]

plot_df = comparison_df.dropna(subset=["Metric"])

ax.bar(
    plot_df["Term"],
    plot_df["Metric"],
    color=PURPLE
)

ax.axhline(
    0,
    color=RED,
    linestyle="--"
)

ax.set_title("Required Metric by Coefficient")
ax.set_ylabel("Metric = 1 - relative error")
ax.tick_params(axis="x", rotation=45)

# =========================================================
# 7. STRESS VS LOG TIME
# =========================================================
ax = axes[2, 0]

ax.scatter(
    df["stress_mpa"],
    np.log(df["t_ter_h"]),
    color=ORANGE,
    alpha=0.7
)

ax.set_title("Stress vs log(Time)")
ax.set_xlabel("Stress (MPa)")
ax.set_ylabel("log(Time to tertiary creep)")

# =========================================================
# 8. 1/TEMP VS LOG TIME
# =========================================================
ax = axes[2, 1]

ax.scatter(
    1.0 / df["temp_k"],
    np.log(df["t_ter_h"]),
    color=PURPLE,
    alpha=0.7
)

ax.set_title("1/Temperature vs log(Time)")
ax.set_xlabel("1 / Temperature (1/K)")
ax.set_ylabel("log(Time to tertiary creep)")

# =========================================================
# 9. SUMMARY PANEL
# =========================================================
ax = axes[2, 2]
ax.axis("off")

best_model = (
    "ExtraTrees"
    if et_rmse < ridge_rmse
    else "Ridge"
)

summary_text = f"""
Summary

Material:
SS316H Stainless Steel

Samples:
{len(df)}

Features:
{len(feature_cols)}

Heat categories:
{len(heat_cols)}

Best model:
{best_model}

ExtraTrees:
MAE   = {et_mae:.2f} h
MAPE  = {et_mape:.2f} %
RMSE  = {et_rmse:.2f} h
R²    = {et_r2_h:.4f}

Ridge:
MAE   = {ridge_mae:.2f} h
MAPE  = {ridge_mape:.2f} %
RMSE  = {ridge_rmse:.2f} h
R²    = {ridge_r2_h:.4f}

Required metric:
Average = {overall_metric:.4f}

Metric formula:
1 - relative error
"""

ax.text(
    0.02,
    0.98,
    summary_text,
    va="top",
    ha="left",
    fontsize=11,
    color=TXT_C,
    family="monospace"
)

# =========================================================
# DASHBOARD TITLE
# =========================================================
fig.suptitle(
    "SS316H Stainless Steel — Creep Prediction + Required Metric Dashboard",
    fontsize=20,
    color=TXT_C,
    y=0.98
)

# =========================================================
# SAVE DASHBOARD
# =========================================================
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    DASHBOARD_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor=FIG_BG
)

print(f"Saved dashboard: {DASHBOARD_OUT}")

plt.show()