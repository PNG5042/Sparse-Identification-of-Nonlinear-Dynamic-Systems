import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary

# =========================================================
# USER SETTINGS
# =========================================================
MATERIAL_NAME = "Alloy 617"

SPEC_COL = "Specimen_Name"
RATE_COL = "Nominal_Strain_Rate"
STRAIN_COL = "Strain_percent"
STRESS_COL = "Stress_MPa"
TEMP_COL = "Nominal_Temperature_C"
MAT_COL = "Material_Name"

DETAIL_CSV = r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_DETAIL_DATA.csv"
SPECIMEN_CSV = r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_SPECIMEN_LIST.csv"

# ----------------------------
# SINDy settings
# ----------------------------
POLY_DEGREE = 3
THRESHOLD = 1e-6
MIN_POINTS = 50

# Analytical coefficients
# Must match SINDy feature names exactly:
# "1", "x0", "x0^2", "x0^3"
ANALYTICAL_COEFS = {
    # "1": 1500.0,
    # "x0": -100.0,
    # "x0^2": 0.5,
    # "x0^3": -0.01,
}

FEATURE_TO_PLOT = "x0"

# Output files
PARAMS_OUT = "sindy_parameters_temperature.csv"
METRICS_OUT = "sindy_metrics_temperature.csv"
AVG_PARAMS_OUT = "sindy_parameters_temperature_avg.csv"
DASHBOARD_OUT = "alloy617_sindy_dashboard.png"

# =========================================================
# DASHBOARD STYLE
# =========================================================
FIG_BG = "#0b1220"
AX_BG = "#111827"
GRID_C = "#374151"
TXT_C = "#dbeafe"
MUTED_C = "#93c5fd"
ACCENT_RED = "#ef4444"
ACCENT_GREEN = "#22c55e"
ACCENT_BLUE = "#60a5fa"
ACCENT_YELLOW = "#f59e0b"
ACCENT_PURPLE = "#a78bfa"
ACCENT_CYAN = "#22d3ee"

plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": TXT_C,
    "axes.labelcolor": TXT_C,
    "axes.titlecolor": MUTED_C,
    "xtick.color": TXT_C,
    "ytick.color": TXT_C,
    "text.color": TXT_C,
    "grid.color": GRID_C,
    "font.size": 10,
})

# =========================================================
# Helper functions
# =========================================================
def make_increasing(x, y):
    """Ensure x is strictly increasing and aligned with y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 2:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    y = y[idx]

    if len(x) < 2:
        return x, y

    keep = np.ones_like(x, dtype=bool)
    keep[1:] = np.diff(x) > 0
    return x[keep], y[keep]


def get_coefficients(model):
    """Version-safe coefficient extraction."""
    if hasattr(model, "coefficients_"):
        coef = model.coefficients_
    else:
        coef = model.coefficients()

    coef = np.asarray(coef)[0]

    if hasattr(model, "get_feature_names"):
        names = model.get_feature_names()
    else:
        names = [f"f{i}" for i in range(len(coef))]

    return coef, names


def compute_metric(analytical, sindy):
    """
    relative_error = (analytical - sindy) / analytical
    metric = 1 - relative_error
    """
    if analytical == 0:
        return np.nan, np.nan

    relative_error = (analytical - sindy) / analytical
    metric = 1 - relative_error
    return relative_error, metric


def print_parameters(names, coefs):
    print("\nModel equation parameters:")
    for n, c in zip(names, coefs):
        print(f"{n:>8s} : {c:.8g}")


def numerical_derivative(x, y):
    """Compute dy/dx."""
    if len(x) < 3:
        return np.array([]), np.array([])
    dydx = np.gradient(y, x)
    return x, dydx


def sindy_predict_derivative(stress, c0, c1, c2, c3):
    """dσ/dε = c0 + c1σ + c2σ² + c3σ³"""
    stress = np.asarray(stress, dtype=float)
    return c0 + c1 * stress + c2 * stress**2 + c3 * stress**3


def equation_string(row):
    c0 = row.get("coef_1", 0.0)
    c1 = row.get("coef_x0", 0.0)
    c2 = row.get("coef_x0^2", 0.0)
    c3 = row.get("coef_x0^3", 0.0)

    return (
        f"dσ/dε = {c0:.2f} "
        f"{c1:+.2f}σ "
        f"{c2:+.2f}σ² "
        f"{c3:+.4f}σ³"
    )


# =========================================================
# LOAD DATA
# =========================================================
detail = pd.read_csv(DETAIL_CSV)
spec = pd.read_csv(SPECIMEN_CSV)

required_detail = [SPEC_COL, RATE_COL, STRAIN_COL, STRESS_COL]
required_spec = [SPEC_COL, MAT_COL, TEMP_COL]

missing_detail = [c for c in required_detail if c not in detail.columns]
missing_spec = [c for c in required_spec if c not in spec.columns]

if missing_detail:
    raise KeyError(f"DETAIL file missing columns: {missing_detail}")
if missing_spec:
    raise KeyError(f"SPECIMEN_LIST file missing columns: {missing_spec}")

detail = detail[required_detail].copy()
spec = spec[required_spec].copy()

for col in [RATE_COL, STRAIN_COL, STRESS_COL]:
    detail[col] = pd.to_numeric(detail[col], errors="coerce")
spec[TEMP_COL] = pd.to_numeric(spec[TEMP_COL], errors="coerce")

detail = detail.dropna(subset=required_detail)
spec = spec.dropna(subset=required_spec)

# Filter invalid values
detail = detail[(detail[RATE_COL] > 0) & (detail[STRESS_COL] > 0)]
detail = detail[detail[STRAIN_COL] >= 0]

# Deduplicate specimen metadata before merge
spec = spec.drop_duplicates(subset=[SPEC_COL], keep="first")

merged = detail.merge(spec, on=SPEC_COL, how="left", validate="many_to_one")
merged = merged[merged[MAT_COL] == MATERIAL_NAME].copy()
merged = merged.dropna(subset=[TEMP_COL, RATE_COL, STRAIN_COL, STRESS_COL])

print("Dataset summary")
print("Rows:", len(merged))
print("Specimens:", merged[SPEC_COL].nunique())
print("Temperatures:", sorted(merged[TEMP_COL].unique()))
print("Strain rates:", sorted(merged[RATE_COL].unique()))

# =========================================================
# FIT SINDy MODELS PER SPECIMEN + TEMPERATURE + RATE
# =========================================================
library = PolynomialLibrary(degree=POLY_DEGREE, include_interaction=False)
optimizer = ps.STLSQ(threshold=THRESHOLD)

param_rows = []
metric_rows = []

for (specimen, T, rate), g in merged.groupby([SPEC_COL, TEMP_COL, RATE_COL], sort=True):
    g = g.sort_values(STRAIN_COL)

    strain = g[STRAIN_COL].to_numpy(float)
    stress = g[STRESS_COL].to_numpy(float)

    strain, stress = make_increasing(strain, stress)

    if len(stress) < MIN_POINTS:
        print(f"\n[SKIP] Specimen={specimen}, Temp={T}, Rate={rate}: only {len(stress)} points after cleaning")
        continue

    X = stress.reshape(-1, 1)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    try:
        model.fit(X, t=strain)
    except Exception as e:
        print(f"\n[WARN] Fit failed for Specimen={specimen}, Temp={T}, Rate={rate}: {e}")
        continue

    print("\n===================================================")
    print(f"SINDy model | Specimen={specimen} | Temp={T} C | Rate={rate}")
    model.print()

    coefs, names = get_coefficients(model)
    print_parameters(names, coefs)

    coef_map = dict(zip(names, coefs))

    row = {
        "Specimen_Name": specimen,
        "Temperature": T,
        "Rate": rate,
    }
    for k, v in coef_map.items():
        row[f"coef_{k}"] = v

    param_rows.append(row)

    if ANALYTICAL_COEFS:
        for feat, a_true in ANALYTICAL_COEFS.items():
            if a_true == 0:
                continue

            s_coef = coef_map.get(feat, 0.0)
            rel, metric = compute_metric(a_true, s_coef)

            print(
                f"Metric {feat}: analytical={a_true}, "
                f"sindy={s_coef}, relative_error={rel}, metric={metric}"
            )

            metric_rows.append({
                "Specimen_Name": specimen,
                "Temperature": T,
                "Rate": rate,
                "feature": feat,
                "analytical": a_true,
                "sindy": s_coef,
                "relative_error": rel,
                "metric": metric,
            })

# =========================================================
# SAVE RESULTS
# =========================================================
params = pd.DataFrame(param_rows)

if params.empty:
    raise RuntimeError("No SINDy parameter rows were produced.")

params.to_csv(PARAMS_OUT, index=False)
print(f"\nSaved: {PARAMS_OUT}")

metrics = pd.DataFrame(metric_rows)

if not metrics.empty:
    metrics.to_csv(METRICS_OUT, index=False)
    print(f"Saved: {METRICS_OUT}")
else:
    print("\nNo metric results. Fill ANALYTICAL_COEFS with nonzero values to compute metrics.")

coef_cols = [c for c in params.columns if c.startswith("coef_")]
avg_params = (
    params.groupby(["Temperature", "Rate"], as_index=False)[coef_cols]
    .mean()
)
avg_params.to_csv(AVG_PARAMS_OUT, index=False)
print(f"Saved: {AVG_PARAMS_OUT}")


# =========================================================
# BUILD DASHBOARD DATA
# =========================================================
dash_rows = []

# use averaged coefficients for dashboard
for _, prow in avg_params.iterrows():
    T = prow["Temperature"]
    rate = prow["Rate"]

    sub = merged[(merged[TEMP_COL] == T) & (merged[RATE_COL] == rate)].copy()
    if sub.empty:
        continue

    sub = sub.sort_values(STRAIN_COL)
    strain, stress = make_increasing(
        sub[STRAIN_COL].to_numpy(),
        sub[STRESS_COL].to_numpy()
    )

    if len(strain) < 5:
        continue

    _, actual_d = numerical_derivative(strain, stress)
    pred_d = sindy_predict_derivative(
        stress,
        prow.get("coef_1", 0.0),
        prow.get("coef_x0", 0.0),
        prow.get("coef_x0^2", 0.0),
        prow.get("coef_x0^3", 0.0),
    )

    n = min(len(actual_d), len(pred_d))
    actual_d = actual_d[:n]
    pred_d = pred_d[:n]

    if n == 0:
        continue

    mae = np.mean(np.abs(actual_d - pred_d))
    rmse = np.sqrt(np.mean((actual_d - pred_d) ** 2))
    ss_res = np.sum((actual_d - pred_d) ** 2)
    ss_tot = np.sum((actual_d - np.mean(actual_d)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    active_terms = int(np.sum(np.abs([
        prow.get("coef_1", 0.0),
        prow.get("coef_x0", 0.0),
        prow.get("coef_x0^2", 0.0),
        prow.get("coef_x0^3", 0.0),
    ]) > 1e-8))

    dash_rows.append({
        "Temperature": T,
        "Rate": rate,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "active_terms": active_terms,
        "coef_1": prow.get("coef_1", 0.0),
        "coef_x0": prow.get("coef_x0", 0.0),
        "coef_x0^2": prow.get("coef_x0^2", 0.0),
        "coef_x0^3": prow.get("coef_x0^3", 0.0),
        "equation": equation_string(prow),
    })

dash_df = pd.DataFrame(dash_rows)
if dash_df.empty:
    raise RuntimeError("No dashboard rows were created.")

# =========================================================
# DASHBOARD TABLES
# =========================================================
coef_cols = ["coef_1", "coef_x0", "coef_x0^2", "coef_x0^3"]
mean_abs_coef = dash_df[coef_cols].abs().mean().sort_values(ascending=False)

coef_by_temp = dash_df.groupby("Temperature")[coef_cols].mean()
rmse_by_temp = dash_df.groupby("Temperature")["RMSE"].mean()
r2_by_temp = dash_df.groupby("Temperature")["R2"].mean()
coef_stability = dash_df.groupby("Rate")[coef_cols].std().fillna(0)
coef_stability_score = coef_stability.mean(axis=1)

temp_sorted = sorted(dash_df["Temperature"].unique())
rate_sorted = sorted(dash_df["Rate"].unique())

best_row = dash_df.sort_values("RMSE").iloc[0]
worst_row = dash_df.sort_values("RMSE").iloc[-1]

# =========================================================
# DASHBOARD FIGURE
# =========================================================
fig = plt.figure(figsize=(18, 14), facecolor=FIG_BG)
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.28)

# 1) coef_x0 vs temperature
ax1 = fig.add_subplot(gs[0, 0])
for rate in rate_sorted:
    sub = dash_df[dash_df["Rate"] == rate]
    ax1.scatter(sub["Temperature"], sub["coef_x0"], s=70, alpha=0.9, label=f"{rate:.0e}")
ax1.set_title("Coefficient x0 vs Temperature")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Coefficient x0")
ax1.grid(True, alpha=0.4)
ax1.legend(title="Rate", fontsize=8, title_fontsize=9, facecolor=AX_BG, edgecolor=GRID_C)

# 2) actual vs predicted derivative for best model
ax2 = fig.add_subplot(gs[0, 1])
sub_best = merged[
    (merged[TEMP_COL] == best_row["Temperature"]) &
    (merged[RATE_COL] == best_row["Rate"])
].sort_values(STRAIN_COL)

strain_b, stress_b = make_increasing(
    sub_best[STRAIN_COL].to_numpy(),
    sub_best[STRESS_COL].to_numpy()
)
_, actual_b = numerical_derivative(strain_b, stress_b)
pred_b = sindy_predict_derivative(
    stress_b,
    best_row["coef_1"],
    best_row["coef_x0"],
    best_row["coef_x0^2"],
    best_row["coef_x0^3"],
)

n = min(len(actual_b), len(pred_b))
actual_b = actual_b[:n]
pred_b = pred_b[:n]

ax2.scatter(actual_b, pred_b, color=ACCENT_GREEN, s=35, alpha=0.8)
mn = min(actual_b.min(), pred_b.min())
mx = max(actual_b.max(), pred_b.max())
ax2.plot([mn, mx], [mn, mx], "--", color=ACCENT_RED, linewidth=1.5)
ax2.set_title(f"Best Model: Actual vs Predicted\nT={best_row['Temperature']:.0f}°C, Rate={best_row['Rate']:.0e}")
ax2.set_xlabel("Actual dσ/dε")
ax2.set_ylabel("Predicted dσ/dε")
ax2.grid(True, alpha=0.4)
ax2.text(
    0.03, 0.95, f"R² = {best_row['R2']:.4f}",
    transform=ax2.transAxes,
    ha="left", va="top",
    color=ACCENT_GREEN, fontsize=10, fontweight="bold"
)

# 3) RMSE vs R2
ax3 = fig.add_subplot(gs[0, 2])
for rate in rate_sorted:
    sub = dash_df[dash_df["Rate"] == rate]
    ax3.scatter(sub["RMSE"], sub["R2"], s=80, alpha=0.9, label=f"{rate:.0e}")
ax3.set_title("Model Quality: RMSE vs R²")
ax3.set_xlabel("RMSE")
ax3.set_ylabel("R²")
ax3.grid(True, alpha=0.4)
ax3.legend(title="Rate", fontsize=8, title_fontsize=9, facecolor=AX_BG, edgecolor=GRID_C)

# 4) mean absolute coefficient importance
ax4 = fig.add_subplot(gs[1, 0])
ax4.barh(mean_abs_coef.index, mean_abs_coef.values, color=ACCENT_GREEN, alpha=0.85)
ax4.set_title("Feature Importance\n(mean absolute coefficient)")
ax4.set_xlabel("Mean |coefficient|")
ax4.grid(True, axis="x", alpha=0.4)

# 5) equation structure by temperature
ax5 = fig.add_subplot(gs[1, 1])
xpos = np.arange(len(temp_sorted))
width = 0.18
temp_plot = coef_by_temp.loc[temp_sorted]

ax5.bar(xpos - 1.5 * width, temp_plot["coef_1"], width, label="1", color=ACCENT_BLUE, alpha=0.85)
ax5.bar(xpos - 0.5 * width, temp_plot["coef_x0"], width, label="x0", color=ACCENT_GREEN, alpha=0.85)
ax5.bar(xpos + 0.5 * width, temp_plot["coef_x0^2"], width, label="x0^2", color=ACCENT_YELLOW, alpha=0.85)
ax5.bar(xpos + 1.5 * width, temp_plot["coef_x0^3"], width, label="x0^3", color=ACCENT_PURPLE, alpha=0.85)

ax5.axhline(0, linestyle="--", color=ACCENT_RED, linewidth=1)
ax5.set_xticks(xpos)
ax5.set_xticklabels([str(int(t)) for t in temp_sorted])
ax5.set_title("Equation Structure by Temperature")
ax5.set_xlabel("Temperature (°C)")
ax5.set_ylabel("Mean coefficient")
ax5.legend(fontsize=8, facecolor=AX_BG, edgecolor=GRID_C)
ax5.grid(True, axis="y", alpha=0.4)

# 6) RMSE and R2 vs temperature
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(temp_sorted, rmse_by_temp.loc[temp_sorted], marker="o", linewidth=2, color=ACCENT_BLUE)
ax6.set_xlabel("Temperature (°C)")
ax6.set_ylabel("RMSE", color=ACCENT_BLUE)
ax6.tick_params(axis="y", labelcolor=ACCENT_BLUE)
ax6.grid(True, alpha=0.4)
ax6.set_title("Prediction Error vs Temperature")

ax6b = ax6.twinx()
ax6b.plot(temp_sorted, r2_by_temp.loc[temp_sorted], marker="s", linestyle="--", linewidth=2, color=ACCENT_RED)
ax6b.set_ylabel("R²", color=ACCENT_RED)
ax6b.tick_params(axis="y", labelcolor=ACCENT_RED)

# 7) coefficient stability vs rate
ax7 = fig.add_subplot(gs[2, 0])
rate_labels = [f"{r:.0e}" for r in coef_stability_score.index]
ax7.plot(rate_labels, coef_stability_score.values, marker="o", linewidth=2, color=ACCENT_GREEN)
ax7.set_title("Coefficient Stability vs Rate")
ax7.set_xlabel("Rate")
ax7.set_ylabel("Mean std of coefficients")
ax7.grid(True, alpha=0.4)

# 8) complexity vs prediction error
ax8 = fig.add_subplot(gs[2, 1])
for rate in rate_sorted:
    sub = dash_df[dash_df["Rate"] == rate]
    ax8.scatter(sub["active_terms"], sub["RMSE"], s=90, alpha=0.9, label=f"{rate:.0e}")
ax8.set_title("Model Complexity vs Prediction Error")
ax8.set_xlabel("Active terms")
ax8.set_ylabel("RMSE")
ax8.grid(True, alpha=0.4)
ax8.annotate("best", (best_row["active_terms"], best_row["RMSE"]),
             xytext=(8, -10), textcoords="offset points", color=ACCENT_GREEN)
ax8.annotate("worst", (worst_row["active_terms"], worst_row["RMSE"]),
             xytext=(8, 8), textcoords="offset points", color=ACCENT_RED)

# 9) summary text panel
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
summary_lines = [
    "Summary",
    "",
    f"Material: {MATERIAL_NAME}",
    f"Models in dashboard: {len(dash_df)}",
    f"Temperatures: {len(temp_sorted)}",
    f"Rates: {len(rate_sorted)}",
    "",
    f"Best RMSE: {best_row['RMSE']:.4f}",
    f"Best R²: {best_row['R2']:.4f}",
    f"Best T, Rate: {best_row['Temperature']:.0f}°C, {best_row['Rate']:.0e}",
    "",
    f"Worst RMSE: {worst_row['RMSE']:.4f}",
    f"Worst R²: {worst_row['R2']:.4f}",
    f"Worst T, Rate: {worst_row['Temperature']:.0f}°C, {worst_row['Rate']:.0e}",
    "",
    "Best Equation:",
    best_row["equation"],
    "",
    f"Mean RMSE: {dash_df['RMSE'].mean():.4f}",
    f"Mean R²: {dash_df['R2'].mean():.4f}",
    f"Mean active terms: {dash_df['active_terms'].mean():.2f}",
]
ax9.text(
    0.02, 0.98,
    "\n".join(summary_lines),
    va="top", ha="left",
    family="monospace",
    fontsize=10,
    color=TXT_C
)

fig.suptitle(
    "Alloy 617 — SINDy Equation Discovery Dashboard",
    fontsize=20,
    color=TXT_C,
    y=0.98
)

plt.savefig(DASHBOARD_OUT, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved: {DASHBOARD_OUT}")

plt.show()
