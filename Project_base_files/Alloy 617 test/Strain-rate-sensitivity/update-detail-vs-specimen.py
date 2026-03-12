import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ----------------------------
# SINDy Settings
# ----------------------------
POLY_DEGREE = 3
THRESHOLD = 1e-6
MIN_POINTS_SEGMENT = 50

# Optional smoothing for stress (set to odd int like 11, 21, or None)
SMOOTH_WINDOW = None

# ----------------------------
# Analytical coefficients (YOU MUST EDIT THESE)
# Keys MUST match SINDy feature names: usually "1", "x0", "x0^2", "x0^3", ...
# Example:
# ANALYTICAL_COEFS = {"1": 1500.0, "x0": -100.0, "x0^2": 0.5, "x0^3": -0.01}
# ----------------------------
ANALYTICAL_COEFS = {
    # "1": ...,
    # "x0": ...,
    # "x0^2": ...,
    # "x0^3": ...,
}

# Which feature to plot metric vs temperature (must exist in ANALYTICAL_COEFS)
FEATURE_TO_PLOT = "x0"


# =========================================================
# HELPERS
# =========================================================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window is None:
        return y
    if window < 3 or window % 2 == 0:
        raise ValueError("SMOOTH_WINDOW must be an odd integer >= 3, or None.")
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def make_strictly_increasing(x: np.ndarray, y: np.ndarray):
    """
    Ensure x is strictly increasing by:
      - removing duplicate x
      - removing non-increasing steps
    """
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    y = y[idx]
    if x.size < 2:
        return x, y
    keep = np.ones_like(x, dtype=bool)
    keep[1:] = np.diff(x) > 0
    return x[keep], y[keep]


def get_sindy_coeffs_and_feature_names(model: ps.SINDy):
    """
    Version-safe extraction of coefficients and feature names.
    Assumes 1-state system.
    """
    # coefficients
    if hasattr(model, "coefficients_"):
        coef_mat = np.asarray(model.coefficients_)
    else:
        c = model.coefficients
        coef_mat = np.asarray(c() if callable(c) else c)

    # feature names
    if hasattr(model, "get_feature_names"):
        feat_names = model.get_feature_names()
    else:
        # fallback
        feat_names = [f"f{i}" for i in range(coef_mat.shape[1])]

    # 1 state -> row 0
    coefs = coef_mat[0, :].astype(float)
    return coefs, feat_names


def relative_error_and_metric(analytical: float, sindy: float):
    """
    REQUIRED by mentor:
      relative_error = (Analytical - SINDy) / Analytical
      metric = 1 - relative_error

    If analytical == 0 -> undefined (division by zero), returns NaN.
    """
    if analytical == 0:
        return np.nan, np.nan
    rel = (analytical - sindy) / analytical
    metric = 1.0 - rel
    return float(rel), float(metric)


def print_model_parameters(feat_names, coefs):
    """
    Print model parameters as feature -> coefficient.
    """
    print("\nModel equation parameters (feature -> coefficient):")
    for f, c in zip(feat_names, coefs):
        print(f"  {f:>8s} : {c:.8g}")


# =========================================================
# 1) LOAD + MERGE DATA
# =========================================================
df_detail = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_DETAIL_DATA.csv")
df_spec = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_SPECIMEN_LIST.csv")

needed_detail = [SPEC_COL, RATE_COL, STRAIN_COL, STRESS_COL]
needed_spec = [SPEC_COL, MAT_COL, TEMP_COL]

missing_detail = [c for c in needed_detail if c not in df_detail.columns]
missing_spec = [c for c in needed_spec if c not in df_spec.columns]
if missing_detail:
    raise KeyError(f"DETAIL missing columns: {missing_detail}")
if missing_spec:
    raise KeyError(f"SPECIMEN_LIST missing columns: {missing_spec}")

df_detail = df_detail[needed_detail].copy()
df_spec = df_spec[needed_spec].copy()

# numeric cleanup
for c in [RATE_COL, STRAIN_COL, STRESS_COL]:
    df_detail[c] = pd.to_numeric(df_detail[c], errors="coerce")
df_detail = df_detail.dropna(subset=[SPEC_COL, RATE_COL, STRAIN_COL, STRESS_COL])

df_spec[TEMP_COL] = pd.to_numeric(df_spec[TEMP_COL], errors="coerce")

# basic filtering
df_detail = df_detail[(df_detail[RATE_COL] > 0) & (df_detail[STRESS_COL] > 0)]
df_detail = df_detail[df_detail[STRAIN_COL] >= 0]

# ensure 1 metadata row per specimen
df_spec = (df_spec.dropna(subset=[TEMP_COL])
                 .drop_duplicates(subset=[SPEC_COL], keep="first"))

merged = df_detail.merge(df_spec, on=SPEC_COL, how="left", validate="many_to_one")

# filter to material
merged = merged[merged[MAT_COL] == MATERIAL_NAME].copy()
if merged.empty:
    raise RuntimeError(f"No data found for material {MATERIAL_NAME!r}")

print("=== MERGED summary ===")
print(f"Rows: {len(merged):,}")
print(f"Specimens: {merged[SPEC_COL].nunique()}")
print("Temperatures:", sorted(merged[TEMP_COL].dropna().unique()))
print("Strain rates:", sorted(merged[RATE_COL].unique()))


# =========================================================
# 2) FIT SINDy PER (Specimen, Rate)
#    Model: dσ/dε = f(σ), treat ε as "t"
# =========================================================
library = PolynomialLibrary(degree=POLY_DEGREE, include_interaction=False, include_bias=True)
optimizer = ps.STLSQ(threshold=THRESHOLD)

param_rows = []
metric_rows = []

for (specimen, rate), g in merged.groupby([SPEC_COL, RATE_COL], sort=False):
    g = g.sort_values(STRAIN_COL)

    eps = g[STRAIN_COL].to_numpy(float)
    sig = g[STRESS_COL].to_numpy(float)

    # make eps strictly increasing
    eps, sig = make_strictly_increasing(eps, sig)
    if sig.size < MIN_POINTS_SEGMENT:
        continue

    # optional smoothing
    sig_used = moving_average(sig, SMOOTH_WINDOW) if SMOOTH_WINDOW is not None else sig

    # SINDy state vector
    X = sig_used.reshape(-1, 1)

    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    try:
        model.fit(X, t=eps)
    except Exception as e:
        print(f"[WARN] SINDy fit failed for specimen={specimen}, rate={rate}: {e}")
        continue

    Tval = g[TEMP_COL].iloc[0]

    print("\n" + "=" * 80)
    print(f"SINDy model for specimen={specimen} | rate={rate:.0e} 1/s | T={Tval}")
    model.print()

    coefs, feat_names = get_sindy_coeffs_and_feature_names(model)
    print_model_parameters(feat_names, coefs)

    # store parameters row (CSV)
    coef_map = {f: float(c) for f, c in zip(feat_names, coefs)}
    row = {
        SPEC_COL: specimen,
        RATE_COL: float(rate),
        TEMP_COL: float(Tval) if pd.notna(Tval) else np.nan,
        MAT_COL: MATERIAL_NAME,
    }
    for f, c in coef_map.items():
        row[f"coef_{f}"] = c
    param_rows.append(row)

    # compute metric rows (mentor requirement)
    if ANALYTICAL_COEFS:
        for f, a_true in ANALYTICAL_COEFS.items():
            # if analytical is 0, formula divides by zero -> skip
            if a_true == 0:
                continue

            a_sindy = coef_map.get(f, 0.0)  # 0 if SINDy doesn't include that term
            rel, metric = relative_error_and_metric(a_true, a_sindy)

            print(
                f"  Metric for {f}: analytical={a_true:.8g}, sindy={a_sindy:.8g}, "
                f"relative_error={rel:.8g}, metric={metric:.8g}"
            )

            metric_rows.append({
                SPEC_COL: specimen,
                RATE_COL: float(rate),
                TEMP_COL: float(Tval) if pd.notna(Tval) else np.nan,
                MAT_COL: MATERIAL_NAME,
                "feature": f,
                "analytical_coef": float(a_true),
                "sindy_coef": float(a_sindy),
                "relative_error": rel,
                "metric": metric,
            })

# =========================================================
# 3) SAVE OUTPUTS
# =========================================================
params_df = pd.DataFrame(param_rows)
params_out = "sindy_model_parameters.csv"
params_df.to_csv(params_out, index=False)
print(f"\nSaved: {params_out} (rows={len(params_df)})")

metrics_df = pd.DataFrame(metric_rows)
metrics_out = "sindy_vs_analytical_metrics.csv"

if metrics_df.empty:
    print("\nNo metric rows saved.")
    print("Check:")
    print(" - ANALYTICAL_COEFS is empty (you must fill it), OR")
    print(" - your analytical coefficients include zeros (those terms are skipped), OR")
    print(" - feature names mismatch (print feature names above and match keys).")
else:
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved: {metrics_out} (rows={len(metrics_df)})")

# =========================================================
# 4) PLOT metric vs temperature for one feature 
# =========================================================
if not metrics_df.empty and FEATURE_TO_PLOT in metrics_df["feature"].unique():
    sub = metrics_df[metrics_df["feature"] == FEATURE_TO_PLOT].dropna(subset=["metric", TEMP_COL])

    if not sub.empty:
        plt.figure(figsize=(7, 5))
        plt.scatter(sub[TEMP_COL], sub["metric"])
        plt.xlabel("Temperature (°C)")
        plt.ylabel("metric = 1 - relative_error")
        plt.title(f"Metric vs Temperature for feature {FEATURE_TO_PLOT} (Material: {MATERIAL_NAME})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()