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
# SINDy settings
# ----------------------------
POLY_DEGREE = 3
THRESHOLD = 1e-6
MIN_POINTS = 50

# Analytical coefficients
# MUST match SINDy feature names
ANALYTICAL_COEFS = {
# example: "1":1500, "x0":-100, "x0^2":0.5, "x0^3":-0.01
}

FEATURE_TO_PLOT = "x0"


# =========================================================
# Helper functions
# =========================================================
def make_increasing(x, y):
    """Ensure x strictly increasing"""
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    y = y[idx]

    keep = np.ones_like(x, dtype=bool)
    keep[1:] = np.diff(x) > 0
    return x[keep], y[keep]


def get_coefficients(model):

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

    if analytical == 0:
        return np.nan, np.nan

    relative_error = (analytical - sindy) / analytical
    metric = 1 - relative_error

    return relative_error, metric


def print_parameters(names, coefs):

    print("\nModel equation parameters:")
    for n, c in zip(names, coefs):
        print(f"{n:>8s} : {c:.8g}")


# =========================================================
# LOAD DATA
# =========================================================
detail = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_DETAIL_DATA.csv")
spec = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_SPECIMEN_LIST.csv")

detail = detail[[SPEC_COL, RATE_COL, STRAIN_COL, STRESS_COL]]
spec = spec[[SPEC_COL, MAT_COL, TEMP_COL]]

merged = detail.merge(spec, on=SPEC_COL, how="left")

merged = merged[merged[MAT_COL] == MATERIAL_NAME]

merged[STRAIN_COL] = pd.to_numeric(merged[STRAIN_COL], errors="coerce")
merged[STRESS_COL] = pd.to_numeric(merged[STRESS_COL], errors="coerce")
merged[RATE_COL] = pd.to_numeric(merged[RATE_COL], errors="coerce")

merged = merged.dropna()

print("Dataset summary")
print("Rows:", len(merged))
print("Temperatures:", sorted(merged[TEMP_COL].unique()))
print("Strain rates:", sorted(merged[RATE_COL].unique()))


# =========================================================
# FIT SINDy MODELS PER TEMPERATURE
# =========================================================
library = PolynomialLibrary(degree=POLY_DEGREE, include_interaction=False)
optimizer = ps.STLSQ(threshold=THRESHOLD)

param_rows = []
metric_rows = []

for (T, rate), g in merged.groupby([TEMP_COL, RATE_COL]):

    g = g.sort_values(STRAIN_COL)

    strain = g[STRAIN_COL].to_numpy(float)
    stress = g[STRESS_COL].to_numpy(float)

    strain, stress = make_increasing(strain, stress)

    if len(stress) < MIN_POINTS:
        continue

    X = stress.reshape(-1,1)

    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    try:
        model.fit(X, t=strain)
    except:
        continue

    print("\n===================================================")
    print(f"SINDy model | Temp={T} C | Rate={rate}")
    model.print()

    coefs, names = get_coefficients(model)

    print_parameters(names, coefs)

    coef_map = dict(zip(names, coefs))

    row = {
        "Temperature":T,
        "Rate":rate
    }

    for k,v in coef_map.items():
        row[f"coef_{k}"] = v

    param_rows.append(row)

    # compute metric
    if ANALYTICAL_COEFS:

        for feat, a_true in ANALYTICAL_COEFS.items():

            if a_true == 0:
                continue

            s_coef = coef_map.get(feat,0)

            rel, metric = compute_metric(a_true, s_coef)

            print(f"Metric {feat}: analytical={a_true}, sindy={s_coef}, metric={metric}")

            metric_rows.append({
                "Temperature":T,
                "Rate":rate,
                "feature":feat,
                "analytical":a_true,
                "sindy":s_coef,
                "relative_error":rel,
                "metric":metric
            })


# =========================================================
# SAVE RESULTS
# =========================================================
params = pd.DataFrame(param_rows)
params.to_csv("sindy_parameters_temperature.csv", index=False)

metrics = pd.DataFrame(metric_rows)

if not metrics.empty:
    metrics.to_csv("sindy_metrics_temperature.csv", index=False)
    print("\nSaved metric results")

else:
    print("\nNo metric results (fill ANALYTICAL_COEFS)")


# =========================================================
# PLOT METRIC VS TEMPERATURE
# =========================================================
if not metrics.empty:

    sub = metrics[metrics["feature"] == FEATURE_TO_PLOT]

    plt.figure(figsize=(7,5))
    plt.scatter(sub["Temperature"], sub["metric"])
    plt.xlabel("Temperature (C)")
    plt.ylabel("Metric = 1 - relative error")
    plt.title("Metric vs Temperature")
    plt.grid(True)
    plt.show()