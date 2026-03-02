import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps  
from pysindy.feature_library import PolynomialLibrary  
from scipy.optimize import curve_fit  
from sklearn.metrics import r2_score  

# AGG: how to combine multiple specimens at the same strain rate (median vs mean).
AGG = "median"   

# MIN_RATES_FOR_FIT: you need at least 2 distinct strain rates to fit a line in log–log space.
MIN_RATES_FOR_FIT = 2          

# MIN_SPECIMENS_PER_RATE: desired minimum number of specimens per strain rate.
MIN_SPECIMENS_PER_RATE = 2

# SAFE_MIN_PCT, SAFE_MAX_PCT: a “safe” strain range (in percent) where curves are reliable for interpolation.
SAFE_MIN_PCT = 0.20
SAFE_MAX_PCT = 2.00

# REQUESTED_TARGET_STRAIN_PCT: user’s preferred target strain (5%).
REQUESTED_TARGET_STRAIN_PCT = 5.0  

# DROP_NEGATIVE_STRAIN: flag saying we don’t want negative strain points.
DROP_NEGATIVE_STRAIN = True


# =========================================================
# LOAD + CLEAN
# =========================================================
# Reads the CSV file.
# Keeps only these columns: specimen ID, strain rate, strain (%), stress (MPa).
df = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_DETAIL_DATA.csv")
needed = ["Specimen_Name", "Nominal_Strain_Rate", "Strain_percent", "Stress_MPa"]
df = df[needed].copy()

# Forces these fields to numeric, turning invalid values into NaN.
# Drops any rows with NaN in those columns.
df["Nominal_Strain_Rate"] = pd.to_numeric(df["Nominal_Strain_Rate"], errors="coerce")
df["Strain_percent"] = pd.to_numeric(df["Strain_percent"], errors="coerce")
df["Stress_MPa"] = pd.to_numeric(df["Stress_MPa"], errors="coerce")
df = df.dropna(subset=needed)

# Keeps only positive strain rates, positive stress, and non-negative strain.
df = df[(df["Nominal_Strain_Rate"] > 0) & (df["Stress_MPa"] > 0)].copy()
df = df[df["Strain_percent"] >= 0].copy()

print("=== Data summary ===")
print(f"Rows: {len(df):,}")
print(f"Strain range (%): {df["Strain_percent"].min():.6g} to {df["Strain_percent"].max():.6g}")
print(f"Unique specimens: {df["Specimen_Name"].nunique()}")
print(f"Unique strain rates: {df["Nominal_Strain_Rate"].nunique()}")


# =========================================================
# CORE FUNCTIONS
# =========================================================
def interp_stress_at_strain(curve_df: pd.DataFrame, target_strain_pct: float) -> float:
    # Extracts arrays of strain (x) and stress (y).
    x = curve_df["Strain_percent"].to_numpy(float)
    y = curve_df["Stress_MPa"].to_numpy(float)

    # Sorts by strain.
    # Removes duplicate strain values, keeping the first.
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]

    # Needs at least 2 points.
    # Target strain must be within the observed strain range for that curve.
    if x_u.size < 2:
        raise ValueError("Not enough points to interpolate.")
    if target_strain_pct < x_u.min() or target_strain_pct > x_u.max():
        raise ValueError("Target strain outside curve range.")

    return float(np.interp(target_strain_pct, x_u, y_u))


def flow_stress_by_rate_at_strain(df_all: pd.DataFrame, target_strain_pct: float, agg: str = "median", min_specimens_per_rate: int = 1,) -> pd.DataFrame:
    rows = []
    # Loops over each specimen × strain rate combination.
    for (spec, rate), g in df_all.groupby(["Specimen_Name", "Nominal_Strain_Rate"], sort=False):
        try:
            sigma = interp_stress_at_strain(g, target_strain_pct)
            rows.append((float(rate), float(sigma), spec))
        except ValueError:
            continue

    if not rows:
        raise ValueError("No curves cover the target strain.")

    tmp = pd.DataFrame(rows, columns=["Nominal_Strain_Rate", "sigma_at_target", "Specimen_Name"])

    if tmp.empty:
        raise ValueError("All rates removed by min_specimens_per_rate filter.")

    if agg == "median":
        s = tmp.groupby("Nominal_Strain_Rate")["sigma_at_target"].median()
    elif agg == "mean":
        s = tmp.groupby("Nominal_Strain_Rate")["sigma_at_target"].mean()
    else:
        raise ValueError("agg must be 'median' or 'mean'.")
    
    n_spec = tmp.groupby("Nominal_Strain_Rate")["Specimen_Name"].nunique()

    out = pd.DataFrame({
        "Nominal_Strain_Rate": s.index.to_numpy(float),
        "flow_stress_MPa": s.to_numpy(float),
        "n_specimens": n_spec.to_numpy(int),
    }).sort_values("Nominal_Strain_Rate").reset_index(drop=True)

    return out

# Because the relationship is nonlinear, the code converts it to a log–log linear equation: 
# ln(σ) = m ln(ε˙) + ln(K)
# Slope = m & Intercept = ln(K)
def fit_m_from_rate_stress(rate_stress: pd.DataFrame):
    rates = rate_stress["Nominal_Strain_Rate"].to_numpy(float)
    sigmas = rate_stress["flow_stress_MPa"].to_numpy(float)

    x = np.log(rates)
    y = np.log(sigmas)

    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(m), float(b), float(r2)


def compute_m_at_strain(df_all: pd.DataFrame, target_strain_pct: float, agg: str, min_specimens_per_rate: int):
    rs = flow_stress_by_rate_at_strain(
        df_all, target_strain_pct, agg=agg, min_specimens_per_rate=min_specimens_per_rate
    )
    if len(rs) < MIN_RATES_FOR_FIT:
        raise ValueError("Need at least 2 strain rates to fit m.")
    m, b, r2 = fit_m_from_rate_stress(rs)
    return m, b, r2, rs


# =========================================================
# TARGET SELECTION (USABLE-RATE AWARE)
# =========================================================
def choose_target_strain_usable_rate_aware(
    df_all: pd.DataFrame,
    requested: float,
    safe_min: float,
    safe_max: float,
    min_rates_needed: int,
    min_specimens_per_rate: int,
    n_grid: int = 200,
):
   
    smin = float(df_all["Strain_percent"].min())
    smax = float(df_all["Strain_percent"].max())
    safe_min = max(safe_min, smin)
    safe_max = min(safe_max, smax)
    if safe_min >= safe_max:
        raise ValueError(f"SAFE band invalid after clamping: [{safe_min}, {safe_max}]")

    grid = np.linspace(safe_min, safe_max, n_grid)

    rows = []
    for eps in grid:
        eps = float(eps)
        try:
            rs = flow_stress_by_rate_at_strain(
                df_all, eps, agg=AGG, min_specimens_per_rate=min_specimens_per_rate
            )
            n_rates = int(len(rs))
        except ValueError:
            n_rates = 0
        rows.append((eps, n_rates))

    cov_df = pd.DataFrame(rows, columns=["strain_pct", "n_usable_rates"])

    feasible = cov_df[cov_df["n_usable_rates"] >= min_rates_needed].copy()

    def requested_ok() -> bool:
        if not (safe_min <= requested <= safe_max):
            return False
        # use nearest grid point for count
        i = int((cov_df["strain_pct"] - requested).abs().idxmin())
        return int(cov_df.loc[i, "n_usable_rates"]) >= min_rates_needed

    req_ok = requested_ok()

    if feasible.empty:
        best_any = cov_df.sort_values(["n_usable_rates", "strain_pct"], ascending=[False, False]).iloc[0]
        return float(best_any["strain_pct"]), cov_df, False, safe_min, safe_max

    best = feasible.sort_values(["n_usable_rates", "strain_pct"], ascending=[False, False]).iloc[0]
    target = float(requested) if req_ok else float(best["strain_pct"])
    return target, cov_df, True, safe_min, safe_max


# =========================================================
# MAIN: try strict rule, then auto-relax if needed
# =========================================================
def run_with_min_specimens(min_specimens_per_rate: int):
    target, cov_df, has_feasible, safe_min, safe_max = choose_target_strain_usable_rate_aware(
        df_all=df,
        requested=REQUESTED_TARGET_STRAIN_PCT,
        safe_min=SAFE_MIN_PCT,
        safe_max=SAFE_MAX_PCT,
        min_rates_needed=MIN_RATES_FOR_FIT,
        min_specimens_per_rate=min_specimens_per_rate,
        n_grid=220
    )

    print("\n=== Target strain selection (usable-rate aware) ===")
    print(f"SAFE band: [{safe_min:.4g}%, {safe_max:.4g}%]")
    print(f"min_specimens_per_rate = {min_specimens_per_rate}")
    if not has_feasible:
        print("WARNING: No strain has enough usable rates in SAFE band with this min_specimens rule.")
    print(f"Selected target strain = {target:.6g}%")

    top = cov_df.sort_values(["n_usable_rates", "strain_pct"], ascending=[False, False]).head(10)
    print("\nTop 10 strains by usable-rate coverage:")
    print(top.to_string(index=False))

    # Plot usable rate coverage
    plt.figure(figsize=(7, 4.5))
    plt.plot(cov_df["strain_pct"], cov_df["n_usable_rates"], marker="o")
    plt.axvline(target, linestyle="--")
    plt.xlabel("Target strain (%)")
    plt.ylabel("Usable strain rates at strain")
    plt.title(f"Usable rate coverage vs strain (min_specimens_per_rate={min_specimens_per_rate})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Try compute m
    m, intercept, r2log, rs_table = compute_m_at_strain(df, target, agg=AGG, min_specimens_per_rate=min_specimens_per_rate)
    return target, m, intercept, r2log, rs_table


# 1) Try strict
try:
    target_strain_pct, m, intercept, r2log, rs_table = run_with_min_specimens(MIN_SPECIMENS_PER_RATE)
except ValueError as e:
    print("\nStrict rule failed:", e)
    print("Auto-relaxing MIN_SPECIMENS_PER_RATE to 1 ...\n")
    MIN_SPECIMENS_PER_RATE = 1
    target_strain_pct, m, intercept, r2log, rs_table = run_with_min_specimens(MIN_SPECIMENS_PER_RATE)

print("\n=== SRS result ===")
print(f"Target strain: {target_strain_pct:.6g}%")
print(f"m = {m:.6f}    R^2(log) = {r2log:.4f}    n_rates = {len(rs_table)}")
print("\nPer-rate aggregated flow stress used in fit:")
print(rs_table.to_string(index=False))

# =========================================================
# VISUALS
# =========================================================

# Stress–strain curves (subset) + target marker
curves = list(df.groupby(["Specimen_Name", "Nominal_Strain_Rate"], sort=False))
curves.sort(key=lambda t: float(t[0][1]))
max_to_plot = min(len(curves), 60)
idx = np.linspace(0, len(curves) - 1, max_to_plot).astype(int)
curves_to_plot = [curves[i] for i in idx]

plt.figure(figsize=(8, 6))
for (spec, rate), g in curves_to_plot:
    g2 = g.sort_values("Strain_percent")
    plt.plot(g2["Strain_percent"], g2["Stress_MPa"], alpha=0.75, label=f"{rate:.2e} 1/s")
plt.axvline(target_strain_pct, linestyle="--")
plt.xlabel("Strain (%)")
plt.ylabel("Stress (MPa)")
plt.title("Alloy 617 Stress–Strain Curves (subset)")
handles, labels = plt.gca().get_legend_handles_labels()
seen = set()
uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
if uniq:
    h, l = zip(*uniq[:12])
    plt.legend(h, l, fontsize=8, loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# Log-log flow stress vs strain rate at target strain + fit
rates = rs_table["Nominal_Strain_Rate"].to_numpy(float)
sigmas = rs_table["flow_stress_MPa"].to_numpy(float)

plt.figure(figsize=(6, 5))
plt.loglog(rates, sigmas, "o", label="Data")

rate_smooth = np.logspace(np.log10(rates.min()), np.log10(rates.max()), 200)
sigma_fit = np.exp(intercept) * rate_smooth ** m
plt.loglog(rate_smooth, sigma_fit, "-", label=f"Fit: m={m:.4f}")

plt.xlabel("Strain rate (1/s)")
plt.ylabel("Flow stress (MPa)")
plt.title(f"SRS at {target_strain_pct:.4g}% strain (R²log={r2log:.3f})")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
