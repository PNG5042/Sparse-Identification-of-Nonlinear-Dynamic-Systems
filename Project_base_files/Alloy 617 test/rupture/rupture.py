import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.model_selection import train_test_split, KFold

# ----------------------------
# Utilities
# ----------------------------
def standardize_columns_617(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common Alloy 617 rupture dataset columns to:
      - T_K
      - stress_MPa
      - time_h
      - (optional) heat

    Handles likely variants:
      - Temperature in K or C
      - Stress in MPa or ksi
      - Time in hours or seconds
    """
    rename = {}
    for c in df.columns:
        lc = c.strip().lower()

        # Temperature
        if "temp" in lc or lc in {"t", "temperature"}:
            if "k" in lc:
                rename[c] = "T_K"
            elif "c" in lc or "°c" in lc:
                rename[c] = "T_C"
            else:
                # default unknown temp column -> treat as K later if values look like K
                rename[c] = "T_raw"

        # Stress
        elif "stress" in lc or lc in {"sigma", "σ"}:
            if "mpa" in lc:
                rename[c] = "stress_MPa"
            elif "ksi" in lc:
                rename[c] = "stress_ksi"
            else:
                rename[c] = "stress_raw"

        # Rupture time
        elif ("rupture" in lc or "tr" in lc or "t_r" in lc) and ("time" in lc or "h" in lc or "hr" in lc or "hour" in lc):
            rename[c] = "time_h"
        elif ("time" in lc) and ("h" in lc or "hr" in lc or "hour" in lc):
            rename[c] = "time_h"
        elif ("time" in lc) and ("s" in lc or "sec" in lc or "second" in lc):
            rename[c] = "time_s"

        # Heat / batch id
        elif lc == "heat" or "heat" in lc or "batch" in lc:
            rename[c] = "heat"

    df = df.rename(columns=rename)

    # --- Temperature normalization ---
    if "T_K" not in df.columns:
        if "T_C" in df.columns:
            df["T_K"] = pd.to_numeric(df["T_C"], errors="coerce") + 273.15
        elif "T_raw" in df.columns:
            t = pd.to_numeric(df["T_raw"], errors="coerce")
            # Heuristic: if median > 200, probably Kelvin; else Celsius
            med = np.nanmedian(t.to_numpy())
            df["T_K"] = t if (med is not None and med > 200) else (t + 273.15)
        else:
            raise ValueError(f"Could not find a temperature column. Found columns: {list(df.columns)}")

    # --- Stress normalization ---
    if "stress_MPa" not in df.columns:
        if "stress_ksi" in df.columns:
            df["stress_MPa"] = pd.to_numeric(df["stress_ksi"], errors="coerce") * 6.89475729
        elif "stress_raw" in df.columns:
            s = pd.to_numeric(df["stress_raw"], errors="coerce")
            # Heuristic: if typical values < 5, maybe GPa? if < 100 maybe ksi? else MPa
            # We'll only auto-convert ksi->MPa if it looks like ksi (e.g., < ~200)
            med = np.nanmedian(s.to_numpy())
            if med is not None and med < 300:  # could be ksi
                # This is a guess; if your file is already MPa but low-stress data exist, remove this.
                df["stress_MPa"] = s * 6.89475729
            else:
                df["stress_MPa"] = s
        else:
            raise ValueError(f"Could not find a stress column. Found columns: {list(df.columns)}")

    # --- Time normalization ---
    if "time_h" not in df.columns:
        if "time_s" in df.columns:
            df["time_h"] = pd.to_numeric(df["time_s"], errors="coerce") / 3600.0
        else:
            raise ValueError(f"Could not find a rupture time column. Found columns: {list(df.columns)}")

    # Coerce numeric + drop NaNs
    df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
    df["stress_MPa"] = pd.to_numeric(df["stress_MPa"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["T_K", "stress_MPa", "time_h"])

    # Guard against non-physical values
    df = df[(df["T_K"] > 0) & (df["stress_MPa"] > 0) & (df["time_h"] > 0)]

    return df


def lmp(T_K: np.ndarray, time_h: np.ndarray, C: float = 20.0) -> np.ndarray:
    """Larson–Miller Parameter: LMP = T * (C + log10(t_r[h]))"""
    return T_K * (C + np.log10(time_h))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------
# PySINDy Model fitting
# ----------------------------
def fit_sindy_lmp_model(
    df: pd.DataFrame,
    C: float = 20.0,
    poly_degree: int = 3,
    threshold: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Use PySINDy to discover relationship:
        LMP = f(log10(stress))

    Returns dict with fitted model + metrics.
    """
    T = df["T_K"].to_numpy()
    t = df["time_h"].to_numpy()
    stress = df["stress_MPa"].to_numpy()

    y = lmp(T, t, C=C).reshape(-1, 1)
    X = np.log10(stress).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    library = ps.PolynomialLibrary(degree=poly_degree, include_bias=True)
    optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        feature_names=["log10(stress)"],
    )

    model.fit(X_train, y_train, quiet=True)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "model": model,
        "C": float(C),
        "poly_degree": int(poly_degree),
        "threshold": float(threshold),
        "train_rmse": rmse(y_train.flatten(), y_pred_train.flatten()),
        "test_rmse": rmse(y_test.flatten(), y_pred_test.flatten()),
        "train_score": float(model.score(X_train, y_train)),
        "test_score": float(model.score(X_test, y_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def optimize_C_with_sindy(df: pd.DataFrame, C_grid=None, poly_degree: int = 3, threshold: float = 0.01):
    """
    Scan C and pick best by test RMSE (single split).
    """
    if C_grid is None:
        C_grid = np.linspace(15.0, 25.0, 21)

    best = None
    results = []

    for C in C_grid:
        try:
            r = fit_sindy_lmp_model(df, C=C, poly_degree=poly_degree, threshold=threshold)
            results.append(r)
            if (best is None) or (r["test_rmse"] < best["test_rmse"]):
                best = r
        except Exception as e:
            print(f"Failed C={C:.2f}: {e}")

    return best, results


def crossval_score_C(df: pd.DataFrame, C: float, poly_degree: int = 3, threshold: float = 0.01, n_splits: int = 5, seed: int = 42):
    """
    Optional: k-fold CV for a single C (more stable than one split).
    Returns mean/std RMSE in LMP space.
    """
    T = df["T_K"].to_numpy()
    t = df["time_h"].to_numpy()
    stress = df["stress_MPa"].to_numpy()

    y_all = lmp(T, t, C=C).reshape(-1, 1)
    X_all = np.log10(stress).reshape(-1, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses = []

    for tr_idx, te_idx in kf.split(X_all):
        X_train, X_test = X_all[tr_idx], X_all[te_idx]
        y_train, y_test = y_all[tr_idx], y_all[te_idx]

        library = ps.PolynomialLibrary(degree=poly_degree, include_bias=True)
        optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)
        model = ps.SINDy(feature_library=library, optimizer=optimizer, feature_names=["log10(stress)"])
        model.fit(X_train, y_train, quiet=True)

        y_pred = model.predict(X_test)
        rmses.append(rmse(y_test.flatten(), y_pred.flatten()))

    return float(np.mean(rmses)), float(np.std(rmses))


def predict_rupture_time_hours(T_K: float, stress_MPa: float, model_dict: dict) -> float:
    """
    Predict rupture time using fitted SINDy model:
      1) predict LMP from log10(stress)
      2) invert LMP to solve for t
    """
    C = model_dict["C"]
    model = model_dict["model"]

    X = np.log10(stress_MPa).reshape(1, -1)
    LMP_pred = float(model.predict(X)[0, 0])

    logt = (LMP_pred / T_K) - C
    return float(10 ** logt)


# ----------------------------
# Main (Alloy 617 test)
# ----------------------------
if __name__ == "__main__":
    # ---- Update these for your Alloy 617 dataset ----
    path = "Alloy617-rupture.xlsx"
    sheet = "Rupture"  # or whatever your sheet is called

    raw = pd.read_excel(path, sheet_name=sheet)
    df = standardize_columns_617(raw)

    print(f"Loaded {len(df)} Alloy 617 data points")
    print(f"T range: {df['T_K'].min():.2f} – {df['T_K'].max():.2f} K")
    print(f"Stress range: {df['stress_MPa'].min():.2f} – {df['stress_MPa'].max():.2f} MPa")
    print(f"Time range: {df['time_h'].min():.4g} – {df['time_h'].max():.4g} h\n")

    # Option 1: Fit with fixed C
    print("=== Fitting SINDy model with C=20 ===")
    model_fixed = fit_sindy_lmp_model(df, C=20.0, poly_degree=3, threshold=0.01)
    print(f"Train RMSE: {model_fixed['train_rmse']:.3f} | Test RMSE: {model_fixed['test_rmse']:.3f}")
    print(f"Train R²:   {model_fixed['train_score']:.4f} | Test R²:  {model_fixed['test_score']:.4f}")
    print("Discovered equation (LMP as f(log10(stress))):")
    model_fixed["model"].print()
    print()

    # Option 2: Optimize C by scan
    print("=== Optimizing C parameter (scan) ===")
    best_model, all_results = optimize_C_with_sindy(df, poly_degree=3, threshold=0.01)
    print(f"Best C: {best_model['C']:.3f}")
    print(f"Train RMSE: {best_model['train_rmse']:.3f} | Test RMSE: {best_model['test_rmse']:.3f}")
    print(f"Train R²:   {best_model['train_score']:.4f} | Test R²:  {best_model['test_score']:.4f}")
    print("Discovered equation (LMP as f(log10(stress))):")
    best_model["model"].print()
    print()

    # Optional: More stable check on the chosen C (k-fold CV)
    print("=== Cross-validation check (k-fold) for best C ===")
    mean_rmse, std_rmse = crossval_score_C(df, C=best_model["C"], poly_degree=3, threshold=0.01, n_splits=5)
    print(f"CV RMSE (LMP): {mean_rmse:.3f} ± {std_rmse:.3f}\n")

    # Example predictions (edit to match typical 617 conditions)
    print("=== Example Predictions (Alloy 617) ===")
    test_conditions = [
        (1173.0, 80.0),   # 900°C, 80 MPa
        (1073.0, 120.0),  # 800°C, 120 MPa
        (973.0, 160.0),   # 700°C, 160 MPa
    ]

    for T_test, stress_test in test_conditions:
        t_pred = predict_rupture_time_hours(T_test, stress_test, best_model)
        print(f"T={T_test:.1f} K, σ={stress_test:.1f} MPa → t_r = {t_pred:.3g} h ({t_pred/8760:.3g} years)")
