import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.model_selection import train_test_split

# ----------------------------
# Utilities
# ----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names easier to match across files.
    """
    rename = {}
    for c in df.columns:
        lc = c.strip().lower()
        if "temp" in lc:
            rename[c] = "T_K"
        elif "stress" in lc:
            rename[c] = "stress_MPa"
        elif "rupture" in lc and ("time" in lc or "h" in lc):
            rename[c] = "time_h"
        elif "time" in lc and ("h" in lc or "hour" in lc):
            rename[c] = "time_h"
        elif lc == "heat":
            rename[c] = "heat"
    df = df.rename(columns=rename)

    required = {"T_K", "stress_MPa", "time_h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    # Coerce numeric
    df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
    df["stress_MPa"] = pd.to_numeric(df["stress_MPa"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["T_K", "stress_MPa", "time_h"])

    # Guard against non-physical values
    df = df[(df["T_K"] > 0) & (df["stress_MPa"] > 0) & (df["time_h"] > 0)]
    return df


def lmp(T_K: np.ndarray, time_h: np.ndarray, C: float = 20.0) -> np.ndarray:
    """
    Larson–Miller Parameter (LMP).
    LMP = T * (C + log10(t_r))
    """
    return T_K * (C + np.log10(time_h))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------
# PySINDy Model fitting
# ----------------------------
def fit_sindy_lmp_model(df: pd.DataFrame, C: float = 20.0, 
                        poly_degree: int = 3, threshold: float = 0.01):
    """
    Use PySINDy to discover the relationship between LMP and stress.
    
    Parameters:
    -----------
    df : DataFrame with T_K, stress_MPa, time_h
    C : LMP constant (can be optimized separately)
    poly_degree : maximum polynomial degree for library
    threshold : sparsity threshold for STLSQ
    
    Returns:
    --------
    dict with model, scaler info, and metadata
    """
    # Calculate LMP
    T = df["T_K"].to_numpy()
    t = df["time_h"].to_numpy()
    stress = df["stress_MPa"].to_numpy()
    
    LMP_vals = lmp(T, t, C=C)
    
    # Transform stress to log scale (common in creep analysis)
    log_stress = np.log10(stress).reshape(-1, 1)
    LMP_vals = LMP_vals.reshape(-1, 1)
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        log_stress, LMP_vals, test_size=0.2, random_state=42
    )
    
    # Create polynomial library
    library = ps.PolynomialLibrary(degree=poly_degree, include_bias=True)
    
    # Use STLSQ optimizer for sparsity
    optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)
    
    # Fit SINDy model
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        feature_names=["log10(stress)"]
    )
    
    model.fit(X_train, y_train, quiet=True)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = rmse(y_train.flatten(), y_pred_train.flatten())
    test_rmse = rmse(y_test.flatten(), y_pred_test.flatten())
    
    return {
        "model": model,
        "C": float(C),
        "poly_degree": poly_degree,
        "threshold": threshold,
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_score": float(model.score(X_train, y_train)),
        "test_score": float(model.score(X_test, y_test))
    }


def optimize_C_with_sindy(df: pd.DataFrame, C_grid=None, poly_degree: int = 3):
    """
    Find optimal C parameter by scanning and fitting SINDy models.
    """
    if C_grid is None:
        C_grid = np.linspace(15.0, 25.0, 21)
    
    best = None
    results = []
    
    for C in C_grid:
        try:
            result = fit_sindy_lmp_model(df, C=C, poly_degree=poly_degree)
            results.append(result)
            
            # Use validation RMSE for selection
            if (best is None) or (result["test_rmse"] < best["test_rmse"]):
                best = result
        except Exception as e:
            print(f"Failed to fit with C={C:.2f}: {e}")
            continue
    
    return best, results


def predict_rupture_time_hours(T_K: float, stress_MPa: float, model_dict: dict) -> float:
    """
    Predict rupture time using fitted SINDy model.
    
    Steps:
      1) Predict LMP from log10(stress) using SINDy model
      2) Invert LMP relation to solve for time
    """
    C = model_dict["C"]
    model = model_dict["model"]
    
    log_stress = np.log10(stress_MPa).reshape(1, -1)
    LMP_pred = float(model.predict(log_stress)[0, 0])
    
    # LMP = T*(C + log10(t))  => log10(t) = (LMP/T) - C
    logt = (LMP_pred / T_K) - C
    t_h = 10 ** logt
    return float(t_h)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Update path if needed
    path = "SS316H-rupture.xlsx"
    sheet = "Rupture"
    
    raw = pd.read_excel(path, sheet_name=sheet)
    df = standardize_columns(raw)
    
    print(f"Loaded {len(df)} data points")
    print(f"Temperature range: {df['T_K'].min():.1f} - {df['T_K'].max():.1f} K")
    print(f"Stress range: {df['stress_MPa'].min():.1f} - {df['stress_MPa'].max():.1f} MPa")
    print(f"Time range: {df['time_h'].min():.1f} - {df['time_h'].max():.1f} h\n")
    
    # Option 1: Fit with fixed C=20
    print("=== Fitting SINDy model with C=20 ===")
    model_fixed = fit_sindy_lmp_model(df, C=20.0, poly_degree=3, threshold=0.01)
    
    print(f"Train RMSE: {model_fixed['train_rmse']:.3f}")
    print(f"Test RMSE: {model_fixed['test_rmse']:.3f}")
    print(f"Train R²: {model_fixed['train_score']:.4f}")
    print(f"Test R²: {model_fixed['test_score']:.4f}\n")
    
    print("Discovered equation (LMP as function of log10(stress)):")
    model_fixed["model"].print()
    print()
    
    # Option 2: Optimize C parameter
    print("=== Optimizing C parameter ===")
    best_model, all_results = optimize_C_with_sindy(df, poly_degree=3)
    
    print(f"Best C: {best_model['C']:.3f}")
    print(f"Train RMSE: {best_model['train_rmse']:.3f}")
    print(f"Test RMSE: {best_model['test_rmse']:.3f}")
    print(f"Train R²: {best_model['train_score']:.4f}")
    print(f"Test R²: {best_model['test_score']:.4f}\n")
    
    print("Discovered equation (LMP as function of log10(stress)):")
    best_model["model"].print()
    print()
    
    # Example predictions
    print("=== Example Predictions ===")
    test_conditions = [
        (873.0, 200.0),
        (923.0, 150.0),
        (823.0, 250.0)
    ]
    
    for T_test, stress_test in test_conditions:
        t_pred = predict_rupture_time_hours(T_test, stress_test, best_model)
        print(f"T={T_test} K, σ={stress_test} MPa → t_rupture = {t_pred:.2f} h ({t_pred/8760:.2f} years)")
