import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


# ==============================
# Utility: Standardize Columns
# ==============================

def standardize_columns(df):
    df = df.copy()

    # Strip whitespace from column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Map EXACT names from your Excel file -> expected names
    column_map = {
        "Temp (K)": "T",
        "Stress (Mpa)": "sigma",
        "Rupture time (h)": "t_r",
        # (Optional extra aliases if you ever change the sheet)
        "Temperature": "T",
        "Temperature (K)": "T",
        "Stress": "sigma",
        "Stress (MPa)": "sigma",
        "Rupture Time": "t_r",
        "Rupture Time (h)": "t_r",
    }

    df = df.rename(columns=column_map)

    # sanity check
    missing = [c for c in ["T", "sigma", "t_r"] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


# ==================================
# Larson-Miller Parameter (LMP)
# ==================================

def compute_lmp(T, t_r, C):
    return T * (C + np.log10(t_r))


# ==============================
# Model Container
# ==============================

class SINDyRuptureModel:
    def __init__(self, model, C, test_r2, cv_mean, cv_std):
        self.model = model
        self.C = C
        self.test_r2 = test_r2
        self.cv_mean = cv_mean
        self.cv_std = cv_std


# ==============================
# Fit Model
# ==============================

def fit_sindy_lmp_model(df, C=20.0, poly_degree=3, threshold=0.01):

    T = df["T"].values
    sigma = df["sigma"].values
    t_r = df["t_r"].values

    lmp = compute_lmp(T, t_r, C)

    X = np.column_stack([sigma])
    y = lmp

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    library = ps.PolynomialLibrary(degree=poly_degree)
    optimizer = ps.STLSQ(threshold=threshold)

    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    t_train = np.arange(len(X_train))
    model.fit(X_train, t=t_train, x_dot=y_train.reshape(-1, 1))

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        t_cv = np.arange(len(train_idx))
        model.fit(X[train_idx], t=t_cv, x_dot=y[train_idx].reshape(-1, 1))
        y_pred_cv = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], y_pred_cv))

    cv_mean = np.mean(scores)
    cv_std = np.std(scores)

    return SINDyRuptureModel(model, C, test_r2, cv_mean, cv_std)


# ==============================
# Predict Rupture Time
# ==============================

def predict_rupture_time(T, sigma, model_obj):

    X = np.array([[sigma]])
    predicted_lmp = model_obj.model.predict(X)[0][0]

    C = model_obj.C

    # invert LMP equation
    log_tr = predicted_lmp / T - C
    t_r = 10 ** log_tr

    return t_r


# ==============================
# Print Summary
# ==============================

def print_model_summary(model_obj):
    print("\nModel Summary")
    print("-" * 40)
    print(f"C Parameter: {model_obj.C}")
    print(f"Test R²: {model_obj.test_r2:.4f}")
    print(f"CV R²: {model_obj.cv_mean:.4f} ± {model_obj.cv_std:.4f}")
    print("\nDiscovered Equation:")
    model_obj.model.print()


# ==============================
# Optimize C Parameter
# ==============================

def optimize_C_parameter(df, C_grid, poly_degree=3, threshold=0.01):

    results = []
    best_model = None
    best_score = -np.inf

    for C in C_grid:
        model = fit_sindy_lmp_model(df, C, poly_degree, threshold)
        results.append((C, model.cv_mean))

        if model.cv_mean > best_score:
            best_score = model.cv_mean
            best_model = model

    results_df = pd.DataFrame(results, columns=["C", "CV_R2"])
    return best_model, results_df


# ==============================
# Plot Optimization
# ==============================

def plot_c_optimization(results_df):

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["C"], results_df["CV_R2"], marker="o")
    plt.xlabel("C Parameter")
    plt.ylabel("Cross-Validation R²")
    plt.title("C Parameter Optimization")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================
# Batch Predictions
# ==============================

def create_prediction_table(conditions, model_obj):

    rows = []

    for T, sigma in conditions:
        t_r = predict_rupture_time(T, sigma, model_obj)
        rows.append((T, sigma, t_r, t_r / 8760))

    return pd.DataFrame(rows, columns=[
        "Temperature [K]",
        "Stress [MPa]",
        "Rupture Time [h]",
        "Rupture Time [years]"
    ])


# ==============================
# Diagnostics Plot
# ==============================

def plot_model_performance(df, model_obj, save_path=None):

    T = df["T"].values
    sigma = df["sigma"].values
    t_r = df["t_r"].values

    predicted = [
        predict_rupture_time(T[i], sigma[i], model_obj)
        for i in range(len(T))
    ]

    plt.figure(figsize=(8, 6))
    plt.scatter(t_r, predicted)
    plt.plot([min(t_r), max(t_r)],
             [min(t_r), max(t_r)], 'r--')
    plt.xlabel("Actual Rupture Time")
    plt.ylabel("Predicted Rupture Time")
    plt.title("Model Performance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
