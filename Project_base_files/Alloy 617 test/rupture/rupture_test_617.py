import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Parsing helpers
# -----------------------------

def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_temperature_and_stress(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Parse temperature and stress from specimen/file text.

    Supported examples:
      900C, 900 C, 900°C, 1173K, 100MPa, 14.5ksi
      A617_900C_100MPa, Test-950C-80MPa, etc.
    """
    s = _safe_str(text)

    temp_value = None
    temp_unit = None
    stress_value = None
    stress_unit = None

    temp_patterns = [
        r'(\d+(?:\.\d+)?)\s*°?C\b',
        r'(\d+(?:\.\d+)?)\s*K\b',
    ]
    for pat in temp_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            temp_value = float(m.group(1))
            temp_unit = 'K' if 'k' in pat.lower() else 'C'
            break

    stress_patterns = [
        r'(\d+(?:\.\d+)?)\s*MPa\b',
        r'(\d+(?:\.\d+)?)\s*ksi\b',
    ]
    for pat in stress_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            stress_value = float(m.group(1))
            stress_unit = 'ksi' if 'ksi' in pat.lower() else 'MPa'
            break

    return temp_value, stress_value, temp_unit, stress_unit


def convert_temp_to_K(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit is None:
        return None
    unit = unit.upper()
    if unit == 'K':
        return float(value)
    if unit == 'C':
        return float(value) + 273.15
    return None


def convert_stress_to_MPa(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit is None:
        return None
    unit = unit.lower()
    if unit == 'mpa':
        return float(value)
    if unit == 'ksi':
        return float(value) * 6.89475729
    return None


# -----------------------------
# Data loading and rupture extraction
# -----------------------------

def load_csv_files(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            df['source_file'] = os.path.basename(path)
            frames.append(df)
            print(f"Loaded: {path} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not frames:
        raise ValueError("No valid CSV files could be loaded.")

    data = pd.concat(frames, ignore_index=True)
    return data


def filter_alloy_617(df: pd.DataFrame) -> pd.DataFrame:
    if 'Material Name' not in df.columns:
        return df.copy()

    mask = df['Material Name'].astype(str).str.contains(r'617', case=False, na=False)
    filtered = df.loc[mask].copy()
    if len(filtered) == 0:
        print("Warning: no rows matched '617' in 'Material Name'. Using all rows.")
        return df.copy()
    return filtered


def is_failure_state(value: str) -> bool:
    s = _safe_str(value).lower()
    failure_keywords = [
        'rupture', 'failed', 'failure', 'broken', 'fracture', 'complete', 'end', 'ended'
    ]
    return any(k in s for k in failure_keywords)


def extract_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ['Specimen Name', 'Elapsed Time Hours', 'Strain Percent']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    work = df.copy()
    work['Elapsed Time Hours'] = pd.to_numeric(work['Elapsed Time Hours'], errors='coerce')
    work['Strain Percent'] = pd.to_numeric(work['Strain Percent'], errors='coerce')
    work = work.dropna(subset=['Specimen Name', 'Elapsed Time Hours'])

    summary_rows: List[Dict] = []

    for specimen, g in work.groupby('Specimen Name', dropna=True):
        g = g.sort_values('Elapsed Time Hours').copy()
        g = g[g['Elapsed Time Hours'] >= 0]
        if g.empty:
            continue

        rupture_detected = False
        rupture_time_h = float(g['Elapsed Time Hours'].max())

        if 'Qual State' in g.columns:
            qual_fail_mask = g['Qual State'].apply(is_failure_state)
            if qual_fail_mask.any():
                rupture_detected = True
                rupture_time_h = float(g.loc[qual_fail_mask, 'Elapsed Time Hours'].max())

        final_row = g.iloc[-1]
        last_strain = pd.to_numeric(final_row.get('Strain Percent', np.nan), errors='coerce')

        # Try specimen name first, then file name.
        t_val, s_val, t_unit, s_unit = parse_temperature_and_stress(_safe_str(specimen))
        parse_source = 'Specimen Name'

        if t_val is None or s_val is None:
            ft, fs, fu_t, fu_s = parse_temperature_and_stress(_safe_str(final_row.get('source_file', '')))
            if t_val is None:
                t_val, t_unit = ft, fu_t
            if s_val is None:
                s_val, s_unit = fs, fu_s
            if ft is not None or fs is not None:
                parse_source = 'source_file'

        temp_K = convert_temp_to_K(t_val, t_unit)
        stress_MPa = convert_stress_to_MPa(s_val, s_unit)

        summary_rows.append({
            'Specimen Name': specimen,
            'Material Name': final_row.get('Material Name', np.nan),
            'Material Form': final_row.get('Material Form', np.nan),
            'Heat': final_row.get('Heat', np.nan),
            'Count_max': g['Count'].max() if 'Count' in g.columns else np.nan,
            'rupture_time_h': rupture_time_h,
            'last_time_h': float(g['Elapsed Time Hours'].max()),
            'last_strain_percent': last_strain,
            'rupture_detected_from_qual_state': rupture_detected,
            'temperature_value_raw': t_val,
            'temperature_unit_raw': t_unit,
            'stress_value_raw': s_val,
            'stress_unit_raw': s_unit,
            'T_K': temp_K,
            'stress_MPa': stress_MPa,
            'parsed_from': parse_source,
            'source_file': final_row.get('source_file', np.nan),
            'n_points': len(g),
        })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("No specimen summaries could be extracted.")

    return summary


# -----------------------------
# Larson-Miller model
# -----------------------------

def compute_lmp(T_K: np.ndarray, time_h: np.ndarray, C: float) -> np.ndarray:
    return T_K * (C + np.log10(time_h))


def fit_larson_miller_model(data: pd.DataFrame, C: float = 20.0, test_size: float = 0.25, random_state: int = 42) -> Dict:
    model_df = data.dropna(subset=['T_K', 'stress_MPa', 'rupture_time_h']).copy()
    model_df = model_df[(model_df['T_K'] > 0) & (model_df['stress_MPa'] > 0) & (model_df['rupture_time_h'] > 0)]

    if len(model_df) < 4:
        raise ValueError(
            "Not enough valid rupture rows for modeling. Need at least 4 rows with parsed temperature, stress, and rupture time."
        )

    X = np.log10(model_df['stress_MPa'].to_numpy()).reshape(-1, 1)
    y = compute_lmp(model_df['T_K'].to_numpy(), model_df['rupture_time_h'].to_numpy(), C)

    if len(model_df) >= 6:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, model_df.index.to_numpy(), test_size=test_size, random_state=random_state
        )
    else:
        # tiny datasets: train/test split can be unstable
        X_train, y_train, idx_train = X, y, model_df.index.to_numpy()
        X_test, y_test, idx_test = X, y, model_df.index.to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        'model': model,
        'C': C,
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'train_r2': float(r2_score(y_train, train_pred)),
        'test_r2': float(r2_score(y_test, test_pred)),
        'model_df': model_df,
        'train_idx': idx_train,
        'test_idx': idx_test,
    }


def optimize_C(data: pd.DataFrame, c_values: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    results = []
    best = None
    best_score = np.inf

    for C in c_values:
        try:
            res = fit_larson_miller_model(data, C=float(C))
            results.append({
                'C': float(C),
                'test_rmse': res['test_rmse'],
                'test_r2': res['test_r2'],
                'train_rmse': res['train_rmse'],
                'train_r2': res['train_r2'],
            })
            if res['test_rmse'] < best_score:
                best_score = res['test_rmse']
                best = res
        except Exception as e:
            results.append({
                'C': float(C),
                'test_rmse': np.nan,
                'test_r2': np.nan,
                'train_rmse': np.nan,
                'train_r2': np.nan,
                'error': str(e),
            })

    if best is None:
        raise ValueError("Could not fit any Larson-Miller model. Check whether temperature and stress can be parsed.")

    return best, pd.DataFrame(results)


def predict_rupture_time_hours(T_K: float, stress_MPa: float, fit_result: Dict) -> float:
    X = np.array([[np.log10(stress_MPa)]])
    lmp_pred = float(fit_result['model'].predict(X)[0])
    C = float(fit_result['C'])
    log10_time_h = lmp_pred / T_K - C
    return float(10 ** log10_time_h)


def add_predictions(summary_df: pd.DataFrame, fit_result: Dict) -> pd.DataFrame:
    out = summary_df.copy()
    preds = []
    for _, row in out.iterrows():
        T = row.get('T_K', np.nan)
        s = row.get('stress_MPa', np.nan)
        if pd.notna(T) and pd.notna(s) and T > 0 and s > 0:
            preds.append(predict_rupture_time_hours(float(T), float(s), fit_result))
        else:
            preds.append(np.nan)
    out['predicted_rupture_time_h'] = preds
    out['prediction_error_ratio'] = out['predicted_rupture_time_h'] / out['rupture_time_h']
    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_actual_vs_predicted(summary_df: pd.DataFrame, output_path: str) -> None:
    plot_df = summary_df.dropna(subset=['rupture_time_h', 'predicted_rupture_time_h']).copy()
    plot_df = plot_df[(plot_df['rupture_time_h'] > 0) & (plot_df['predicted_rupture_time_h'] > 0)]
    if plot_df.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.loglog(plot_df['rupture_time_h'], plot_df['predicted_rupture_time_h'], 'o')

    mn = min(plot_df['rupture_time_h'].min(), plot_df['predicted_rupture_time_h'].min())
    mx = max(plot_df['rupture_time_h'].max(), plot_df['predicted_rupture_time_h'].max())
    plt.loglog([mn, mx], [mn, mx], '--')

    plt.xlabel('Actual rupture time (h)')
    plt.ylabel('Predicted rupture time (h)')
    plt.title('Alloy 617 Rupture: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_c_optimization(results_df: pd.DataFrame, output_path: str) -> None:
    plot_df = results_df.dropna(subset=['C', 'test_rmse']).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(plot_df['C'], plot_df['test_rmse'], marker='o')
    plt.xlabel('Larson-Miller constant C')
    plt.ylabel('Test RMSE (LMP units)')
    plt.title('C Optimization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stress_rupture(summary_df: pd.DataFrame, output_path: str) -> None:
    plot_df = summary_df.dropna(subset=['stress_MPa', 'rupture_time_h']).copy()
    plot_df = plot_df[(plot_df['stress_MPa'] > 0) & (plot_df['rupture_time_h'] > 0)]
    if plot_df.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.loglog(plot_df['stress_MPa'], plot_df['rupture_time_h'], 'o')
    plt.xlabel('Stress (MPa)')
    plt.ylabel('Rupture time (h)')
    plt.title('Alloy 617 Stress-Rupture Data')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Optional user predictions
# -----------------------------

def parse_temperature_and_stress(text):
    """
    Parse temperature and stress from specimen/file text.

    Works with examples like:
      900C
      900 C
      900°C
      1173K
      100MPa
      100_MPA
      14.5ksi
      Creep_A-13A_1000C_12MPA_G-52.csv
    """
    import re

    s = str(text) if text is not None else ""

    temp_value = None
    temp_unit = None
    stress_value = None
    stress_unit = None

    temp_patterns = [
        r'(\d+(?:\.\d+)?)\s*°?\s*C(?=[^A-Za-z0-9]|$)',
        r'(\d+(?:\.\d+)?)\s*K(?=[^A-Za-z0-9]|$)',
    ]

    stress_patterns = [
        r'(\d+(?:\.\d+)?)\s*M\s*P\s*A(?=[^A-Za-z0-9]|$)',
        r'(\d+(?:\.\d+)?)\s*K\s*S\s*I(?=[^A-Za-z0-9]|$)',
    ]

    for pat in temp_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            temp_value = float(m.group(1))
            if 'k' in pat.lower():
                temp_unit = "K"
            else:
                temp_unit = "C"
            break

    for pat in stress_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            stress_value = float(m.group(1))
            if 'k' in pat.lower():
                stress_unit = "ksi"
            else:
                stress_unit = "MPa"
            break

    return temp_value, stress_value, temp_unit, stress_unit

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Alloy 617 creep CSV to rupture model')
    parser.add_argument('--pattern', default='*.csv', help='CSV glob pattern, e.g. "*.csv"')
    parser.add_argument('--out', default='alloy617_creep_rupture_results.xlsx', help='Output Excel filename')
    parser.add_argument('--predict', default='', help='Predictions as "T_K,stress_MPa; T_K,stress_MPa"')
    args = parser.parse_args()

    raw = load_csv_files(args.pattern)
    raw_617 = filter_alloy_617(raw)

    summary = extract_test_summary(raw_617)

    print(summary[[
        "Specimen Name",
        "source_file",
        "temperature_value_raw",
        "temperature_unit_raw",
        "stress_value_raw",
        "stress_unit_raw",
        "T_K",
        "stress_MPa",
        "rupture_time_h"
    ]].head(20).to_string(index=False))

    valid_model_rows = summary.dropna(subset=["T_K", "stress_MPa", "rupture_time_h"]).copy()
    valid_model_rows = valid_model_rows[
        (valid_model_rows["T_K"] > 0) &
        (valid_model_rows["stress_MPa"] > 0) &
        (valid_model_rows["rupture_time_h"] > 0)
    ]
    print(f"\nRows usable for rupture model: {len(valid_model_rows)}")

    print("Summary rows:", len(summary))

    c_grid = np.linspace(15.0, 25.0, 21)
    best_fit, c_results = optimize_C(summary, c_grid)

    # Add model predictions back onto the summary table
    summary_with_pred = summary.copy()
    summary_with_pred["predicted_rupture_time_h"] = np.nan

    fit_rows = summary_with_pred.dropna(subset=["T_K", "stress_MPa"]).copy()
    fit_rows = fit_rows[(fit_rows["T_K"] > 0) & (fit_rows["stress_MPa"] > 0)]

    if not fit_rows.empty:
        summary_with_pred.loc[fit_rows.index, "predicted_rupture_time_h"] = fit_rows.apply(
            lambda r: predict_rupture_time_hours(
                float(r["T_K"]),
                float(r["stress_MPa"]),
                best_fit
            ),
            axis=1
        )

    user_predictions = pd.DataFrame()
    if args.predict.strip():
        user_predictions = parse_predict_argument(args.predict)
        user_predictions["predicted_rupture_time_h"] = user_predictions.apply(
            lambda r: predict_rupture_time_hours(
                float(r["T_K"]),
                float(r["stress_MPa"]),
                best_fit
            ),
            axis=1
        )

    diagnostics = pd.DataFrame([
        {'metric': 'raw_rows_loaded', 'value': len(raw)},
        {'metric': 'rows_after_alloy617_filter', 'value': len(raw_617)},
        {'metric': 'unique_specimens', 'value': summary['Specimen Name'].nunique()},
        {'metric': 'rows_with_parsed_temperature_and_stress', 'value': len(valid_model_rows)},
        {'metric': 'best_C', 'value': best_fit['C']},
        {'metric': 'train_rmse', 'value': best_fit['train_rmse']},
        {'metric': 'test_rmse', 'value': best_fit['test_rmse']},
        {'metric': 'train_r2', 'value': best_fit['train_r2']},
        {'metric': 'test_r2', 'value': best_fit['test_r2']},
        {
            'metric': 'model_equation',
            'value': f"LMP = {best_fit['model'].intercept_:.6g} + ({best_fit['model'].coef_[0]:.6g}) * log10(stress_MPa)"
        },
    ])

    actual_vs_pred_png = 'alloy617_actual_vs_predicted.png'
    c_opt_png = 'alloy617_C_optimization.png'
    stress_rupture_png = 'alloy617_stress_vs_rupture.png'

    plot_actual_vs_predicted(summary_with_pred, actual_vs_pred_png)
    plot_c_optimization(c_results, c_opt_png)
    plot_stress_rupture(summary_with_pred, stress_rupture_png)

    with pd.ExcelWriter(args.out, engine='openpyxl') as writer:
        raw_617.to_excel(writer, sheet_name='raw_filtered_alloy617', index=False)
        summary.to_excel(writer, sheet_name='specimen_summary', index=False)
        summary_with_pred.to_excel(writer, sheet_name='summary_with_predictions', index=False)
        c_results.to_excel(writer, sheet_name='C_optimization', index=False)
        diagnostics.to_excel(writer, sheet_name='diagnostics', index=False)
        if not user_predictions.empty:
            user_predictions.to_excel(writer, sheet_name='user_predictions', index=False)

    print('\nFinished.')
    print(f"Output Excel: {args.out}")
    print(f"Plot: {actual_vs_pred_png}")
    print(f"Plot: {c_opt_png}")
    print(f"Plot: {stress_rupture_png}")
    print(f"Best C: {best_fit['C']:.2f}")
    print(f"Model: LMP = {best_fit['model'].intercept_:.6g} + ({best_fit['model'].coef_[0]:.6g}) * log10(stress_MPa)")
    print(f"Train R^2: {best_fit['train_r2']:.4f}")
    print(f"Test R^2:  {best_fit['test_r2']:.4f}")

    missing_parsed = summary_with_pred[
        summary_with_pred['T_K'].isna() | summary_with_pred['stress_MPa'].isna()
    ]
    if not missing_parsed.empty:
        print('\nWarning: some specimens did not have temperature/stress parsed from Specimen Name or filename.')
        print('Those rows are still included in the summary, but not used for fitting.')
        print(missing_parsed[['Specimen Name', 'source_file', 'parsed_from']].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
