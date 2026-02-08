# ================================
# FINAL OPTIMIZED SS316H CREEP MODEL
# With Temperature Range Analysis and CSV Export
# ================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Load data
csv_path = Path(__file__).parent / "SS316H-1percent.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("SS316H CREEP MODEL - WITH TEMP RANGES & CSV EXPORT")
print("="*80)

# Encode Heat
if df["Heat"].dtype == object:
    heat_mapping = {heat: i for i, heat in enumerate(df["Heat"].unique())}
    df["Heat_encoded"] = df["Heat"].map(heat_mapping)

Heat = df["Heat_encoded"].values
Temp = df["Temp (K)"].values
Stress = df["Stress (Mpa)"].values
Time = df["Time (h) to 1% strain"].values

print(f"\nDataset: {len(Time)} samples")
print(f"Time range: {Time.min():.1f} - {Time.max():.1f} hours")
print(f"Temperature range: {Temp.min():.1f} - {Temp.max():.1f} K")

# =====================================
# DEFINE TEMPERATURE RANGES
# =====================================
print(f"\n{'='*80}")
print("TEMPERATURE RANGE DISTRIBUTION")
print(f"{'='*80}")

# Define temperature bins (adjust these ranges based on your data)
temp_ranges = [
    (Temp.min(), 950, "Low"),
    (951, 1050, "Medium"),
    (1051, Temp.max()+1, "High")
]

# Analyze temperature distribution
for temp_min, temp_max, label in temp_ranges:
    mask = (Temp >= temp_min) & (Temp < temp_max)
    count = mask.sum()
    if count > 0:
        temps_in_range = Temp[mask]
        times_in_range = Time[mask]
        print(f"{label} Temp ({temp_min:.0f}-{temp_max:.0f}K): {count} samples")
        print(f"  Temp range: {temps_in_range.min():.1f} - {temps_in_range.max():.1f} K")
        print(f"  Time range: {times_in_range.min():.1f} - {times_in_range.max():.1f} h")

# =====================================
# STRATIFIED SAMPLING
# =====================================
print(f"\n{'='*80}")
print("STRATIFIED TRAIN/TEST SPLIT")
print(f"{'='*80}")

# Create strata based on time duration (short vs long tests)
time_bins = np.digitize(np.log10(Time), bins=[0, 1, 2, 3, 4, 5, 6])
print(f"\nTime bins distribution:")
for i in range(1, 7):
    count = (time_bins == i).sum()
    if count > 0:
        times_in_bin = Time[time_bins == i]
        print(f"  Bin {i} (10^{i-1}-10^{i}h): {count} samples, range={times_in_bin.min():.1f}-{times_in_bin.max():.1f}h")

# =====================================
# ENHANCED FEATURE ENGINEERING
# =====================================
R = 8.314  # Gas constant

# Physics-based features with emphasis on short-duration behavior
X_features = np.column_stack([
    # Basic features
    np.ones_like(Heat), Heat, Heat**2,
    
    # Temperature features
    Temp, 1/Temp, Temp**2, 1/(Temp**2),
    np.log(Temp), Temp**(-0.5),
    
    # Stress features  
    Stress, np.log(Stress), Stress**2, Stress**3,
    Stress**(-1), Stress**(-2), Stress**(-3),
    np.sqrt(Stress), Stress**0.5,
    
    # Interaction features - KEY for short-duration
    (1/Temp) * Stress,                    
    (1/Temp) * np.log(Stress),            
    Temp * np.log(Stress),                
    np.log(Temp) * np.log(Stress),        
    (1/Temp)**2 * Stress,                 
    Stress**2 / Temp,                     
    
    # Heat interactions
    Heat * (1/Temp),                     
    Heat * np.log(Stress),
    Heat * Stress,                        
    
    # Advanced physics
    Stress * np.exp(-1/Temp),             
    np.log(Stress) / Temp,                
    (Stress**(-1)) * (1/Temp),           
])

feature_names = [
    'Const', 'Heat', 'Heat²', 
    'T', '1/T', 'T²', '1/T²', 'log(T)', 'T^-0.5',
    'σ', 'log(σ)', 'σ²', 'σ³', 'σ⁻¹', 'σ⁻²', 'σ⁻³', '√σ', 'σ^0.5',
    '(1/T)σ', '(1/T)log(σ)', 'T·log(σ)', 'log(T)·log(σ)', '(1/T)²σ', 'σ²/T',
    'Heat·(1/T)', 'Heat·log(σ)', 'Heat·σ',
    'σ·exp(-1/T)', 'log(σ)/T', 'σ⁻¹·(1/T)'
]

y_log = np.log(Time)

# =====================================
# FEATURE SELECTION
# =====================================
print(f"\n{'='*80}")
print("FEATURE SELECTION")
print(f"{'='*80}")

selector = SelectKBest(f_regression, k=15)
X_selected = selector.fit_transform(X_features, y_log)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

print(f"Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# =====================================
# DUAL SCALING APPROACH
# =====================================
print(f"\n{'='*80}")
print("FEATURE SCALING")
print(f"{'='*80}")

# Standard scaling
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_selected)

# Quantile transformation (robust to outliers)
scaler_quantile = QuantileTransformer(output_distribution='normal', random_state=42)
X_quantile = scaler_quantile.fit_transform(X_selected)

print("Testing both Standard and Quantile scaling...")

# =====================================
# STRATIFIED SPLIT
# =====================================
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(sss.split(X_standard, time_bins))

X_train_std = X_standard[train_idx]
X_test_std = X_standard[test_idx]
X_train_qnt = X_quantile[train_idx]
X_test_qnt = X_quantile[test_idx]
y_train = y_log[train_idx]
y_test = y_log[test_idx]

print(f"\nStratified split ensures balanced representation:")
print(f"  Training set: {len(train_idx)} samples")
print(f"  Test set: {len(test_idx)} samples")

# =====================================
# OPTIMIZED MODEL COMPARISON
# =====================================
print(f"\n{'='*80}")
print("OPTIMIZED MODEL COMPARISON")
print(f"{'='*80}")

# Test both scalings
scaling_results = {}

for scaling_name, X_train, X_test in [
    ('Standard', X_train_std, X_test_std),
    ('Quantile', X_train_qnt, X_test_qnt)
]:
    
    # Optimized Ridge
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
    ridge_grid.fit(X_train, y_train)
    ridge_best = ridge_grid.best_estimator_
    
    # Optimized Random Forest with reduced overfitting
    rf_best = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features='sqrt',
        random_state=42
    )
    rf_best.fit(X_train, y_train)
    
    # Weighted ensemble
    ensemble = VotingRegressor([
        ('ridge', ridge_best),
        ('ridge2', Ridge(alpha=ridge_grid.best_params_['alpha'])),
        ('rf', rf_best)
    ])
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    for name, model in [('Ridge', ridge_best), ('Random Forest', rf_best), ('Ensemble', ensemble)]:
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        Time_test_vals = np.exp(y_test)
        Time_pred_vals = np.exp(y_pred)
        errors = np.abs((Time_pred_vals - Time_test_vals) / Time_test_vals) * 100
        
        # Get test set temperatures
        Temp_test = Temp[test_idx]
        
        # Analyze short vs long duration performance
        short_mask = Time_test_vals < 100
        short_errors = errors[short_mask] if short_mask.sum() > 0 else np.array([np.nan])
        long_errors = errors[~short_mask] if (~short_mask).sum() > 0 else np.array([np.nan])
        
        # Analyze by temperature ranges
        temp_errors = {}
        for temp_min, temp_max, label in temp_ranges:
            temp_mask = (Temp_test >= temp_min) & (Temp_test < temp_max)
            if temp_mask.sum() > 0:
                temp_errors[label] = {
                    'errors': errors[temp_mask],
                    'median': np.median(errors[temp_mask]),
                    'count': temp_mask.sum()
                }
            else:
                temp_errors[label] = {
                    'errors': np.array([np.nan]),
                    'median': np.nan,
                    'count': 0
                }
        
        key = f"{scaling_name}_{name}"
        scaling_results[key] = {
            'model': model,
            'scaling': scaling_name,
            'test_r2': test_r2,
            'median_error': np.median(errors),
            'short_median': np.median(short_errors),
            'long_median': np.median(long_errors),
            'temp_errors': temp_errors
        }
        
        print(f"\n{scaling_name} Scaling - {name}:")
        print(f"  Test R²:           {test_r2:.4f}")
        print(f"  Overall Median:    {np.median(errors):.1f}%")
        print(f"  Short (<100h):     {np.median(short_errors):.1f}% ({short_mask.sum()} samples)")
        print(f"  Long (≥100h):      {np.median(long_errors):.1f}% ({(~short_mask).sum()} samples)")
        print(f"  Temperature ranges:")
        for temp_min, temp_max, label in temp_ranges:
            info = temp_errors[label]
            print(f"    {label} ({temp_min:.0f}-{temp_max:.0f}K): {info['median']:.1f}% ({info['count']} samples)")

# Select best overall model
best_key = max(scaling_results.items(), key=lambda x: x[1]['test_r2'])[0]
best_result = scaling_results[best_key]
best_model = best_result['model']
best_scaling = best_result['scaling']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_key}")
print(f"{'='*80}")

# =====================================
# FINAL PREDICTIONS WITH BEST MODEL
# =====================================
X_train_final = X_train_std if best_scaling == 'Standard' else X_train_qnt
X_test_final = X_test_std if best_scaling == 'Standard' else X_test_qnt

y_pred_train = best_model.predict(X_train_final)
y_pred_test = best_model.predict(X_test_final)

Time_train_vals = np.exp(y_train)
Time_pred_train = np.exp(y_pred_train)
Time_test_vals = np.exp(y_test)
Time_pred_test = np.exp(y_pred_test)

train_errors = np.abs((Time_pred_train - Time_train_vals) / Time_train_vals) * 100
test_errors = np.abs((Time_pred_test - Time_test_vals) / Time_test_vals) * 100

print(f"\nFinal Performance:")
print(f"  Test R²:           {best_result['test_r2']:.4f}")
print(f"  Median Error:      {np.median(test_errors):.1f}%")
print(f"  25th percentile:   {np.percentile(test_errors, 25):.1f}%")
print(f"  75th percentile:   {np.percentile(test_errors, 75):.1f}%")
print(f"  90th percentile:   {np.percentile(test_errors, 90):.1f}%")

# Within factor-of-2 accuracy
within_2x = (test_errors <= 100).sum()
print(f"\nPredictions within ±100% (factor of 2): {within_2x}/{len(test_errors)} ({100*within_2x/len(test_errors):.1f}%)")

# =====================================
# EXPORT PREDICTIONS TO CSV
# =====================================
print(f"\n{'='*80}")
print("EXPORTING PREDICTIONS TO CSV")
print(f"{'='*80}")

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "Test_Output"
output_dir.mkdir(exist_ok=True)

# Prepare test set data
test_data = {
    'Heat': Heat[test_idx],
    'Temperature_K': Temp[test_idx],
    'Stress_MPa': Stress[test_idx],
    'Actual_Time_h': Time_test_vals,
    'Predicted_Time_h': Time_pred_test,
    'Error_Percent': test_errors,
    'Actual_log_Time': y_test,
    'Predicted_log_Time': y_pred_test,
    'Residual': y_test - y_pred_test
}

# Add categorical labels
duration_labels = ['Short' if t < 100 else 'Long' for t in Time_test_vals]
test_data['Duration_Category'] = duration_labels

# Add temperature range labels
temp_labels_list = []
for temp in Temp[test_idx]:
    for temp_min, temp_max, label in temp_ranges:
        if temp >= temp_min and temp < temp_max:
            temp_labels_list.append(label)
            break
test_data['Temperature_Range'] = temp_labels_list

# Create DataFrame
df_test = pd.DataFrame(test_data)

# Sort by actual time for easier analysis
df_test = df_test.sort_values('Actual_Time_h')

# Save test set predictions
test_csv_path = output_dir / "predictions_test_set.csv"
df_test.to_csv(test_csv_path, index=False, float_format='%.6f')
print(f"Test set predictions saved to: {test_csv_path}")
print(f"  Total samples: {len(df_test)}")

# Prepare training set data
train_data = {
    'Heat': Heat[train_idx],
    'Temperature_K': Temp[train_idx],
    'Stress_MPa': Stress[train_idx],
    'Actual_Time_h': Time_train_vals,
    'Predicted_Time_h': Time_pred_train,
    'Error_Percent': train_errors,
    'Actual_log_Time': y_train,
    'Predicted_log_Time': y_pred_train,
    'Residual': y_train - y_pred_train
}

# Add categorical labels for training set
duration_labels_train = ['Short' if t < 100 else 'Long' for t in Time_train_vals]
train_data['Duration_Category'] = duration_labels_train

temp_labels_list_train = []
for temp in Temp[train_idx]:
    for temp_min, temp_max, label in temp_ranges:
        if temp >= temp_min and temp < temp_max:
            temp_labels_list_train.append(label)
            break
train_data['Temperature_Range'] = temp_labels_list_train

# Create DataFrame
df_train = pd.DataFrame(train_data)
df_train = df_train.sort_values('Actual_Time_h')

# Save training set predictions
train_csv_path = output_dir / "predictions_train_set.csv"
df_train.to_csv(train_csv_path, index=False, float_format='%.6f')
print(f"Training set predictions saved to: {train_csv_path}")
print(f"  Total samples: {len(df_train)}")

# Create a combined file with all predictions
df_train['Dataset'] = 'Train'
df_test['Dataset'] = 'Test'
df_combined = pd.concat([df_train, df_test], ignore_index=True)
df_combined = df_combined.sort_values('Actual_Time_h')

combined_csv_path = output_dir / "predictions_all_data.csv"
df_combined.to_csv(combined_csv_path, index=False, float_format='%.6f')
print(f"Combined predictions saved to: {combined_csv_path}")
print(f"  Total samples: {len(df_combined)}")

# Create summary statistics CSV
summary_stats = {
    'Category': [],
    'N_Samples': [],
    'Median_Error_%': [],
    'Mean_Error_%': [],
    'Std_Error_%': [],
    'Min_Error_%': [],
    'Max_Error_%': [],
    'Within_Factor2_%': []
}

# Overall statistics
summary_stats['Category'].append('Overall_Test')
summary_stats['N_Samples'].append(len(test_errors))
summary_stats['Median_Error_%'].append(np.median(test_errors))
summary_stats['Mean_Error_%'].append(np.mean(test_errors))
summary_stats['Std_Error_%'].append(np.std(test_errors))
summary_stats['Min_Error_%'].append(np.min(test_errors))
summary_stats['Max_Error_%'].append(np.max(test_errors))
summary_stats['Within_Factor2_%'].append(100 * (test_errors <= 100).sum() / len(test_errors))

# By duration
for duration, mask in [('Short_<100h', Time_test_vals < 100), ('Long_≥100h', Time_test_vals >= 100)]:
    if mask.sum() > 0:
        errors_subset = test_errors[mask]
        summary_stats['Category'].append(duration)
        summary_stats['N_Samples'].append(mask.sum())
        summary_stats['Median_Error_%'].append(np.median(errors_subset))
        summary_stats['Mean_Error_%'].append(np.mean(errors_subset))
        summary_stats['Std_Error_%'].append(np.std(errors_subset))
        summary_stats['Min_Error_%'].append(np.min(errors_subset))
        summary_stats['Max_Error_%'].append(np.max(errors_subset))
        summary_stats['Within_Factor2_%'].append(100 * (errors_subset <= 100).sum() / len(errors_subset))

# By temperature range
Temp_test = Temp[test_idx]
for temp_min, temp_max, label in temp_ranges:
    temp_mask = (Temp_test >= temp_min) & (Temp_test < temp_max)
    if temp_mask.sum() > 0:
        errors_subset = test_errors[temp_mask]
        summary_stats['Category'].append(f'Temp_{label}_{temp_min:.0f}-{temp_max:.0f}K')
        summary_stats['N_Samples'].append(temp_mask.sum())
        summary_stats['Median_Error_%'].append(np.median(errors_subset))
        summary_stats['Mean_Error_%'].append(np.mean(errors_subset))
        summary_stats['Std_Error_%'].append(np.std(errors_subset))
        summary_stats['Min_Error_%'].append(np.min(errors_subset))
        summary_stats['Max_Error_%'].append(np.max(errors_subset))
        summary_stats['Within_Factor2_%'].append(100 * (errors_subset <= 100).sum() / len(errors_subset))

# Combined duration × temperature
short_mask = Time_test_vals < 100
for temp_min, temp_max, label in temp_ranges:
    temp_mask = (Temp_test >= temp_min) & (Temp_test < temp_max)
    
    # Short duration + temp range
    combined_mask = short_mask & temp_mask
    if combined_mask.sum() > 0:
        errors_subset = test_errors[combined_mask]
        summary_stats['Category'].append(f'Short_{label}_{temp_min:.0f}-{temp_max:.0f}K')
        summary_stats['N_Samples'].append(combined_mask.sum())
        summary_stats['Median_Error_%'].append(np.median(errors_subset))
        summary_stats['Mean_Error_%'].append(np.mean(errors_subset))
        summary_stats['Std_Error_%'].append(np.std(errors_subset))
        summary_stats['Min_Error_%'].append(np.min(errors_subset))
        summary_stats['Max_Error_%'].append(np.max(errors_subset))
        summary_stats['Within_Factor2_%'].append(100 * (errors_subset <= 100).sum() / len(errors_subset))
    
    # Long duration + temp range
    combined_mask = (~short_mask) & temp_mask
    if combined_mask.sum() > 0:
        errors_subset = test_errors[combined_mask]
        summary_stats['Category'].append(f'Long_{label}_{temp_min:.0f}-{temp_max:.0f}K')
        summary_stats['N_Samples'].append(combined_mask.sum())
        summary_stats['Median_Error_%'].append(np.median(errors_subset))
        summary_stats['Mean_Error_%'].append(np.mean(errors_subset))
        summary_stats['Std_Error_%'].append(np.std(errors_subset))
        summary_stats['Min_Error_%'].append(np.min(errors_subset))
        summary_stats['Max_Error_%'].append(np.max(errors_subset))
        summary_stats['Within_Factor2_%'].append(100 * (errors_subset <= 100).sum() / len(errors_subset))

df_summary = pd.DataFrame(summary_stats)
summary_csv_path = output_dir / "prediction_summary_statistics.csv"
df_summary.to_csv(summary_csv_path, index=False, float_format='%.2f')
print(f"Summary statistics saved to: {summary_csv_path}")
print(f"  Categories analyzed: {len(df_summary)}")

print(f"\nCSV Export Complete!")
print(f"{'='*80}")

# =====================================
# COMPREHENSIVE VISUALIZATION
# =====================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Get test set temperatures
Temp_test = Temp[test_idx]

# Plot 1: Log-space predictions
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_pred_train, alpha=0.4, s=30, label='Train', color='blue')
ax1.scatter(y_test, y_pred_test, alpha=0.6, s=50, label='Test', 
            edgecolors='black', linewidth=0.5, color='red')
ax1.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()], 
         'k--', linewidth=2, label='Perfect')
ax1.set_xlabel("Measured log(Time)", fontsize=11)
ax1.set_ylabel("Predicted log(Time)", fontsize=11)
ax1.set_title(f"Log-Space Predictions (R²={best_result['test_r2']:.3f})", fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Real-space (log-log)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(Time_train_vals, Time_pred_train, alpha=0.4, s=30, label='Train', color='blue')
ax2.scatter(Time_test_vals, Time_pred_test, alpha=0.6, s=50, label='Test',
            edgecolors='black', linewidth=0.5, color='red')
ax2.plot([Time.min(), Time.max()], [Time.min(), Time.max()], 
         'k--', linewidth=2, label='Perfect')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Measured Time (h)", fontsize=11)
ax2.set_ylabel("Predicted Time (h)", fontsize=11)
ax2.set_title("Real-Space (Log-Log)", fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Residuals
ax3 = fig.add_subplot(gs[0, 2])
residuals_test = y_test - y_pred_test
ax3.scatter(y_pred_test, residuals_test, alpha=0.6, s=50, 
            edgecolors='black', linewidth=0.5)
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel("Predicted log(Time)", fontsize=11)
ax3.set_ylabel("Residual", fontsize=11)
ax3.set_title("Residuals (Test Set)", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(test_errors, bins=20, edgecolor='black', alpha=0.7)
ax4.axvline(np.median(test_errors), color='red', linestyle='--', 
            linewidth=2, label=f'Median: {np.median(test_errors):.1f}%')
ax4.set_xlabel("Prediction Error (%)", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title("Error Distribution", fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Feature importance (for Ridge model)
ax5 = fig.add_subplot(gs[1, 1])
if hasattr(best_model, 'coef_'):
    coef_abs = np.abs(best_model.coef_)
    top_n = 10
    top_indices = np.argsort(coef_abs)[-top_n:]
    ax5.barh(range(top_n), coef_abs[top_indices])
    ax5.set_yticks(range(top_n))
    ax5.set_yticklabels([selected_features[i] for i in top_indices])
    ax5.set_xlabel("Absolute Coefficient", fontsize=11)
    ax5.set_title("Top 10 Features", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
else:
    # If not Ridge, show scaling comparison
    scaling_types = ['Standard', 'Quantile']
    model_types = ['Ridge', 'Random Forest', 'Ensemble']
    x_pos = np.arange(len(model_types))
    width = 0.35
    std_r2 = [scaling_results[f"Standard_{m}"]['test_r2'] for m in model_types]
    qnt_r2 = [scaling_results[f"Quantile_{m}"]['test_r2'] for m in model_types]
    ax5.bar(x_pos - width/2, std_r2, width, label='Standard', color='steelblue', edgecolor='black')
    ax5.bar(x_pos + width/2, qnt_r2, width, label='Quantile', color='orange', edgecolor='black')
    ax5.set_ylabel("Test R²", fontsize=11)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(model_types, rotation=15, ha='right')
    ax5.set_title("Model Comparison", fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Error vs Temperature
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(Temp_test, test_errors, alpha=0.6, s=50, 
            edgecolors='black', linewidth=0.5)
ax6.set_xlabel("Temperature (K)", fontsize=11)
ax6.set_ylabel("Prediction Error (%)", fontsize=11)
ax6.set_title("Error vs Temperature", fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Error vs Stress
ax7 = fig.add_subplot(gs[2, 0])
stress_test = Stress[test_idx]
ax7.scatter(stress_test, test_errors, alpha=0.6, s=50,
            edgecolors='black', linewidth=0.5)
ax7.set_xlabel("Stress (MPa)", fontsize=11)
ax7.set_ylabel("Prediction Error (%)", fontsize=11)
ax7.set_title("Error vs Stress", fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Short vs Long duration comparison
ax8 = fig.add_subplot(gs[2, 1])
short_mask = Time_test_vals < 100
categories = ['Short\n(<100h)', 'Long\n(≥100h)']
median_errors_plot = [np.median(test_errors[short_mask]), np.median(test_errors[~short_mask])]
sample_counts = [short_mask.sum(), (~short_mask).sum()]
bars = ax8.bar(categories, median_errors_plot, color=['coral', 'lightgreen'], 
               edgecolor='black', linewidth=2, alpha=0.7)
ax8.set_ylabel("Median Error (%)", fontsize=11)
ax8.set_title("Performance by Duration", fontsize=12, fontweight='bold')
for bar, count in zip(bars, sample_counts):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%\n(n={count})',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Temperature Range Performance
ax9 = fig.add_subplot(gs[2, 2])
temp_labels = [label for _, _, label in temp_ranges]
temp_medians = [best_result['temp_errors'][label]['median'] for label in temp_labels]
temp_counts = [best_result['temp_errors'][label]['count'] for label in temp_labels]
colors = ['steelblue', 'coral', 'lightgreen']
bars = ax9.bar(temp_labels, temp_medians, color=colors, 
               edgecolor='black', linewidth=2, alpha=0.7)
ax9.set_ylabel("Median Error (%)", fontsize=11)
ax9.set_title("Performance by Temperature Range", fontsize=12, fontweight='bold')
for bar, count in zip(bars, temp_counts):
    height = bar.get_height()
    if not np.isnan(height):
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%\n(n={count})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# Plot 10: Prediction range visualization
ax10 = fig.add_subplot(gs[3, 0])
sorted_idx = np.argsort(Time_test_vals)
x_pos = np.arange(len(Time_test_vals))
ax10.plot(x_pos, Time_test_vals[sorted_idx], 'ko-', label='Actual', markersize=6, linewidth=2)
ax10.plot(x_pos, Time_pred_test[sorted_idx], 'rs-', label='Predicted', 
         markersize=6, linewidth=2, alpha=0.7)
ax10.fill_between(x_pos, Time_test_vals[sorted_idx]*0.5, Time_test_vals[sorted_idx]*2,
                  alpha=0.2, color='gray', label='±100% band')
ax10.set_yscale('log')
ax10.set_xlabel("Sample Index (sorted by actual time)", fontsize=11)
ax10.set_ylabel("Time to 1% Strain (h)", fontsize=11)
ax10.set_title("Predictions Sorted by Actual Time", fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3, which='both')

# Plot 11: Error by Temperature and Duration
ax11 = fig.add_subplot(gs[3, 1])
short_mask = Time_test_vals < 100
for temp_min, temp_max, label in temp_ranges:
    temp_mask = (Temp_test >= temp_min) & (Temp_test < temp_max)
    
    short_temp = short_mask & temp_mask
    long_temp = (~short_mask) & temp_mask
    
    if short_temp.sum() > 0:
        ax11.scatter(Temp_test[short_temp], test_errors[short_temp], 
                    alpha=0.7, s=80, marker='o', label=f'{label} Short', edgecolors='black')
    if long_temp.sum() > 0:
        ax11.scatter(Temp_test[long_temp], test_errors[long_temp], 
                    alpha=0.7, s=80, marker='s', label=f'{label} Long', edgecolors='black')

ax11.set_xlabel("Temperature (K)", fontsize=11)
ax11.set_ylabel("Prediction Error (%)", fontsize=11)
ax11.set_title("Error vs Temp (by Duration)", fontsize=12, fontweight='bold')
ax11.legend(fontsize=8, ncol=2)
ax11.grid(True, alpha=0.3)

# Plot 12: Combined comparison bar chart
ax12 = fig.add_subplot(gs[3, 2])
x = np.arange(len(temp_labels))
width = 0.35

short_temp_medians = []
long_temp_medians = []

for temp_min, temp_max, label in temp_ranges:
    temp_mask = (Temp_test >= temp_min) & (Temp_test < temp_max)
    short_temp = short_mask & temp_mask
    long_temp = (~short_mask) & temp_mask
    
    if short_temp.sum() > 0:
        short_temp_medians.append(np.median(test_errors[short_temp]))
    else:
        short_temp_medians.append(np.nan)
        
    if long_temp.sum() > 0:
        long_temp_medians.append(np.median(test_errors[long_temp]))
    else:
        long_temp_medians.append(np.nan)

bars1 = ax12.bar(x - width/2, short_temp_medians, width, label='Short (<100h)', 
                 color='coral', edgecolor='black', alpha=0.7)
bars2 = ax12.bar(x + width/2, long_temp_medians, width, label='Long (≥100h)', 
                 color='lightgreen', edgecolor='black', alpha=0.7)

ax12.set_ylabel("Median Error (%)", fontsize=11)
ax12.set_title("Duration × Temperature Performance", fontsize=12, fontweight='bold')
ax12.set_xticks(x)
ax12.set_xticklabels(temp_labels)
ax12.legend()
ax12.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax12.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=8)

plt.suptitle(f'SS316H Creep Model - With Temperature Range Analysis\nBest: {best_key} (R²={best_result["test_r2"]:.3f})', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_dir / "1percent_final_temp_ranges.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print("FINAL SUMMARY WITH TEMPERATURE RANGES")
print(f"{'='*80}")
print(f"Best configuration: {best_key}")
print(f"Test R²: {best_result['test_r2']:.4f}")
print(f"Overall median error: {best_result['median_error']:.1f}%")
print(f"\nBy Duration:")
print(f"  Short (<100h):  {best_result['short_median']:.1f}%")
print(f"  Long (≥100h):   {best_result['long_median']:.1f}%")
print(f"\nBy Temperature:")
for temp_min, temp_max, label in temp_ranges:
    info = best_result['temp_errors'][label]
    print(f"  {label} ({temp_min:.0f}-{temp_max:.0f}K): {info['median']:.1f}% (n={info['count']})")
print(f"\nRecommendation: Use {best_key.split('_')[1]} model with {best_scaling} scaling")
print(f"Apply ±2x safety factor for engineering applications")
print(f"\nFiles saved:")
print(f"  - Plot: Test_Output/1percent_final_temp_ranges.png")
print(f"  - Test predictions: {test_csv_path}")
print(f"  - Train predictions: {train_csv_path}")
print(f"  - Combined predictions: {combined_csv_path}")
print(f"  - Summary statistics: {summary_csv_path}")
print(f"{'='*80}")