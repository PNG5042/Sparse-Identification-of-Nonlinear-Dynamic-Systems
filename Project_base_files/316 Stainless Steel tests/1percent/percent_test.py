# ================================
# FINAL OPTIMIZED SS316H CREEP MODEL
# Addresses short-duration prediction issues
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

# Load data
df = pd.read_csv(r"C:\Users\phili\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Test_data\SS316H-1percent.csv")

print("="*80)
print("SS316H CREEP MODEL - FINAL OPTIMIZED VERSION")
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

# =====================================
# STRATIFIED SAMPLING
# =====================================
print(f"\n{'='*80}")
print("STRATIFIED TRAIN/TEST SPLIT")
print(f"{'='*80}")

# Create strata based on time duration (short vs long tests)
time_bins = np.digitize(np.log10(Time), bins=[0, 1, 2, 3, 4, 5, 6])
print(f"Time bins distribution:")
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
    np.log(Temp) * np.log(Stress),        # NEW
    (1/Temp)**2 * Stress,                 # NEW - strong temperature effect
    Stress**2 / Temp,                     # NEW
    
    # Heat interactions
    Heat * (1/Temp),                     
    Heat * np.log(Stress),
    Heat * Stress,                        # NEW
    
    # Advanced physics
    Stress * np.exp(-1/Temp),             # Thermally activated
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
# FEATURE SELECTION - More features for complex behavior
# =====================================
print(f"\n{'='*80}")
print("FEATURE SELECTION")
print(f"{'='*80}")

selector = SelectKBest(f_regression, k=15)  # Increased from 12
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
        max_depth=6,           # Reduced from 8
        min_samples_split=12,  # Increased from 8
        min_samples_leaf=6,    # Increased from 4
        max_features='sqrt',   # Use sqrt instead of all features
        random_state=42
    )
    rf_best.fit(X_train, y_train)
    
    # Weighted ensemble - more weight to Ridge for stability
    ensemble = VotingRegressor([
        ('ridge', ridge_best),
        ('ridge2', Ridge(alpha=ridge_grid.best_params_['alpha'])),  # Double Ridge weight
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
        
        # Analyze short vs long duration performance
        short_mask = Time_test_vals < 100
        short_errors = errors[short_mask] if short_mask.sum() > 0 else np.array([np.nan])
        long_errors = errors[~short_mask] if (~short_mask).sum() > 0 else np.array([np.nan])
        
        key = f"{scaling_name}_{name}"
        scaling_results[key] = {
            'model': model,
            'scaling': scaling_name,
            'test_r2': test_r2,
            'median_error': np.median(errors),
            'short_median': np.median(short_errors),
            'long_median': np.median(long_errors)
        }
        
        print(f"\n{scaling_name} Scaling - {name}:")
        print(f"  Test R²:           {test_r2:.4f}")
        print(f"  Overall Median:    {np.median(errors):.1f}%")
        print(f"  Short (<100h):     {np.median(short_errors):.1f}% ({short_mask.sum()} samples)")
        print(f"  Long (≥100h):      {np.median(long_errors):.1f}% ({(~short_mask).sum()} samples)")

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
# COMPREHENSIVE VISUALIZATION
# =====================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

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
temps_test = Temp[test_idx]
ax6.scatter(temps_test, test_errors, alpha=0.6, s=50, 
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

# Plot 9: Prediction range visualization
ax9 = fig.add_subplot(gs[2, 2])
sorted_idx = np.argsort(Time_test_vals)
x_pos = np.arange(len(Time_test_vals))
ax9.plot(x_pos, Time_test_vals[sorted_idx], 'ko-', label='Actual', markersize=6, linewidth=2)
ax9.plot(x_pos, Time_pred_test[sorted_idx], 'rs-', label='Predicted', 
         markersize=6, linewidth=2, alpha=0.7)
ax9.fill_between(x_pos, Time_test_vals[sorted_idx]*0.5, Time_test_vals[sorted_idx]*2,
                  alpha=0.2, color='gray', label='±100% band')
ax9.set_yscale('log')
ax9.set_xlabel("Sample Index (sorted by actual time)", fontsize=11)
ax9.set_ylabel("Time to 1% Strain (h)", fontsize=11)
ax9.set_title("Predictions Sorted by Actual Time", fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3, which='both')

plt.suptitle(f'SS316H Creep Model - Final Optimized\nBest: {best_key} (R²={best_result["test_r2"]:.3f})', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig("Test_Output/1percent_final.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
print(f"Best configuration: {best_key}")
print(f"Test R²: {best_result['test_r2']:.4f}")
print(f"Overall median error: {best_result['median_error']:.1f}%")
print(f"Short-duration median: {best_result['short_median']:.1f}%")
print(f"Long-duration median: {best_result['long_median']:.1f}%")
print(f"\nRecommendation: Use {best_key.split('_')[1]} model with {best_scaling} scaling")
print(f"Apply ±2x safety factor for engineering applications")
print(f"\nPlot saved as 'Test_Output/1percent_final.png'")
print(f"{'='*80}")