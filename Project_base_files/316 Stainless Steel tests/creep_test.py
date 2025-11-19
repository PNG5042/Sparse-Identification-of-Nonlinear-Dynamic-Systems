import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from mpl_toolkits.mplot3d import Axes3D
from pysindy.feature_library import CustomLibrary, PolynomialLibrary
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# Creep Test variables to track:

# Primary strain (ε) - your main observable
# Strain rate (dε/dt) - first derivative
# Time (t)
# Temperature (T) - constant for isothermal tests
# Applied stress (σ) - constant for standard creep tests
# Secondary variables: strain acceleration (d²ε/dt²)


# 1.Data preprocessing:

# Smooth noisy experimental data
strain_smooth = savgol_filter(strain_raw, window_length=51, polyorder=3)
# Estimate derivatives numerically
strain_rate = np.gradient(strain_smooth, time)

# 2. Custom library for creep phenomena:
poly_library = ps.PolynomialLibrary(degree=3)
fourier_library = ps.FourierLibrary(n_frequencies=2)
custom_library = ps.CustomLibrary(
    library_functions=[
        lambda x: np.exp(-x),
        lambda x: x**0.5,
        lambda x: np.log(x + 1e-6),
    ],
    function_names=[
        lambda x: f"exp(-{x})",
        lambda x: f"sqrt({x})",
        lambda x: f"log({x})"
    ]
)

# Combine libraries
library = poly_library + custom_library

# 3. Model Formulation Approaches

# Approach A: Primary, Secondary, Tertiary Creep Stages

# Segment your data by creep stage
primary_creep = data[time < t_primary]
secondary_creep = data[(time >= t_primary) & (time < t_secondary)]
tertiary_creep = data[time >= t_secondary]

# Fit separate models for each stage
model_primary = ps.SINDy(
    feature_library=library,
    optimizer=ps.STLSQ(threshold=0.05)
)
model_primary.fit(primary_creep, t=time_primary)

# Approach B: Unified Model with All Stages

# Single model across all creep stages
model_unified = ps.SINDy(
    feature_library=library,
    optimizer=ps.STLSQ(threshold=0.02, alpha=0.01)  # regularization
)
model_unified.fit(strain_data, t=time, x_dot=strain_rate)

# Approach C: Multi-Experiment Ensemble

# Multiple experiments at different stress/temperature
model_ensemble = ps.SINDy(
    feature_library=library,
    optimizer=ps.SR3(threshold=0.1)  # robust to outliers
)

# Stack data from multiple tests
X_multi = [strain_test1, strain_test2, strain_test3]
t_multi = [time_test1, time_test2, time_test3]

model_ensemble.fit(X_multi, t=t_multi, multiple_trajectories=True)

#  4. Physical Validation Strategy:

# Extract governing equation
model.print()
model.equations()

# Compare with theoretical models
coefficients = model.coefficients()

# Validate physical reasonableness
# - Positive strain rates
# - Monotonic strain increase
# - Stress/temperature dependencies match metallurgy

# 5. Handling Sparse Regression Challenges

# Try multiple optimizers to find best fit
optimizers = [
    ps.STLSQ(threshold=0.05),
    ps.SR3(threshold=0.1, nu=1.0),
    ps.SSR(criteria='model_size', kappa=1e-3),
    ps.FROLS(normalize_columns=True)
]

results = []
for opt in optimizers:
    model = ps.SINDy(feature_library=library, optimizer=opt)
    model.fit(strain_data, t=time)
    score = model.score(strain_data, t=time)
    results.append((opt, model, score))
    
# Select best model based on score and interpretability

# 6. Incorporating Temperature and Stress

# Multi-variable SINDy for parametric studies
# X = [strain, temperature, stress]
X = np.column_stack([strain, temp*np.ones_like(strain), stress*np.ones_like(strain)])

# Include interactions in library
library_parametric = ps.PolynomialLibrary(degree=2, include_interaction=True)

model_parametric = ps.SINDy(feature_library=library_parametric)
model_parametric.fit(X, t=time)

# 7. Validation Against Material Database

# Train on subset, validate on hold-out tests
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(all_tests, test_size=0.2)

model.fit(train_data['strain'], t=train_data['time'])

# Predict on test data
strain_predicted = model.simulate(test_data['strain'][0], t=test_data['time'])

# Calculate metrics
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(test_data['strain'], strain_predicted)
rmse = np.sqrt(mean_squared_error(test_data['strain'], strain_predicted))

# Generate publication-ready figures
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Experimental vs Predicted
axes[0,0].plot(time, strain_experimental, 'o', label='Experimental')
axes[0,0].plot(time, strain_predicted, '-', label='SINDy Model')
axes[0,0].set_xlabel('Time (h)')
axes[0,0].set_ylabel('Strain')

# Plot 2: Identified coefficients
model.print()

# Plot 3: Residuals
# Plot 4: Parametric study results