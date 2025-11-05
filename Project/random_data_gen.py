# synth_gen_materials.py
"""
Synthetic Material Test Data Generator for 316H SS and Alloy 617
Adapted for SINDy project on nuclear materials

Generates realistic material behavior datasets with:
- Known governing equations (ground truth)
- Manufacturing/test condition variability
- Realistic noise and measurement artifacts
- Multiple material systems (creep, tensile, oxidation)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os
import json


# ============================================================================
# MATERIAL CONSTITUTIVE MODELS (Right-Hand Side Functions)
# ============================================================================

def norton_bailey_creep(t,x,params):
    """
    Norton-bailey creep Law for 316H SS
    
    Governing equations:
    dε/dt = A * σ^n * exp(-Q/RT) + primary_term + tertiary_term

    State: x = [Strain, Damage]
    Parameters: [A, N, Q, R, T, sigma, k_primary, k_tertiary]
    """
    A, n, Q, R, T, sigma, k_primary, k_tertiary = params
    strain, damage = x
    
    # Secondary (steady-state) creep rate
    secondary_rate = A * (sigma ** n) * np.exp(-Q / (R * T))
    
    # Primary creep (decreases with time)
    primary_rate = k_primary * np.exp(-t / 10) * secondary_rate
    
    # Tertiary creep (increases with damage)
    tertiary_rate = k_tertiary * damage * secondary_rate
    
    # Total strain rate
    d_strain = secondary_rate + primary_rate + tertiary_rate
    
    # Damage accumulation (simplified)
    d_damage = 0.0001 * strain * secondary_rate
    
    return[d_strain, d_damage]

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def simulate_system(rhs, x0, params, t_span=(0,10), dt=0.01, 
                   noise_std=0.0, jitter_timestamps=False, missing_rate=0.0,
                   relative_noise=True):
    """
    Simulate material system with realistic measurement artifacts.
    
    Parameters:
    -----------
    rhs : callable
        Right-hand side function (constitutive model)
    x0 : list
        Initial conditions
    params : list
        Model parameters
    relative_noise : bool
        If True, noise scales with signal magnitude (more realistic)
    """
    t_eval = np.arange(t_span[0], t_span[1]+dt/2, dt)
    
    # Solve ODE
    sol = solve_ivp(lambda tt, xx: rhs(tt, xx, params), t_span, x0, 
                   t_eval=t_eval, rtol=1e-8, atol=1e-10, method='RK45')
    
    X = sol.y.T  # shape (N, n_states)
    
    # Add noise
    if noise_std > 0:
        if relative_noise:
            # Noise proportional to signal magnitude (typical in material testing)
            for i in range(X.shape[1]):
                signal_range = np.ptp(X[:, i])  # peak-to-peak
                noise = np.random.normal(0, noise_std * signal_range, size=X[:, i].shape)
                X[:, i] = X[:, i] + noise
        else:
            noise = np.random.normal(0, noise_std, size=X.shape)
            X = X + noise
    
    # Jitter timestamps (equipment timing variations)
    if jitter_timestamps:
        jitter = np.random.normal(0, dt*0.2, size=len(t_eval))
        t_eval = t_eval + jitter
        t_eval = np.sort(t_eval)
    
    # Missing data points (sensor dropouts, data acquisition issues)
    mask = np.ones(len(t_eval), dtype=bool)
    if missing_rate > 0:
        mask = np.random.rand(len(t_eval)) > missing_rate
    
    # Create dataframe
    df = pd.DataFrame(X[mask], columns=[f"x{i}" for i in range(X.shape[1])])
    df.insert(0, 't', t_eval[mask])
    
    return df
        
# ============================================================================
# MATERIAL-SPECIFIC DATASET GENERATORS
# ============================================================================

def generate_creep_tests_316H(out_dir='synth_data/316H_creep', n_tests=10):
    """
    Generate creep test data for 316H Stainless Steel
    Varying stress and temperature conditions
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 316H SS creep parameters (literature-based ranges)
    base_params = {
        'A': 1e-15,      # Material constant (1/MPa^n/hr)
        'n': 5.0,        # Stress exponent (typically 4-7 for 316H)
        'Q': 300000.0,   # Activation energy (J/mol)
        'R': 8.314,      # Gas constant
        'k_primary': 0.5,
        'k_tertiary': 0.3
    }
    
    datasets_info = []
    
    for i in range(n_tests):
        # Vary test conditions (simulating different experiments)
        T = np.random.uniform(873, 973)  # 600-700°C
        sigma = np.random.uniform(80, 200)  # 80-200 MPa
        
        # Add parameter variability (different heats/manufacturing)
        A = base_params['A'] * (1 + np.random.uniform(-0.2, 0.2))
        n = base_params['n'] + np.random.uniform(-0.5, 0.5)
        Q = base_params['Q'] * (1 + np.random.uniform(-0.1, 0.1))
        
        params = [A, n, Q, base_params['R'], T, sigma, 
                 base_params['k_primary'], base_params['k_tertiary']]
        
        x0 = [0.0, 0.0]  # [initial_strain, initial_damage]
        
        # Vary test duration and data quality
        t_max = np.random.choice([500, 1000, 2000])  # hours
        noise_std = np.random.choice([0.01, 0.02, 0.03])
        missing_rate = np.random.choice([0.0, 0.01, 0.02])
        jitter = np.random.choice([False, True], p=[0.6, 0.4])
        
        df = simulate_system(norton_bailey_creep, x0, params, 
                           t_span=(0, t_max), dt=1.0,
                           noise_std=noise_std, jitter_timestamps=jitter, 
                           missing_rate=missing_rate)
        
        # Rename columns to physical quantities
        df.rename(columns={'x0': 'strain', 'x1': 'damage'}, inplace=True)
        df['stress'] = sigma
        df['temperature'] = T
        
        # Save data
        name = f"creep_316H_{i}_T{int(T)}K_S{int(sigma)}MPa"
        df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
        
        # Save metadata
        meta = {
            'material': '316H Stainless Steel',
            'test_type': 'Creep',
            'model': 'Norton-Bailey with primary and tertiary stages',
            'equation': 'dε/dt = A*σ^n*exp(-Q/RT) + primary + tertiary',
            'parameters': {
                'A': float(A), 'n': float(n), 'Q': float(Q), 'R': float(base_params['R']),
                'stress_MPa': float(sigma), 'temperature_K': float(T)
            },
            'data_quality': {
                'noise_level': float(noise_std),
                'missing_rate': float(missing_rate),
                'time_jitter': bool(jitter),
                'duration_hours': int(t_max)
            }
        }
        
        with open(os.path.join(out_dir, name + "_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        
        datasets_info.append(meta)
    
    print(f"✓ Generated {n_tests} creep tests for 316H SS in {out_dir}")
    return datasets_info


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_material_datasets(base_dir='synth_data_materials', 
                                   n_per_type=10):
    """
    Generate complete suite of material test datasets for SINDy validation.
    """
    print("="*70)
    print("Generating Material Test Datasets for SINDy Project")
    print("Materials: 316H SS and Alloy 617")
    print("="*70)
    
    all_info = {}
    
    # 316H Creep tests
    print("\n[1/1] Generating 316H Stainless Steel Creep Tests...")
    all_info['316H_creep'] = generate_creep_tests_316H(
        os.path.join(base_dir, '316H_creep'), n_per_type
    )
    
    print("\n" + "="*70)
    print(f"✓ Complete! Generated {sum(len(v) for v in all_info.values())} datasets")
    print(f"✓ Location: {os.path.abspath(base_dir)}")
    print("="*70)
    
    return all_info
    
if __name__ == "__main__":
    # Generate all datasets
    generate_all_material_datasets(n_per_type=10)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Explore the generated CSV files in each subdirectory")
    print("2. Check *_meta.json files for ground truth equations")
    print("3. Use MaterialDataPreprocessor to prepare data")
    print("4. Apply SINDy and compare results to ground truth")
    print("5. Tune preprocessing parameters based on validation results")
    print("="*70)
    