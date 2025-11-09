import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary

# ============================================================================
# TENSILE TEST MODEL: Stress-Strain Behavior for Alloy 617
# ============================================================================

class TensileTestModel:
    """
    Tensile test model for Alloy 617 incorporating:
    - Elastic region (Hooke's Law)
    - Yield transition (0.2% offset yield strength)
    - Plastic hardening (power law and Voce hardening)
    - Necking and failure
    """
    
    def __init__(self, temp=25, strain_rate=1e-3):
        """
        Parameters:
        - temp: Temperature (°C)
        - strain_rate: Strain rate (1/s)
        """
        self.temp = temp
        self.strain_rate = strain_rate
        
        # Temperature-dependent properties for Alloy 617
        if temp <= 100:
            self.E = 211000  # Young's modulus (MPa) at room temp
            self.yield_stress = 380  # Yield strength (MPa)
            self.K = 1200  # Strength coefficient (MPa)
            self.n = 0.35  # Strain hardening exponent
            self.UTS = 750  # Ultimate tensile strength (MPa)
        elif temp <= 500:
            self.E = 195000
            self.yield_stress = 320
            self.K = 1000
            self.n = 0.30
            self.UTS = 650
        elif temp <= 700:
            self.E = 180000
            self.yield_stress = 280
            self.K = 850
            self.n = 0.25
            self.UTS = 550
        else:  # High temperature (750-950°C)
            self.E = 170000
            self.yield_stress = 200
            self.K = 650
            self.n = 0.20
            self.UTS = 400
        
        # Calculate yield strain
        self.yield_strain = self.yield_stress / self.E
        
        # Necking parameters
        self.necking_strain = self.n  # Necking occurs at strain = n (Considère criterion)
        self.fracture_strain = self.necking_strain + 0.15  # Additional elongation in neck
        
    def stress_strain_response(self, strain):
        """
        Complete stress-strain curve with elastic, plastic, and necking regions
        """
        stress = np.zeros_like(strain)
        
        for i, eps in enumerate(strain):
            if eps <= self.yield_strain:
                # Elastic region (Hooke's Law)
                stress[i] = self.E * eps
                
            elif eps <= self.necking_strain:
                # Plastic region (Hollomon power law with smooth transition)
                # Use Ramberg-Osgood type transition
                eps_plastic = eps - self.yield_strain
                stress[i] = self.K * (eps_plastic + self.yield_strain) ** self.n
                
            elif eps <= self.fracture_strain:
                # Necking region (stress decreases)
                # Linear decrease from UTS to failure stress
                progress = (eps - self.necking_strain) / (self.fracture_strain - self.necking_strain)
                failure_stress = 0.7 * self.UTS  # Stress at fracture
                stress[i] = self.UTS - (self.UTS - failure_stress) * progress
                
            else:
                # Fracture
                stress[i] = 0
        
        return stress
    
    def add_noise(self, data, noise_level=0.02):
        """Add realistic measurement noise"""
        noise = np.random.normal(0, noise_level * np.std(data), len(data))
        return data + noise

# ============================================================================
# GENERATE SYNTHETIC TENSILE TEST DATA
# ============================================================================

def generate_tensile_dataset(temperatures, strain_rate=1e-3, n_points=1000):
    """
    Generate synthetic tensile test data for multiple temperatures
    """
    datasets = []
    
    for temp in temperatures:
        model = TensileTestModel(temp=temp, strain_rate=strain_rate)
        
        # Generate strain array up to fracture
        strain = np.linspace(0, model.fracture_strain, n_points)
        
        # Calculate stress response
        stress = model.stress_strain_response(strain)
        
        # Add realistic noise
        stress_noisy = model.add_noise(stress, noise_level=0.01)
        
        # Create dataframe
        df = pd.DataFrame({
            'strain': strain,
            'stress': stress_noisy,
            'true_stress': stress,
            'temperature': temp,
            'strain_rate': strain_rate
        })
        
        # Calculate engineering properties
        df['elastic_modulus'] = np.gradient(df['stress'], df['strain'])
        
        # Calculate true stress and true strain (for plastic region)
        df['true_strain'] = np.log(1 + df['strain'])
        df['true_stress_corrected'] = df['stress'] * (1 + df['strain'])
        
        datasets.append(df)
    
    return pd.concat(datasets, ignore_index=True)

# Generate data for multiple temperatures
temperatures = [25, 400, 650, 850]  # °C
strain_rate = 1e-3  # 1/s (quasi-static)

print("Generating synthetic tensile test data...")
tensile_data = generate_tensile_dataset(temperatures, strain_rate=strain_rate)

# ============================================================================
# MATERIALS PROPERTY EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("EXTRACTED MECHANICAL PROPERTIES")
print("="*70)

properties_summary = []

for temp in temperatures:
    data_subset = tensile_data[tensile_data['temperature'] == temp].copy()
    
    # Young's Modulus (initial slope)
    elastic_region = data_subset[data_subset['strain'] < 0.005]
    E_measured = np.polyfit(elastic_region['strain'], elastic_region['stress'], 1)[0]
    
    # Yield Strength (0.2% offset)
    offset_strain = 0.002
    offset_line = E_measured * (data_subset['strain'] - offset_strain)
    yield_idx = np.where(data_subset['stress'] > offset_line)[0]
    if len(yield_idx) > 0:
        yield_stress = data_subset.iloc[yield_idx[0]]['stress']
        yield_strain = data_subset.iloc[yield_idx[0]]['strain']
    else:
        yield_stress = np.nan
        yield_strain = np.nan
    
    # Ultimate Tensile Strength
    UTS = data_subset['stress'].max()
    UTS_strain = data_subset.loc[data_subset['stress'].idxmax(), 'strain']
    
    # Fracture properties
    fracture_strain = data_subset['strain'].max()
    fracture_stress = data_subset.iloc[-1]['stress']
    
    # Ductility metrics
    elongation = fracture_strain * 100  # Percent elongation
    
    # Toughness (area under stress-strain curve)
    toughness = np.trapz(data_subset['stress'], data_subset['strain'])
    
    properties_summary.append({
        'Temperature (°C)': temp,
        'Young\'s Modulus (GPa)': E_measured / 1000,
        'Yield Strength (MPa)': yield_stress,
        'UTS (MPa)': UTS,
        'Fracture Strain': fracture_strain,
        'Elongation (%)': elongation,
        'Toughness (MJ/m³)': toughness
    })
    
    print(f"\nTemperature: {temp}°C")
    print(f"  Young's Modulus:    {E_measured/1000:.1f} GPa")
    print(f"  Yield Strength:     {yield_stress:.1f} MPa")
    print(f"  UTS:                {UTS:.1f} MPa")
    print(f"  Fracture Strain:    {fracture_strain:.3f}")
    print(f"  Elongation:         {elongation:.1f}%")
    print(f"  Toughness:          {toughness:.2f} MJ/m³")

print("="*70 + "\n")

# ============================================================================
# CONSTITUTIVE MODEL FITTING (POWER LAW HARDENING)
# ============================================================================

print("="*70)
print("POWER LAW HARDENING MODEL FITTING")
print("="*70)

def power_law_hardening(strain, K, n):
    """Hollomon equation: σ = K * ε^n"""
    return K * strain ** n

for temp in temperatures:
    data_subset = tensile_data[tensile_data['temperature'] == temp].copy()
    
    # Fit only plastic region (after yield, before necking)
    plastic_region = data_subset[
        (data_subset['strain'] > 0.005) & 
        (data_subset['strain'] < data_subset['stress'].idxmax() / len(data_subset))
    ]
    
    if len(plastic_region) > 10:
        try:
            popt, _ = curve_fit(
                power_law_hardening,
                plastic_region['strain'],
                plastic_region['stress'],
                p0=[1000, 0.3],
                maxfev=5000
            )
            K_fit, n_fit = popt
            
            # Calculate R² for fit quality
            stress_pred = power_law_hardening(plastic_region['strain'], K_fit, n_fit)
            r2 = r2_score(plastic_region['stress'], stress_pred)
            
            print(f"\n{temp}°C: σ = {K_fit:.1f} * ε^{n_fit:.3f}  (R² = {r2:.4f})")
        except:
            print(f"\n{temp}°C: Fitting failed")

print("\n" + "="*70 + "\n")

# ============================================================================
# PYSINDY MODEL FITTING
# ============================================================================

print("="*70)
print("PYSINDY CONSTITUTIVE MODEL DISCOVERY")
print("="*70)

# Use room temperature data for SINDy analysis
sample_data = tensile_data[tensile_data['temperature'] == 25].copy()

# Prepare data: strain as input, stress as output
# We'll model dσ/dε as a function of strain and stress
X = sample_data[['strain', 'stress']].values
strain_vals = X[:, 0].reshape(-1, 1)

# Calculate strain increment
d_strain = np.gradient(sample_data['strain'])

# Initialize SINDy with polynomial library
poly_lib = PolynomialLibrary(degree=3, include_bias=True)

model_sindy = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1, alpha=0.01),
    feature_library=poly_lib,
    differentiation_method=ps.FiniteDifference(order=2),
)

# Fit model
print("\nFitting SINDy to stress-strain relationship...")
model_sindy.fit(X, t=d_strain, feature_names=['strain', 'stress'])

print("\nDiscovered equations:")
print("="*70)
model_sindy.print()
print("="*70)

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Plot 1: Multi-temperature stress-strain curves
ax1 = plt.subplot(3, 3, 1)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
for i, temp in enumerate(temperatures):
    data_subset = tensile_data[tensile_data['temperature'] == temp]
    ax1.plot(data_subset['strain'], data_subset['stress'], 
             linewidth=2.5, label=f'{temp}°C', color=colors[i])
ax1.set_xlabel('Engineering Strain', fontsize=12, fontweight='bold')
ax1.set_ylabel('Engineering Stress (MPa)', fontsize=12, fontweight='bold')
ax1.set_title('Stress-Strain Curves', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Young's Modulus vs Temperature
ax2 = plt.subplot(3, 3, 2)
props_df = pd.DataFrame(properties_summary)
ax2.plot(props_df['Temperature (°C)'], props_df['Young\'s Modulus (GPa)'], 
         'o-', linewidth=2.5, markersize=8, color='#e41a1c')
ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Young\'s Modulus (GPa)', fontsize=12, fontweight='bold')
ax2.set_title('Modulus vs Temperature', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Yield Strength vs Temperature
ax3 = plt.subplot(3, 3, 3)
ax3.plot(props_df['Temperature (°C)'], props_df['Yield Strength (MPa)'], 
         'o-', linewidth=2.5, markersize=8, color='#377eb8')
ax3.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Yield Strength (MPa)', fontsize=12, fontweight='bold')
ax3.set_title('Yield Strength vs Temperature', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: UTS vs Temperature
ax4 = plt.subplot(3, 3, 4)
ax4.plot(props_df['Temperature (°C)'], props_df['UTS (MPa)'], 
         'o-', linewidth=2.5, markersize=8, color='#4daf4a')
ax4.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax4.set_ylabel('UTS (MPa)', fontsize=12, fontweight='bold')
ax4.set_title('Ultimate Tensile Strength vs Temperature', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Elongation vs Temperature
ax5 = plt.subplot(3, 3, 5)
ax5.plot(props_df['Temperature (°C)'], props_df['Elongation (%)'], 
         'o-', linewidth=2.5, markersize=8, color='#984ea3')
ax5.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Elongation (%)', fontsize=12, fontweight='bold')
ax5.set_title('Ductility vs Temperature', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Toughness vs Temperature
ax6 = plt.subplot(3, 3, 6)
ax6.plot(props_df['Temperature (°C)'], props_df['Toughness (MJ/m³)'], 
         'o-', linewidth=2.5, markersize=8, color='#ff7f00')
ax6.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Toughness (MJ/m³)', fontsize=12, fontweight='bold')
ax6.set_title('Energy Absorption vs Temperature', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: True stress-strain (room temp)
ax7 = plt.subplot(3, 3, 7)
rt_data = tensile_data[tensile_data['temperature'] == 25]
ax7.plot(rt_data['strain'], rt_data['stress'], 
         linewidth=2.5, label='Engineering', color='#e41a1c')
ax7.plot(rt_data['true_strain'], rt_data['true_stress_corrected'], 
         linewidth=2.5, label='True', linestyle='--', color='#377eb8')
ax7.set_xlabel('Strain', fontsize=12, fontweight='bold')
ax7.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
ax7.set_title('Engineering vs True Stress-Strain (25°C)', fontsize=13, fontweight='bold')
ax7.legend(loc='best', fontsize=10)
ax7.grid(True, alpha=0.3)

# Plot 8: Strain hardening rate
ax8 = plt.subplot(3, 3, 8)
for i, temp in enumerate(temperatures):
    data_subset = tensile_data[tensile_data['temperature'] == temp]
    hardening_rate = np.gradient(data_subset['stress'], data_subset['strain'])
    ax8.plot(data_subset['strain'], hardening_rate, 
             linewidth=2, label=f'{temp}°C', color=colors[i], alpha=0.7)
ax8.set_xlabel('Strain', fontsize=12, fontweight='bold')
ax8.set_ylabel('Hardening Rate (dσ/dε, MPa)', fontsize=12, fontweight='bold')
ax8.set_title('Strain Hardening Behavior', fontsize=13, fontweight='bold')
ax8.set_ylim(0, 250000)
ax8.legend(loc='best', fontsize=10)
ax8.grid(True, alpha=0.3)

# Plot 9: Work hardening exponent (log-log plot)
ax9 = plt.subplot(3, 3, 9)
for i, temp in enumerate(temperatures):
    data_subset = tensile_data[tensile_data['temperature'] == temp]
    # Plastic region only
    plastic = data_subset[(data_subset['strain'] > 0.005) & (data_subset['strain'] < 0.3)]
    if len(plastic) > 10:
        # Filter out zeros and negatives for log plot
        valid = (plastic['strain'] > 0) & (plastic['stress'] > 0)
        ax9.loglog(plastic[valid]['strain'], plastic[valid]['stress'], 
                   linewidth=2, label=f'{temp}°C', color=colors[i], alpha=0.7)
ax9.set_xlabel('Log(Plastic Strain)', fontsize=12, fontweight='bold')
ax9.set_ylabel('Log(Stress, MPa)', fontsize=12, fontweight='bold')
ax9.set_title('Power Law Hardening (Log Scale)', fontsize=13, fontweight='bold')
ax9.legend(loc='best', fontsize=10)
ax9.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# ============================================================================
# EXPORT RESULTS
# ============================================================================

print("Exporting results...")

# Export full dataset
tensile_data.to_csv('alloy617_tensile_data.csv', index=False)
print("Tensile data saved to: alloy617_tensile_data.csv")

# Export properties summary
props_df.to_csv('alloy617_mechanical_properties.csv', index=False)
print("Properties summary saved to: alloy617_mechanical_properties.csv")

print("\nTensile test analysis complete!")
print("="*70)