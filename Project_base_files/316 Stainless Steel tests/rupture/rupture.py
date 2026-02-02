from creep_rupture_sindy import *
import numpy as np


def example_basic_workflow():
    """Example 1: Basic model fitting and prediction."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Workflow")
    print("="*70)

    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    # Fit model
    model = fit_sindy_lmp_model(df, C=20.0, poly_degree=3)
    print_model_summary(model)
    
    # Single prediction
    t_rupture = predict_rupture_time(873.0, 200.0, model)
    print(f"\nPrediction for T=873K, σ=200MPa:")
    print(f"  Rupture time: {t_rupture:.1f} hours ({t_rupture/8760:.2f} years)")


def example_parameter_optimization():
    """Example 2: Optimize C parameter."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Parameter Optimization")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    # Fine grid around typical values
    C_grid = np.linspace(18, 22, 21)
    
    best_model, results = optimize_C_parameter(
        df, 
        C_grid=C_grid,
        poly_degree=3,
        threshold=0.01
    )
    
    print(f"\nOptimal C: {best_model.C:.3f}")
    print(f"Test R²: {best_model.test_r2:.4f}")
    print(f"CV R²: {best_model.cv_mean:.4f} ± {best_model.cv_std:.4f}")
    
    # Plot results
    plot_c_optimization(results)


def example_batch_predictions():
    """Example 3: Generate predictions for multiple conditions."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Predictions")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    # Fit model
    model = fit_sindy_lmp_model(df, C=20.0)
    
    # Define operating conditions
    conditions = [
        (773.0, 300.0),  # Low temp, high stress
        (823.0, 250.0),
        (873.0, 200.0),
        (923.0, 150.0),
        (973.0, 100.0),  # High temp, low stress
    ]
    
    # Generate predictions
    pred_table = create_prediction_table(conditions, model)
    
    print("\nPredictions:")
    print(pred_table.to_string(index=False))
    
    # Save to file
    pred_table.to_csv("example_predictions.csv", index=False)
    print("\nSaved to: example_predictions.csv")


def example_custom_analysis():
    """Example 4: Custom analysis with different polynomial degrees."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Polynomial Degrees")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    degrees = [2, 3, 4, 5]
    results = []
    
    print("\nTesting polynomial degrees:", degrees)
    
    for deg in degrees:
        model = fit_sindy_lmp_model(
            df, 
            C=20.0, 
            poly_degree=deg,
            threshold=0.01
        )
        results.append(model)
        print(f"  Degree {deg}: Test R² = {model.test_r2:.4f}, "
              f"CV R² = {model.cv_mean:.4f} ± {model.cv_std:.4f}")
    
    # Find best
    best_idx = np.argmax([r.cv_mean for r in results])
    best_degree = degrees[best_idx]
    
    print(f"\nBest polynomial degree: {best_degree}")
    print(f"Equation:")
    results[best_idx].model.print()


def example_visualization():
    """Example 5: Create comprehensive visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Generating Visualizations")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    # Fit model
    model = fit_sindy_lmp_model(df, C=20.0, poly_degree=3)
    
    print("\nCreating diagnostic plots...")
    plot_model_performance(df, model, save_path="diagnostics.png")
    print("Saved: diagnostics.png")


def example_sensitivity_analysis():
    """Example 6: Analyze prediction sensitivity to temperature and stress."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Sensitivity Analysis")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    model = fit_sindy_lmp_model(df, C=20.0)
    
    # Base condition
    T_base = 873.0  # K
    stress_base = 200.0  # MPa
    t_base = predict_rupture_time(T_base, stress_base, model)
    
    print(f"\nBase condition: T={T_base}K, σ={stress_base}MPa")
    print(f"Rupture time: {t_base:.1f} hours\n")
    
    # Temperature sensitivity
    print("Temperature sensitivity (+/- 50K):")
    for dT in [-50, -25, 0, 25, 50]:
        T = T_base + dT
        t = predict_rupture_time(T, stress_base, model)
        change = 100 * (t - t_base) / t_base
        print(f"  T={T:.0f}K: {t:.1f}h ({change:+.1f}%)")
    
    # Stress sensitivity
    print("\nStress sensitivity (+/- 50MPa):")
    for dS in [-50, -25, 0, 25, 50]:
        stress = stress_base + dS
        t = predict_rupture_time(T_base, stress, model)
        change = 100 * (t - t_base) / t_base
        print(f"  σ={stress:.0f}MPa: {t:.1f}h ({change:+.1f}%)")


def example_design_curve():
    """Example 7: Generate design curve for specific temperature."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Design Curve Generation")
    print("="*70)
    
    df = pd.read_excel("SS316H-rupture.xlsx", sheet_name="Rupture")
    df = standardize_columns(df)
    
    model = fit_sindy_lmp_model(df, C=20.0)
    
    # Generate curve for T=873K (600°C)
    T = 873.0
    stresses = np.linspace(100, 300, 20)
    times = [predict_rupture_time(T, s, model) for s in stresses]
    
    curve_df = pd.DataFrame({
        'Stress [MPa]': stresses,
        'Rupture Time [h]': times,
        'Rupture Time [years]': [t/8760 for t in times]
    })
    
    print(f"\nDesign curve for T={T}K ({T-273.15:.0f}°C):")
    print(curve_df.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(stresses, times, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Stress [MPa]', fontsize=12)
    plt.ylabel('Rupture Time [hours]', fontsize=12)
    plt.title(f'Design Curve at T = {T}K ({T-273.15:.0f}°C)', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('design_curve.png', dpi=300)
    print("\nSaved: design_curve.png")
    plt.show()


if __name__ == "__main__":
    """
    Run specific examples by uncommenting the desired function calls.
    """
    
    # Or run all examples:
    print("\n" + "#"*70)
    print("# CREEP RUPTURE PREDICTION - EXAMPLE USAGE")
    print("#"*70)
    
    try:
        example_basic_workflow()
        example_batch_predictions()
        example_custom_analysis()
        example_sensitivity_analysis()
    
        
    except FileNotFoundError:
        print("\nError: SS316H-rupture.xlsx not found.")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#"*70)
    print("# EXAMPLES COMPLETE")
    print("#"*70)
