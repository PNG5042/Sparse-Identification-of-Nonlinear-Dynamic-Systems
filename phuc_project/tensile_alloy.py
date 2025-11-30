# -------------------------------------------------------------------------
# numpy (np) → numerical arrays and calculations.
# pandas (pd) → structured tabular data (DataFrame).
# matplotlib.pyplot (plt) → for plotting (optional here).
# pysindy → sparse identification of dynamics library.
# PolynomialLibrary → constructs polynomial features for SINDy.
# curve_fit → fits nonlinear functions (here used for Hollomon power-law).
# r2_score → evaluates goodness of fit.
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# -------------------------------------------------------------------------
# Synthetic Tensile Test Script for Alloy 617 (prints formatted summary +
# Hollomon fits + PySINDy discovery). Uses only the requested libraries.
# -------------------------------------------------------------------------

# -------------------------
# Constitutive model class
# -------------------------
class TensileTestModel:
    def __init__(self, temp_c=25.0, strain_rate=1e-3):
        self.temp_c = float(temp_c)
        self.strain_rate = float(strain_rate)
        # Representative temperature-dependent parameters (MPa)
        if temp_c <= 100:
            self.E = 211000.0
            self.yield_stress = 380.0
            self.K = 1200.0
            self.n = 0.35
            self.UTS = 750.0
        elif temp_c <= 500:
            self.E = 195000.0
            self.yield_stress = 320.0
            self.K = 1000.0
            self.n = 0.30
            self.UTS = 650.0
        elif temp_c <= 700:
            self.E = 180000.0
            self.yield_stress = 280.0
            self.K = 850.0
            self.n = 0.25
            self.UTS = 550.0
        else:
            self.E = 170000.0
            self.yield_stress = 200.0
            self.K = 650.0
            self.n = 0.20
            self.UTS = 400.0

        self.yield_strain = self.yield_stress / self.E
        self.necking_strain = self.n   # approximate necking point
        self.fracture_strain = self.necking_strain + 0.15

    def stress_strain(self, strain):
        strain = np.asarray(strain)
        stress = np.zeros_like(strain, dtype=float)
        for i, eps in enumerate(strain):
            if eps <= self.yield_strain:
                stress[i] = self.E * eps
            elif eps <= self.necking_strain:
                eps_pl = eps - self.yield_strain
                stress[i] = self.K * (eps_pl + self.yield_strain) ** self.n
            elif eps <= self.fracture_strain:
                failure_stress = 0.7 * self.UTS
                progress = (eps - self.necking_strain) / (self.fracture_strain - self.necking_strain)
                stress[i] = self.UTS - (self.UTS - failure_stress) * progress
            else:
                stress[i] = 0.0
        return stress

    def add_noise(self, stress, noise_level=0.02):
        sigma = np.std(stress)
        noise = np.random.normal(0.0, noise_level * sigma, size=stress.shape)
        return stress + noise



# -------------------------
# Data generation
# -------------------------
def generate_tensile_dataset(temperatures_c, strain_rate=1e-3, n_points=1000):
    rows = []
    for temp in temperatures_c:
        model = TensileTestModel(temp_c=temp, strain_rate=strain_rate)
        strain = np.linspace(0.0, model.fracture_strain, n_points)
        true_stress = model.stress_strain(strain)
        stress_noisy = model.add_noise(true_stress, noise_level=0.01)
        df = pd.DataFrame({
            "temperature_C": temp,
            "strain": strain,
            "stress": stress_noisy,
            "true_stress": true_stress,
            "strain_rate": strain_rate
        })
        df["true_strain"] = np.log(1 + df["strain"])
        df["true_stress_corrected"] = df["stress"] * (1 + df["strain"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# -------------------------
# Property extraction
# -------------------------
def extract_properties(df):
    out = {}
    # choose small-strain window for slope estimate (robust to noise)
    small_eps = df[df["strain"] <= 0.005]
    E_measured = np.polyfit(small_eps["strain"], small_eps["stress"], 1)[0]

    offset = 0.002
    offset_line = E_measured * (df["strain"] - offset)
    idx = np.where(df["stress"].values > offset_line.values)[0]
    if idx.size > 0:
        yield_idx = idx[0]
        yield_stress = float(df.iloc[yield_idx]["stress"])
        yield_strain = float(df.iloc[yield_idx]["strain"])
    else:
        yield_stress = np.nan
        yield_strain = np.nan

    UTS = float(df["stress"].max())
    UTS_strain = float(df.loc[df["stress"].idxmax(), "strain"])

    fracture_strain = float(df["strain"].max())
    fracture_stress = float(df.iloc[-1]["stress"])

    elongation_pct = fracture_strain * 100.0

    # toughness (area under engineering stress-strain curve) - use np.trapz to match deprecation note
    toughness = float(np.trapz(df["stress"].values, df["strain"].values))

    out.update({
        "E_measured_MPa": E_measured,
        "yield_stress_MPa": yield_stress,
        "UTS_MPa": UTS,
        "fracture_strain": fracture_strain,
        "elongation_pct": elongation_pct,
        "toughness_MPa_strain": toughness
    })

    return out


# -------------------------
# Hollomon fit (power-law)
# -------------------------
def power_law_hardening(eps, K, n):
    return K * (eps ** n)


def fit_power_law(df, strain_min=0.005, upper_quantile=0.8):
    upper = df["strain"].quantile(upper_quantile)
    plastic = df[(df["strain"] > strain_min) & (df["strain"] < upper)]
    if len(plastic) < 10:
        return None, None, None
    eps = plastic["strain"].values
    sig = plastic["stress"].values
    eps_safe = np.maximum(eps, 1e-8)
    try:
        popt, _ = curve_fit(power_law_hardening, eps_safe, sig, p0=[1000, 0.3], maxfev=5000)
        K_fit, n_fit = popt
        sig_pred = power_law_hardening(eps_safe, K_fit, n_fit)
        r2 = r2_score(sig, sig_pred)
        return float(K_fit), float(n_fit), float(r2)
    except Exception:
        return None, None, None


# -------------------------
# Small PySINDy demo
# -------------------------
def small_sindy_demo(df, degree=3, threshold=0.5):
    # Model d(stress)/d(strain) = f(stress). Use strain as pseudo-time.
    X = df["stress"].values.reshape(-1, 1)
    t = df["strain"].values
    mask = np.concatenate([[True], np.diff(t) > 0])
    X = X[mask]
    t = t[mask]
    lib = PolynomialLibrary(degree=degree, include_bias=True)
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=0.01),
        feature_library=lib,
        differentiation_method=ps.FiniteDifference(order=2),
    )
    model.fit(X, t=t, feature_names=["stress"])
    return model


# -------------------------
# Main flow: generate, analyze, print formatted blocks
# -------------------------
def format_block_header():
    print("=" * 70)
    print("EXTRACTED MECHANICAL PROPERTIES")
    print("=" * 70)


def format_properties_block(props, temperature):
    # Print formatted block similar to the example
    E_gpa = props["E_measured_MPa"] / 1000.0
    yield_mpa = props["yield_stress_MPa"]
    uts_mpa = props["UTS_MPa"]
    frac = props["fracture_strain"]
    elong = props["elongation_pct"]
    tough = props["toughness_MPa_strain"]

    print(f"\nTemperature: {int(temperature)}°C")
    print(f"  Young's Modulus:    {E_gpa:.1f} GPa")
    print(f"  Yield Strength:     {yield_mpa:.1f} MPa")
    print(f"  UTS:                {uts_mpa:.1f} MPa")
    print(f"  Fracture Strain:    {frac:.3f}")
    print(f"  Elongation:         {elong:.1f}%")
    print(f"  Toughness:          {tough:.2f} MJ/m³")


def main():
    temps = [25, 400, 650, 850]
    print("Generating synthetic tensile test data...\n")
    data = generate_tensile_dataset(temps, strain_rate=1e-3)

    format_block_header()

    summaries = []
    for temp in temps:
        df_temp = data[data["temperature_C"] == temp].reset_index(drop=True)
        props = extract_properties(df_temp)
        # Append powerlaw results
        K, n, r2 = fit_power_law(df_temp)
        props.update({"powerlaw_K": K, "powerlaw_n": n, "powerlaw_r2": r2, "temperature_C": temp})
        summaries.append(props)

        # Print formatted block (note: trapz deprecation will be shown when running)
        format_properties_block(props, temperature=temp)

    print("=" * 70 + "\n")

    # -------------------------
    # POWER LAW HARDENING FIT REPORT
    # -------------------------
    print("=" * 70)
    print("POWER LAW HARDENING MODEL FITTING")
    print("=" * 70)
    for s in summaries:
        t = int(s["temperature_C"])
        if s["powerlaw_K"] is not None:
            print(f"\n{t}°C: σ = {s['powerlaw_K']:.1f} * ε^{s['powerlaw_n']:.3f}  (R² = {s['powerlaw_r2']:.4f})")
        else:
            print(f"\n{t}°C: Fit not available")
    print("\n" + "=" * 70 + "\n")

    # -------------------------
    # PYSINDY DISCOVERY (room temp and plastic region)
    # -------------------------
    print("=" * 70)
    print("PYSINDY CONSTITUTIVE MODEL DISCOVERY")
    print("=" * 70)
    # Room temperature SINDy
    df25 = data[data["temperature_C"] == 25].reset_index(drop=True)
    print("\nFitting SINDy to stress-strain relationship...")
    print("Modeling: dσ/dε = f(σ)\n")
    try:
        model = small_sindy_demo(df25, degree=3, threshold=0.5)
        print("Discovered equation for dσ/dε:")
        print("=" * 70)
        model.print()
        print("=" * 70)
        # print coefficients in a compact form
        coeffs = model.coefficients().ravel()
        names = model.get_feature_names()
        print("\nCoefficients:")
        for name, c in zip(names, coeffs):
            if abs(c) > 0.01:
                print(f"  {name}: {c:.4f}")
    except Exception as e:
        print("SINDy failed (room temp):", e)

    print("\n" + "=" * 70 + "\n")

    # Alternative: SINDy on plastic region only
    print("=" * 70)
    print("ALTERNATIVE: PYSINDY ON PLASTIC REGION ONLY")
    print("=" * 70)
    plastic_region = df25[(df25["strain"] > 0.01) & (df25["strain"] < df25["strain"].quantile(0.8))].reset_index(drop=True)
    try:
        if len(plastic_region) > 50:
            model_pl = small_sindy_demo(plastic_region, degree=2, threshold=0.1)
            print("\nFitting SINDy to plastic deformation region...")
            print("Modeling: dσ/dε = f(σ) for plastic strain\n")
            print("Discovered equation for plastic region:")
            print("=" * 70)
            model_pl.print()
            print("=" * 70)
            # evaluate R2 by simulation (simulate stress evolution)
            X0 = np.array([plastic_region["stress"].values[0]])
            t_span = plastic_region["strain"].values
            X_pred = model_pl.simulate(X0, t_span)
            r2_pl = r2_score(plastic_region["stress"].values.reshape(-1, 1), X_pred.reshape(-1, 1))
            print(f"\nModel R² score: {r2_pl:.4f}")
        else:
            print("\nNot enough plastic-region data for SINDy fit.")
    except Exception as e:
        print("SINDy (plastic) failed:", e)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
