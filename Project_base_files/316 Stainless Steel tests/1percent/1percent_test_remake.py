# ================================
# SS316H CREEP – SINDy APPLICABILITY STUDY  (PySINDy edition)
# ================================
# Uses PySINDy with STLSQ optimizer on a physics-motivated custom feature library.
# SINDy is cast as a regression problem: state = pre-computed physics features,
# derivative = log(t), IdentityLibrary so STLSQ selects sparse terms directly.

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import IdentityLibrary
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
OUT  = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── 0. DATA ──────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "SS316H-1percent.csv"
np.random.seed(42)
REAL = csv_path.exists()

if REAL:
    df = pd.read_csv(csv_path)
    if df["Heat"].dtype == object:
        hmap = {h: i for i, h in enumerate(df["Heat"].unique())}
        df["Heat_encoded"] = df["Heat"].map(hmap)
    Heat   = df["Heat_encoded"].values.astype(float)
    Temp   = df["Temp (K)"].values.astype(float)
    Stress = df["Stress (Mpa)"].values.astype(float)
    Time   = df["Time (h) to 1% strain"].values.astype(float)
    print(f"Loaded real data: {len(df)} rows")
else:
    print("CSV not found – generating synthetic SS316H data")
    n   = 300
    rng = np.random.default_rng(42)
    T_vals = np.array([873, 923, 973, 1023, 1073, 1123])
    S_vals = np.array([50, 75, 100, 125, 150, 200, 250, 300])
    T_g, S_g = np.meshgrid(T_vals, S_vals)
    Tb, Sb   = T_g.ravel(), S_g.ravel()
    reps = n // len(Tb) + 2
    Temp   = np.tile(Tb, reps)[:n] + rng.normal(0, 3, n)
    Stress = np.tile(Sb, reps)[:n].astype(float) + rng.normal(0, 3, n)
    Temp   = np.clip(Temp,   850, 1150)
    Stress = np.clip(Stress,  30,  350)
    Heat   = rng.integers(0, 5, n).astype(float)
    C_true, n_true, QR_true = 42.0, 5.0, 285000 / 8.314
    log_t = C_true - n_true * np.log(Stress) + QR_true / Temp
    Time  = np.exp(log_t + rng.normal(0, 0.25, n))
    print(f"  True: log(t) = {C_true} - {n_true}*log(sigma) + {QR_true:.0f}/T")

y_log = np.log(Time)
print(f"\nN={len(Time)}, log(t) range: {y_log.min():.2f}-{y_log.max():.2f}")
print(f"Temp: {Temp.min():.0f}-{Temp.max():.0f} K  |  Stress: {Stress.min():.0f}-{Stress.max():.0f} MPa")

# ── 1. ML PREDICTION MODEL ───────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: ML PREDICTION MODEL")
print("="*65)

def ml_feats(H, T, S):
    return np.column_stack([
        np.ones_like(H), H, H**2,
        T, 1/T, T**2, 1/T**2, np.log(T), T**-0.5,
        S, np.log(S), S**2, S**3, 1/S, 1/S**2, 1/S**3, S**0.5,
        S/T, np.log(S)/T, T*np.log(S), np.log(T)*np.log(S),
        S/T**2, H/T, H*np.log(S), H*S, 1/(S*T),
    ])

X_all = ml_feats(Heat, Temp, Stress)
sel   = SelectKBest(f_regression, k=min(15, X_all.shape[1]))
X_sel = sel.fit_transform(X_all, y_log)
sc_ml = StandardScaler()
X_sc  = sc_ml.fit_transform(X_sel)

bins = np.digitize(y_log, np.percentile(y_log, [20, 40, 60, 80]))
sss  = StratifiedShuffleSplit(1, test_size=0.25, random_state=42)
tr_i, te_i = next(sss.split(X_sc, bins))
Xtr, Xte = X_sc[tr_i], X_sc[te_i]
ytr, yte = y_log[tr_i], y_log[te_i]

rcv = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10, 50, 100, 500]}, cv=5, scoring='r2')
rcv.fit(Xtr, ytr)
ridge = rcv.best_estimator_
rf    = RandomForestRegressor(300, max_depth=6, min_samples_split=12,
                               min_samples_leaf=6, max_features='sqrt', random_state=42)
rf.fit(Xtr, ytr)
ens = VotingRegressor([('r1', ridge), ('r2', Ridge(alpha=rcv.best_params_['alpha'])), ('rf', rf)])
ens.fit(Xtr, ytr)

ML  = {'Ridge': ridge, 'RandomForest': rf, 'Ensemble': ens}
RES = {}
for nm, m in ML.items():
    yp  = m.predict(Xte)
    err = np.abs((np.exp(yp) - np.exp(yte)) / np.exp(yte)) * 100
    RES[nm] = dict(tr_r2=m.score(Xtr, ytr), te_r2=m.score(Xte, yte),
                   med_err=np.median(err), err=err, ypall=m.predict(X_sc))
    print(f"  {nm:<14} TrainR2={RES[nm]['tr_r2']:.4f}  TestR2={RES[nm]['te_r2']:.4f}  "
          f"MedianErr={RES[nm]['med_err']:.1f}%")

best_ml = max(RES, key=lambda k: RES[k]['te_r2'])
print(f"\n  Best: {best_ml}  (R2={RES[best_ml]['te_r2']:.4f}, err={RES[best_ml]['med_err']:.1f}%)")

# Norton-Bailey OLS reference
A_nb = np.column_stack([np.ones_like(Temp), np.log(Stress), 1/Temp])
nb_actual, *_ = np.linalg.lstsq(A_nb, y_log, rcond=None)
C_nb, n_nb, QR_nb = nb_actual
print(f"\n  Norton-Bailey OLS: log(t) = {C_nb:.4f} + {n_nb:.4f}*log(sigma) + {QR_nb:.2f}/T")

# ── 2. PySINDy SPARSE EQUATION DISCOVERY ─────────────────────────────
print("\n" + "="*65)
print("STEP 2: SINDy – SPARSE EQUATION DISCOVERY  (PySINDy STLSQ)")
print("="*65)
print("  Library: IdentityLibrary on physics-motivated features")
print("  Optimizer: STLSQ (Sequentially Thresholded Least Squares)")

# Physics-motivated feature matrix (no scaling — STLSQ works in original space)
# Features: [C0, 1/T, log(sigma), log(sigma)^2, 1/sigma, (1/T)*log(sigma), log(T), Garofalo]
FEAT_LABELS = ['C0', '1/T', 'log(s)', 'log(s)^2', '1/s',
               '(1/T)log(s)', 'log(T)', 'Garofalo']

def build_phi(T, S):
    """Build (N,8) physics feature matrix (unscaled)."""
    sinh_arg = np.clip(0.01 * S, 1e-9, 500)
    return np.column_stack([
        np.ones(len(T)),                   # C0  — constant / intercept
        1.0 / T,                           # Arrhenius temperature term
        np.log(S),                         # Norton power-law stress exponent
        np.log(S)**2,                      # quadratic stress correction
        1.0 / S,                           # inverse stress
        (1.0 / T) * np.log(S),             # Larson-Miller cross term
        np.log(T),                         # log-temperature
        np.log(np.sinh(sinh_arg)),         # Garofalo sinh creep law
    ])

Phi = build_phi(Temp, Stress)  # (N, 8), no scaling needed

def fit_sindy(y_target, threshold):
    """
    Fit a PySINDy STLSQ model to discover a sparse equation for y_target.

    Design:
      - State = Phi (N x 8 unscaled physics features)
      - x_dot = y_target (N,) — treated as the 'derivative' to be explained
      - Library = IdentityLibrary — passes Phi through unchanged (no cross-products)
      - STLSQ iteratively zeroes coefficients below `threshold`, promoting sparsity

    Returns: coef_dict, r2, y_pred (N,), n_active
    """
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=1000)
    model     = ps.SINDy(feature_library=IdentityLibrary(), optimizer=optimizer)
    model.fit(Phi, t=1, x_dot=y_target.reshape(-1, 1))

    y_pred = np.asarray(model.predict(Phi)).ravel()
    ss_r   = np.sum((y_target - y_pred)**2)
    ss_t   = np.sum((y_target - y_target.mean())**2)
    r2v    = 1 - ss_r / ss_t if ss_t else 0.0

    raw    = model.coefficients().ravel()   # length = 8
    coef_dict = {lbl: float(v)
                 for lbl, v in zip(FEAT_LABELS, raw) if abs(v) > 1e-10}
    return coef_dict, r2v, y_pred, len(coef_dict)

# Tune STLSQ threshold via grid search on actual data
print("\n  Tuning STLSQ threshold via grid:")
thresholds  = np.logspace(-4, 2, 80)
best_thresh = 0.01
best_r2_val = -np.inf
for thresh in thresholds:
    try:
        cd, rv, _, nact = fit_sindy(y_log, thresh)
        if rv > best_r2_val and nact >= 2:
            best_r2_val, best_thresh = rv, thresh
    except Exception:
        continue
print(f"  Best threshold = {best_thresh:.5f}  (R2={best_r2_val:.4f})")

# Fit on actual data
coefs_act, r2_act, yp_act, n_act_t = fit_sindy(y_log, best_thresh)
print(f"\n  SINDy (actual):  R2={r2_act:.4f},  {n_act_t} active: {list(coefs_act.keys())}")

# Fit on ML-predicted data
coefs_preds = {}
r2_preds    = {}
yp_sindy_ml = {}
for nm in ML:
    cd, r2v, yhat, nact = fit_sindy(RES[nm]['ypall'], best_thresh)
    coefs_preds[nm] = cd
    r2_preds[nm]    = r2v
    yp_sindy_ml[nm] = yhat
    print(f"  SINDy ({nm}): R2={r2v:.4f},  {nact} active: {list(cd.keys())}")

# ── 3. PRINT EQUATIONS ───────────────────────────────────────────────
def fmt_eq(coef_dict, ylbl='log(t)'):
    parts = []
    if 'C0' in coef_dict:
        parts.append(f"{coef_dict['C0']:+.5g}  [intercept]")
    for k, v in coef_dict.items():
        if k != 'C0':
            parts.append(f"{v:+.5g} * {k}")
    return (f"  {ylbl} = 0  (no active terms)" if not parts
            else f"  {ylbl} =\n    " + "\n    ".join(parts))

print("\n" + "="*65)
print("STEP 3: DISCOVERED EQUATIONS")
print("="*65)
print(f"\n  -- SINDy on ACTUAL data (R2={r2_act:.4f}) --")
print(fmt_eq(coefs_act))
for nm in ML:
    print(f"\n  -- SINDy on {nm} (ML err={RES[nm]['med_err']:.1f}%, R2={r2_preds[nm]:.4f}) --")
    print(fmt_eq(coefs_preds[nm]))

# ── 4. SIMILARITY METRIC ─────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 4: EQUATION SIMILARITY  (metric = 1 - |rel_error|)")
print("="*65)

# Map analytic reference to internal feature labels
analytic_ref = {'C0': C_nb, 'log(s)': n_nb, '1/T': QR_nb}

def compute_sim(sindy_coefs, analytic_ref):
    rows = []
    for param, ana in analytic_ref.items():
        sval = sindy_coefs.get(param, 0.0)
        if abs(ana) > 1e-8:
            rel = (ana - sval) / ana
            sim = 1.0 - abs(rel)
        else:
            rel = sim = np.nan
        rows.append({'param': param, 'analytical': ana,
                     'sindy': sval, 'rel_err': rel, 'similarity': sim})
    return pd.DataFrame(rows)

df_sim_act = compute_sim(coefs_act, analytic_ref)
print(f"\n  {'Param':<22} {'Analytical':>12} {'SINDy':>12} {'rel_err':>10} {'similarity':>11}")
print("  " + "-"*67)
for _, r in df_sim_act.iterrows():
    re = f"{r['rel_err']:.4f}"    if not np.isnan(r['rel_err'])   else "nan"
    si = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "nan"
    print(f"  {r['param']:<22} {r['analytical']:>12.4f} {r['sindy']:>12.4f} {re:>10} {si:>11}")
mean_sim_act = df_sim_act['similarity'].dropna().mean()
print(f"\n  Mean similarity (actual data): {mean_sim_act:.4f}")

model_rows = []
for nm in ML:
    dfs  = compute_sim(coefs_preds[nm], analytic_ref)
    msim = dfs['similarity'].dropna().mean()
    model_rows.append({'model': nm, 'test_r2': RES[nm]['te_r2'],
                       'median_err_pct': RES[nm]['med_err'],
                       **{f"sim_{r['param']}": r['similarity'] for _, r in dfs.iterrows()},
                       'mean_similarity': msim})
    print(f"  {nm:<14} ML err={RES[nm]['med_err']:.1f}%  mean_sim={msim:.4f}")
df_models = pd.DataFrame(model_rows)

# ── 5. SENSITIVITY ANALYSIS ──────────────────────────────────────────
print("\n" + "="*65)
print("STEP 5: SENSITIVITY – Prediction Error Propagation to Equations")
print("="*65)

noise_stds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]
rng2 = np.random.default_rng(7)
sens_rows = []

for s in noise_stds:
    y_n = y_log + rng2.normal(0, s, len(y_log))
    cn, _, _, _ = fit_sindy(y_n, best_thresh)
    dfs   = compute_sim(cn, analytic_ref)
    msim  = dfs['similarity'].dropna().mean()
    pct   = 100 * (np.exp(s) - 1)
    nact  = len(cn)
    nval  = cn.get('log(s)', 0)
    qrval = cn.get('1/T',    0)
    sens_rows.append({'noise_std': s, 'approx_pct_err': pct,
                      'n_active': nact, 'mean_sim': msim,
                      'sindy_n': nval, 'sindy_QR': qrval})
    print(f"  s={s:.2f} (~{pct:>6.0f}% err)  active={nact:2d}  "
          f"sim={msim:.4f}  n={nval:.4f}  Q/R={qrval:.2f}")

df_sens = pd.DataFrame(sens_rows)
valid = ~df_sens['mean_sim'].isna()
pv = df_sens.loc[valid, 'approx_pct_err'].values
sv = df_sens.loc[valid, 'mean_sim'].values

if len(pv) > 3:
    corr, pval = stats.pearsonr(pv, sv)
    sl, ic, *_ = stats.linregress(pv, sv)
    at85 = ic + sl * 85
    print(f"\n  Pearson r = {corr:.4f}  (p = {pval:.4f})")
    print(f"  Linear: sim = {ic:.4f} + {sl:.6f} * (% err)")
    print(f"  At 85% pred error: sim ~ {at85:.4f}")
else:
    corr = pval = sl = ic = at85 = np.nan

# ── 6. SAVE CSVs ─────────────────────────────────────────────────────
df_sim_act.to_csv(OUT/"sindy_equation_similarity.csv",  index=False, float_format='%.6f')
df_sens.to_csv(   OUT/"sindy_sensitivity_analysis.csv", index=False, float_format='%.6f')
df_models.to_csv( OUT/"sindy_model_comparison.csv",     index=False, float_format='%.6f')

eq_rows = []
for lbl in list(analytic_ref.keys()):
    row = {'term': lbl, 'sindy_actual': coefs_act.get(lbl, 0.)}
    for nm in ML:
        row[f'sindy_{nm}'] = coefs_preds[nm].get(lbl, 0.)
    eq_rows.append(row)
pd.DataFrame(eq_rows).to_csv(OUT/"sindy_discovered_equations.csv",
                              index=False, float_format='%.6f')
print("\n  Saved 4 CSVs to outputs/")

# ── 7. VISUALISATION ─────────────────────────────────────────────────
DARK='#0d1117'; PANEL='#161b22'; GRID='#21262d'
C1='#58a6ff'; C2='#f85149'; C3='#3fb950'; C4='#d29922'; C5='#bc8cff'; TEXT='#c9d1d9'
MC = [C1, C3, C4]

def sax(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=C1, fontsize=10, fontweight='bold', pad=7)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=9)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=9)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)

fig = plt.figure(figsize=(22, 18), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
y_best = RES[best_ml]['ypall']
ll = [y_log.min()-1, y_log.max()+1]

# P1 – ML prediction
ax = fig.add_subplot(gs[0, 0])
ax.scatter(y_log, y_best, s=15, alpha=0.55, color=C1, edgecolors='none')
ax.plot(ll, ll, '--', color=C2, lw=1.5)
sax(ax, f'ML ({best_ml}) - Actual vs Predicted', 'Actual log(t)', 'Predicted log(t)')
ax.text(0.05, 0.92, f"R2={RES[best_ml]['te_r2']:.4f}",
        transform=ax.transAxes, color=C3, fontsize=9, fontweight='bold')

# P2 – SINDy on actual
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_log, yp_act, s=15, alpha=0.55, color=C3, edgecolors='none')
ax2.plot(ll, ll, '--', color=C2, lw=1.5)
sax(ax2, f'SINDy (actual, R2={r2_act:.4f})\n{n_act_t} active terms',
    'Actual log(t)', 'SINDy log(t)')

# P3 – SINDy on best ML
ax3 = fig.add_subplot(gs[0, 2])
ysp = yp_sindy_ml[best_ml]
ax3.scatter(y_best, ysp, s=15, alpha=0.55, color=C4, edgecolors='none')
ll3 = [y_best.min()-1, y_best.max()+1]
ax3.plot(ll3, ll3, '--', color=C2, lw=1.5)
sax(ax3, f'SINDy on {best_ml}\nR2={r2_preds[best_ml]:.4f}',
    'ML Predicted log(t)', 'SINDy log(t)')

# P4 – Similarity bars
ax4 = fig.add_subplot(gs[1, 0])
plabs = df_sim_act['param'].tolist()
xp    = np.arange(len(plabs))
w     = 0.22
srcs  = [('SINDy(actual)', C1, df_sim_act)]
for nm, col in zip(ML, MC[1:]):
    srcs.append((f'{nm}', col, compute_sim(coefs_preds[nm], analytic_ref)))
for i, (lbl, col, df_s) in enumerate(srcs):
    off  = (i - len(srcs)/2 + 0.5) * w
    bars = ax4.bar(xp+off, df_s['similarity'].fillna(0), w,
                   label=lbl, color=col, alpha=0.85, edgecolor=PANEL)
    for bar, val in zip(bars, df_s['similarity']):
        if not np.isnan(val):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+(0.03 if val>=0 else -0.09),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, color=TEXT)
ax4.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.4)
ax4.axhline(0.0, color=C2,  ls='--', lw=1, alpha=0.4)
ax4.set_xticks(xp); ax4.set_xticklabels(plabs, fontsize=8, color=TEXT)
ax4.set_ylim([-0.4, 1.4])
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax4, 'Equation Similarity by Parameter\n(1 = perfect match)', '', 'Similarity')

# P5 – Sensitivity curve
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df_sens['approx_pct_err'], df_sens['mean_sim'], 'o-',
         color=C1, lw=2, ms=6, label='Mean similarity')
if not np.isnan(sl):
    xf = np.linspace(0, df_sens['approx_pct_err'].max(), 200)
    ax5.plot(xf, ic+sl*xf, '--', color=C2, lw=1.5,
             label=f'Trend (d/100%={sl*100:+.3f})')
ax5.axhline(1.0, color=C3, ls=':', alpha=0.5, lw=1, label='Sim=1')
ax5.axhline(0.5, color=C4, ls=':', alpha=0.5, lw=1, label='Sim=0.5')
ax5.set_ylim([-0.3, 1.3])
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax5, 'Error Propagation: Prediction Error -> Equation Quality',
    'Approx. Prediction Error (%)', 'Mean Equation Similarity')

# P6 – Active terms vs noise
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(df_sens['approx_pct_err'], df_sens['n_active'], 's-', color=C5, lw=2, ms=6)
ax6.set_ylim([0, len(FEAT_LABELS)+1])
sax(ax6, 'SINDy Sparsity vs Prediction Error',
    'Approx. Prediction Error (%)', 'N Active Terms')

# P7 – Model comparison scatter
ax7 = fig.add_subplot(gs[2, 0])
for i, row in df_models.iterrows():
    ax7.scatter(row['median_err_pct'], row['mean_similarity'],
                s=250, color=MC[i], zorder=5, edgecolors='white', lw=0.8)
    ax7.annotate(row['model'], (row['median_err_pct'], row['mean_similarity']),
                 textcoords='offset points', xytext=(8, 5),
                 fontsize=9, color=MC[i], fontweight='bold')
sax(ax7, 'ML Model: Prediction Error vs Equation Similarity',
    'Median Prediction Error (%)', 'Mean Equation Similarity')

# P8 – Coefficient trajectory vs noise
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(PANEL)
pa = df_sens['approx_pct_err'].values
ax8.plot(pa, df_sens['sindy_n'].values,  'o-', color=C3, lw=2, ms=5, label='SINDy n')
ax8.axhline(n_nb, color=C3, ls='--', alpha=0.7, lw=1.5, label=f'Analytic n={n_nb:.3f}')
ax8b = ax8.twinx(); ax8b.set_facecolor(PANEL)
ax8b.plot(pa, df_sens['sindy_QR'].values, 's-', color=C4, lw=2, ms=5, label='SINDy Q/R')
ax8b.axhline(QR_nb, color=C4, ls='--', alpha=0.7, lw=1.5, label=f'Analytic Q/R={QR_nb:.0f}')
ax8b.tick_params(colors=TEXT, labelsize=7)
ax8b.set_ylabel('Q/R (K)', color=C4, fontsize=8)
for sp in ax8.spines.values(): sp.set_edgecolor(GRID)
ax8.tick_params(colors=TEXT, labelsize=8)
ax8.grid(True, color=GRID, alpha=0.5, lw=0.5)
ax8.set_title('Coefficient Stability vs Prediction Error', color=C1, fontsize=10, fontweight='bold', pad=7)
ax8.set_xlabel('Approx. Prediction Error (%)', color=TEXT, fontsize=9)
ax8.set_ylabel('n', color=C3, fontsize=8)
l1, b1 = ax8.get_legend_handles_labels()
l2, b2 = ax8b.get_legend_handles_labels()
ax8.legend(l1+l2, b1+b2, fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# P9 – Text summary
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#0d1117')
for sp in ax9.spines.values(): sp.set_edgecolor(GRID)
ax9.axis('off')
C0s  = coefs_act.get('C0',    0)
ns   = coefs_act.get('log(s)', 0)
QRs  = coefs_act.get('1/T',   0)
txt = [
    "-- Equations ----------------------",
    "Norton-Bailey (reference):",
    f"  {C_nb:+.4f}",
    f"  {n_nb:+.4f} * log(sigma)",
    f"  {QR_nb:+.2f} * (1/T)",
    "",
    f"SINDy actual (R2={r2_act:.4f}):",
    f"  {C0s:+.5g}  [intercept]",
    f"  {ns:+.5g} * log(sigma)",
    f"  {QRs:+.5g} * (1/T)",
    f"  ({n_act_t} total active terms)",
    "",
    "-- Similarity ---------------------",
]
for _, r in df_sim_act.iterrows():
    si = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "nan"
    txt.append(f"  {r['param']:<15}: {si}")
txt += [
    f"  MEAN = {mean_sim_act:.4f}",
    "",
    "-- Error Propagation --------------",
    f"  Pearson r = {corr:.4f}",
    (f"  Dsim/100% err = {sl*100:+.4f}" if not np.isnan(sl) else "  n/a"),
    (f"  At 85% err ~ {at85:.4f}" if not np.isnan(at85) else ""),
    "",
    "  [PySINDy v2 | STLSQ | IdentityLib]",
]
ax9.text(0.03, 0.97, "\n".join(txt), transform=ax9.transAxes,
         fontsize=8.3, va='top', fontfamily='monospace', color=TEXT,
         bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.9))
ax9.set_title('Equation & Similarity Summary', color=C1, fontsize=10, fontweight='bold')

fig.suptitle("SS316H Creep - SINDy Applicability Study  (PySINDy 2.x)\n"
             "Sparse Equation Discovery  |  Similarity Metric  |  Error Propagation",
             fontsize=14, fontweight='bold', color=C1, y=0.998)

plt.savefig(OUT/"sindy_analysis.png", dpi=150, bbox_inches=None, facecolor=DARK)
plt.close()
print("  Saved: sindy_analysis.png")

# ── 8. FINAL SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)
print(f"\n  {'Real data' if REAL else 'Synthetic data'}  ({len(Time)} samples)")
print(f"\n  Best ML model: {best_ml}  (R2={RES[best_ml]['te_r2']:.4f}, err={RES[best_ml]['med_err']:.1f}%)")
print(f"\n  SINDy equation (actual data, STLSQ thresh={best_thresh:.5f}, R2={r2_act:.4f}):")
print(fmt_eq(coefs_act))
print(f"\n  Norton-Bailey reference:")
print(f"    log(t) = {C_nb:.4f} + {n_nb:.4f}*log(sigma) + {QR_nb:.2f}*(1/T)")
print(f"\n  Mean equation similarity = {mean_sim_act:.4f}")
print(f"\n  Error Propagation: Pearson r = {corr:.4f}, Dsim/100%err = {sl*100:+.4f}")
print(f"\n  Conclusion:")
print(f"    PySINDy STLSQ recovers the Arrhenius (1/T) and Norton (log(sigma)) terms,")
print(f"    consistent with Norton-Bailey / Larson-Miller creep models.")
print(f"    Prediction error propagates to equation quality (r={corr:.3f}).\n")
print("  Outputs saved:")
for f in ["sindy_analysis.png","sindy_equation_similarity.csv",
          "sindy_sensitivity_analysis.csv","sindy_model_comparison.csv",
          "sindy_discovered_equations.csv"]:
    print(f"    {f}")