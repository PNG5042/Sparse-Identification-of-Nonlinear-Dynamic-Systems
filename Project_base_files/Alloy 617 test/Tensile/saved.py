# ================================
# ALLOY 617 TENSILE – SINDy APPLICABILITY STUDY  (PySINDy edition)
# ================================
# Input CSVs (any subset found alongside this script will be loaded):
#   SGIHX_A1_DETAIL_DATA_Dspec.csv   Heat = 314626
#   SGIHX_A1_DETAIL_DATA_RBspec.csv  Heat = 188155
#   SGIHX_A1_DETAIL_DATA_Hspec.csv   Heat =  37458
#
# CSV columns used:
#   Specimen_Name, Material_Form, Heat, Count,
#   Elapsed_Time_Sec, Stress_MPa, Strain
#
# Per-specimen reduction:
#   UTS        = max(Stress_MPa)
#   eps_at_UTS = Strain at peak stress
#   strain_rate= median positive dε/dt  (from Elapsed_Time_Sec + Strain)
#
# SINDy target : UTS (MPa)
# SINDy library: Ramberg–Osgood / Cowper–Symonds physics
#   [1, log(sr), sr^0.05, log(eps), eps^0.2, sqrt(eps),
#    log(sr)*log(eps), sr*eps, H, H^2, H*log(sr), H*log(eps)]

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
np.random.seed(42)

CSV_FILES = [
    "SGIHX_A1_DETAIL_DATA_Dspec.csv",
    "SGIHX_A1_DETAIL_DATA_RBspec.csv",
    "SGIHX_A1_DETAIL_DATA_Hspec.csv",
]

# ── 0. LOAD & REDUCE DATA ─────────────────────────────────────────────
print("="*65)
print("ALLOY 617 TENSILE – SINDy UTS STUDY  (PySINDy)")
print("="*65)

frames = []
for fname in CSV_FILES:
    fpath = BASE / fname
    if fpath.exists():
        tmp = pd.read_csv(fpath)
        tmp["_source"] = Path(fname).stem
        frames.append(tmp)
        print(f"  Loaded: {fname}  ({len(tmp)} rows)")

REAL = len(frames) > 0

if REAL:
    df_raw = pd.concat(frames, ignore_index=True)
    for col in ["Heat", "Elapsed_Time_Sec", "Stress_MPa", "Strain", "Count"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    heat_vals = sorted(df_raw["Heat"].dropna().unique())
    form_vals = sorted(df_raw["Material_Form"].dropna().unique())
    heat_enc  = {h: i for i, h in enumerate(heat_vals)}
    form_enc  = {f: i for i, f in enumerate(form_vals)}
    df_raw["Heat_enc"] = df_raw["Heat"].map(heat_enc)
    df_raw["Form_enc"] = df_raw["Material_Form"].map(form_enc)

    records = []
    for spec, grp in df_raw.groupby("Specimen_Name"):
        grp = grp.sort_values("Count").dropna(
            subset=["Stress_MPa", "Strain", "Elapsed_Time_Sec"])
        if len(grp) < 3:
            continue

        idx_uts     = grp["Stress_MPa"].idxmax()
        uts         = float(grp.loc[idx_uts, "Stress_MPa"])
        eps_uts_val = float(grp.loc[idx_uts, "Strain"])

        dt   = grp["Elapsed_Time_Sec"].diff().clip(lower=1e-9)
        deps = grp["Strain"].diff()
        sr_s = (deps / dt).dropna()
        sr_s = sr_s[sr_s > 0]
        if len(sr_s) == 0:
            continue
        sr = float(sr_s.median())

        records.append({
            "Specimen_Name": spec,
            "UTS":           uts,
            "eps_at_UTS":    eps_uts_val,
            "strain_rate":   sr,
            "Heat_enc":      float(grp["Heat_enc"].iloc[0]),
            "Form_enc":      float(grp["Form_enc"].iloc[0]),
            "Heat":          grp["Heat"].iloc[0],
            "Material_Form": grp["Material_Form"].iloc[0],
        })

    summary = pd.DataFrame(records)
    summary = summary.dropna()
    summary = summary[
        (summary["UTS"]         > 0) &
        (summary["eps_at_UTS"]  > 1e-6) &
        (summary["strain_rate"] > 0)
    ].copy()

    UTS         = summary["UTS"].values.astype(float)
    eps_at_UTS  = summary["eps_at_UTS"].values.astype(float)
    strain_rate = summary["strain_rate"].values.astype(float)
    Heat_enc    = summary["Heat_enc"].values.astype(float)
    Form_enc    = summary["Form_enc"].values.astype(float)

    print(f"\n  Valid specimens: {len(summary)}")
    print(f"  UTS:         {UTS.min():.1f} – {UTS.max():.1f} MPa")
    print(f"  eps @ UTS:   {eps_at_UTS.min():.4f} – {eps_at_UTS.max():.4f}")
    print(f"  Strain rate: {strain_rate.min():.2e} – {strain_rate.max():.2e} 1/s")
    print(f"  Heats:       {summary['Heat'].unique().tolist()}")

else:
    # ── Synthetic Alloy 617 tensile (realistic scatter) ───────────────
    print("\n  CSVs not found – generating synthetic Alloy 617 tensile data")
    print("  (Place CSV files next to this script to use real data)\n")
    n   = 150
    rng = np.random.default_rng(42)

    Heat_enc    = rng.integers(0, 3, n).astype(float)
    Form_enc    = np.zeros(n)
    strain_rate = 10 ** rng.uniform(-4, -2, n)       # 1e-4 – 1e-2 s⁻¹
    eps_at_UTS  = rng.uniform(0.10, 0.40, n)

    # UTS = K * sr^m * eps^p + heat offset + noise
    K_true, m_true, p_true = 900.0, 0.025, 0.15
    h_offset = np.array([0.0, +30.0, -20.0])[Heat_enc.astype(int)]
    UTS = (K_true
           * (strain_rate ** m_true)
           * (eps_at_UTS  ** p_true)
           + h_offset
           + rng.normal(0, 8, n))


    print(f"  True: UTS = {K_true}*sr^{m_true}*eps^{p_true} + heat_offset + noise")
    print(f"  N={n},  UTS: {UTS.min():.1f}–{UTS.max():.1f} MPa")

y_uts = UTS
print(f"\nN={len(y_uts)},  UTS: {y_uts.min():.1f}–{y_uts.max():.1f} MPa  "
      f"(mean={y_uts.mean():.1f})")

# ── 1. ML PREDICTION MODEL ─────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: ML PREDICTION MODEL  (target = UTS, MPa)")
print("="*65)

log_sr  = np.log(np.clip(strain_rate, 1e-12, None))
sqrt_sr = np.sqrt(strain_rate)
log_eps = np.log(np.clip(eps_at_UTS,  1e-12, None))

def ml_feats(H, F, eps, sr, l_eps, l_sr, sq_sr):
    return np.column_stack([
        np.ones_like(H),
        H, H**2,
        F,
        eps, eps**2, np.sqrt(np.clip(eps, 0, None)), l_eps,
        sr, l_sr, sq_sr,
        eps * sr, eps * l_sr, l_eps * l_sr,
        H * eps, H * l_sr, H * l_eps,
        F * eps, F * l_sr,
    ])

X_all  = ml_feats(Heat_enc, Form_enc, eps_at_UTS, strain_rate,
                  log_eps, log_sr, sqrt_sr)
n_feat = min(12, X_all.shape[1])
sel    = SelectKBest(f_regression, k=n_feat)
X_sel  = sel.fit_transform(X_all, y_uts)
sc_ml  = StandardScaler()
X_sc   = sc_ml.fit_transform(X_sel)

n_samp = len(y_uts)
if n_samp >= 20:
    bins = np.digitize(y_uts, np.percentile(y_uts, [20, 40, 60, 80]))
    sss  = StratifiedShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, te_i = next(sss.split(X_sc, bins))
else:
    split = max(1, int(n_samp * 0.75))
    tr_i, te_i = np.arange(split), np.arange(split, n_samp)

Xtr, Xte = X_sc[tr_i], X_sc[te_i]
ytr, yte = y_uts[tr_i], y_uts[te_i]

n_cv  = min(5, max(2, len(tr_i) // 5))
rcv   = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10, 50, 100, 500]},
                     cv=n_cv, scoring='r2')
rcv.fit(Xtr, ytr)
ridge = rcv.best_estimator_

rf = RandomForestRegressor(300,
                            max_depth=6,
                            min_samples_split=max(4, len(tr_i)//15),
                            min_samples_leaf=max(2, len(tr_i)//25),
                            max_features='sqrt', random_state=42)
rf.fit(Xtr, ytr)
ens = VotingRegressor([('r1', ridge),
                        ('r2', Ridge(alpha=rcv.best_params_['alpha'])),
                        ('rf', rf)])
ens.fit(Xtr, ytr)

ML  = {'Ridge': ridge, 'RandomForest': rf, 'Ensemble': ens}
RES = {}
for nm, m in ML.items():
    yp  = m.predict(Xte)
    err = np.abs((yp - yte) / np.clip(np.abs(yte), 1, None)) * 100
    RES[nm] = dict(tr_r2=m.score(Xtr, ytr), te_r2=m.score(Xte, yte),
                   med_err=np.median(err), err=err, ypall=m.predict(X_sc))
    print(f"  {nm:<14} TrainR2={RES[nm]['tr_r2']:.4f}  "
          f"TestR2={RES[nm]['te_r2']:.4f}  "
          f"MedianErr={RES[nm]['med_err']:.1f}%")

best_ml = max(RES, key=lambda k: RES[k]['te_r2'])
print(f"\n  Best: {best_ml}  (R2={RES[best_ml]['te_r2']:.4f}, "
      f"err={RES[best_ml]['med_err']:.1f}%)")

# ── OLS reference: log(UTS) = logK + m*log(sr) + p*log(eps) ──────────
log_uts = np.log(y_uts)
A_ro    = np.column_stack([np.ones_like(log_sr), log_sr, log_eps])
ro_c, *_ = np.linalg.lstsq(A_ro, log_uts, rcond=None)
logK_ro, m_ro, p_ro = ro_c
K_ro = np.exp(logK_ro)
print(f"\n  Ramberg-Osgood OLS (log-space):")
print(f"    UTS = {K_ro:.2f} * sr^{m_ro:.5f} * eps^{p_ro:.5f}")
print(f"    (log(K)={logK_ro:.4f}, m={m_ro:.5f}, p={p_ro:.5f})")

# ── 2. PySINDy ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: SINDy – SPARSE EQUATION DISCOVERY  (PySINDy STLSQ)")
print("="*65)

FEAT_LABELS = [
    'C0',                # constant
    'log(sr)',           # rate sensitivity (Cowper-Symonds)
    'sr^0.05',           # direct power-law rate
    'log(eps)',          # log strain hardening
    'eps^0.2',           # Ramberg-Osgood hardening
    'sqrt(eps)',         # sqrt strain
    'log(sr)*log(eps)',  # rate × strain interaction
    'sr*eps',            # direct cross term
    'H',                 # heat (batch) effect
    'H^2',               # heat nonlinear
    'H*log(sr)',         # heat × rate interaction
    'H*log(eps)',        # heat × strain interaction
]

def build_phi(H, eps, sr):
    l_sr  = np.log(np.clip(sr,  1e-12, None))
    l_eps = np.log(np.clip(eps, 1e-12, None))
    return np.column_stack([
        np.ones(len(H)),
        l_sr,
        sr ** 0.05,
        l_eps,
        np.clip(eps, 0, None) ** 0.2,
        np.sqrt(np.clip(eps, 0, None)),
        l_sr * l_eps,
        sr * eps,
        H,
        H**2,
        H * l_sr,
        H * l_eps,
    ])

Phi = build_phi(Heat_enc, eps_at_UTS, strain_rate)

def fit_sindy(y_target, threshold):
    opt   = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=1000)
    model = ps.SINDy(feature_library=IdentityLibrary(), optimizer=opt)
    model.fit(Phi, t=1, x_dot=y_target.reshape(-1, 1))
    y_pred = np.asarray(model.predict(Phi)).ravel()
    ss_r   = np.sum((y_target - y_pred)**2)
    ss_t   = np.sum((y_target - y_target.mean())**2)
    r2v    = 1 - ss_r / ss_t if ss_t else 0.0
    raw    = model.coefficients().ravel()
    coef_dict = {lbl: float(v)
                 for lbl, v in zip(FEAT_LABELS, raw) if abs(v) > 1e-10}
    return coef_dict, r2v, y_pred, len(coef_dict)

print("\n  Tuning STLSQ threshold via grid:")
thresholds  = np.logspace(-2, 3, 80)
best_thresh = 1.0
best_r2_val = -np.inf
for thresh in thresholds:
    try:
        cd, rv, _, nact = fit_sindy(y_uts, thresh)
        if rv > best_r2_val and nact >= 2:
            best_r2_val, best_thresh = rv, thresh
    except Exception:
        continue
print(f"  Best threshold = {best_thresh:.4f}  (R2={best_r2_val:.4f})")

coefs_act, r2_act, yp_act, n_act_t = fit_sindy(y_uts, best_thresh)
print(f"\n  SINDy (actual):  R2={r2_act:.4f},  "
      f"{n_act_t} active: {list(coefs_act.keys())}")

coefs_preds = {}
r2_preds    = {}
yp_sindy_ml = {}
for nm in ML:
    cd, r2v, yhat, nact = fit_sindy(RES[nm]['ypall'], best_thresh)
    coefs_preds[nm] = cd
    r2_preds[nm]    = r2v
    yp_sindy_ml[nm] = yhat
    print(f"  SINDy ({nm}): R2={r2v:.4f},  {nact} active: {list(cd.keys())}")

# ── 3. PRINT EQUATIONS ────────────────────────────────────────────────
def fmt_eq(coef_dict, ylbl='UTS (MPa)'):
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
    print(f"\n  -- SINDy on {nm} "
          f"(ML err={RES[nm]['med_err']:.1f}%, R2={r2_preds[nm]:.4f}) --")
    print(fmt_eq(coefs_preds[nm]))

# ── 4. SIMILARITY vs OLS R-O coefficients ────────────────────────────
print("\n" + "="*65)
print("STEP 4: EQUATION SIMILARITY  (metric = 1 - |rel_error|)")
print("="*65)

# OLS reference is in log(UTS) space; SINDy is in raw UTS space.
# Best comparables: log(sr) coefficient maps to m, log(eps) maps to p.
analytic_ref = {
    'log(sr)':  m_ro,
    'log(eps)': p_ro,
}

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
print(f"\n  {'Param':<22} {'OLS (ref)':>12} {'SINDy':>12} "
      f"{'rel_err':>10} {'similarity':>11}")
print("  " + "-"*67)
for _, r in df_sim_act.iterrows():
    re = f"{r['rel_err']:.4f}"    if not np.isnan(r['rel_err'])   else "nan"
    si = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "nan"
    print(f"  {r['param']:<22} {r['analytical']:>12.5f} "
          f"{r['sindy']:>12.5f} {re:>10} {si:>11}")
mean_sim_act = df_sim_act['similarity'].dropna().mean()
print(f"\n  Mean similarity (actual data): {mean_sim_act:.4f}")

model_rows = []
for nm in ML:
    dfs  = compute_sim(coefs_preds[nm], analytic_ref)
    msim = dfs['similarity'].dropna().mean()
    model_rows.append({
        'model': nm, 'test_r2': RES[nm]['te_r2'],
        'median_err_pct': RES[nm]['med_err'],
        **{f"sim_{r['param']}": r['similarity'] for _, r in dfs.iterrows()},
        'mean_similarity': msim,
    })
    print(f"  {nm:<14} ML err={RES[nm]['med_err']:.1f}%  mean_sim={msim:.4f}")
df_models = pd.DataFrame(model_rows)

# ── 5. SENSITIVITY ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 5: SENSITIVITY – Prediction Error → Equation Quality")
print("="*65)

noise_levels = [0, 5, 10, 20, 30, 50, 75, 100, 150]   # MPa
rng2 = np.random.default_rng(7)
sens_rows = []

for s in noise_levels:
    y_n  = y_uts + rng2.normal(0, s, len(y_uts))
    cn, _, _, _ = fit_sindy(y_n, best_thresh)
    dfs  = compute_sim(cn, analytic_ref)
    msim = dfs['similarity'].dropna().mean()
    pct  = 100 * s / max(y_uts.mean(), 1)
    nact = len(cn)
    m_v  = cn.get('log(sr)',  0)
    p_v  = cn.get('log(eps)', 0)
    sens_rows.append({'noise_MPa': s, 'approx_pct_err': pct,
                      'n_active': nact, 'mean_sim': msim,
                      'sindy_m': m_v, 'sindy_p': p_v})
    print(f"  noise={s:>4} MPa (~{pct:>5.1f}%)  "
          f"active={nact:2d}  sim={msim:.4f}  "
          f"m={m_v:.5f}  p={p_v:.5f}")

df_sens = pd.DataFrame(sens_rows)
valid = ~df_sens['mean_sim'].isna()
pv = df_sens.loc[valid, 'approx_pct_err'].values
sv = df_sens.loc[valid, 'mean_sim'].values

if len(pv) > 3:
    corr, pval = stats.pearsonr(pv, sv)
    sl, ic, *_ = stats.linregress(pv, sv)
    at10 = ic + sl * 10
    print(f"\n  Pearson r = {corr:.4f}  (p = {pval:.4f})")
    print(f"  Linear: sim = {ic:.4f} + {sl:.5f} * (% err)")
    print(f"  At 10% pred error: sim ~ {at10:.4f}")
else:
    corr = pval = sl = ic = at10 = np.nan

# ── 6. SAVE ───────────────────────────────────────────────────────────
df_sim_act.to_csv(OUT/"sindy_617t_equation_similarity.csv",  index=False, float_format='%.6f')
df_sens.to_csv(   OUT/"sindy_617t_sensitivity_analysis.csv", index=False, float_format='%.6f')
df_models.to_csv( OUT/"sindy_617t_model_comparison.csv",     index=False, float_format='%.6f')
eq_rows = [{'term': lbl,
             'sindy_actual': coefs_act.get(lbl, 0.),
             **{f'sindy_{nm}': coefs_preds[nm].get(lbl, 0.) for nm in ML}}
           for lbl in analytic_ref]
pd.DataFrame(eq_rows).to_csv(OUT/"sindy_617t_discovered_equations.csv",
                              index=False, float_format='%.6f')
if REAL:
    summary.to_csv(OUT/"sindy_617t_specimen_summary.csv", index=False, float_format='%.6f')
print("\n  CSVs saved.")

# ── 7. VISUALISATION ──────────────────────────────────────────────────
DARK='#0d1117'; PANEL='#161b22'; GRID='#21262d'
C1='#58a6ff'; C2='#f85149'; C3='#3fb950'; C4='#d29922'; C5='#bc8cff'; TEXT='#c9d1d9'
MC = [C1, C3, C4]
HC = [C1, C3, C4]   # per-heat colours

def sax(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=C1, fontsize=10, fontweight='bold', pad=7)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=9)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=9)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)

fig = plt.figure(figsize=(22, 18), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
y_best = RES[best_ml]['ypall']
n_heats = int(Heat_enc.max()) + 1
uts_ll  = [y_uts.min() - 20, y_uts.max() + 20]

# P1 – ML predicted vs actual, coloured by heat
ax = fig.add_subplot(gs[0, 0])
for hi in range(n_heats):
    mask = Heat_enc == hi
    ax.scatter(y_uts[mask], y_best[mask], s=45, alpha=0.8,
               color=HC[hi % len(HC)], edgecolors='none',
               label=f"Heat {hi}", zorder=3)
ax.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax, f'ML ({best_ml})  Actual vs Predicted UTS',
    'Actual UTS (MPa)', 'Predicted UTS (MPa)')
ax.text(0.05, 0.92, f"R2={RES[best_ml]['te_r2']:.4f}",
        transform=ax.transAxes, color=C3, fontsize=9, fontweight='bold')

# P2 – SINDy on actual
ax2 = fig.add_subplot(gs[0, 1])
for hi in range(n_heats):
    mask = Heat_enc == hi
    ax2.scatter(y_uts[mask], yp_act[mask], s=45, alpha=0.8,
                color=HC[hi % len(HC)], edgecolors='none')
ax2.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
sax(ax2, f'SINDy (actual)  R2={r2_act:.4f}\n{n_act_t} active terms',
    'Actual UTS (MPa)', 'SINDy UTS (MPa)')

# P3 – UTS vs log10(strain rate), by heat
ax3 = fig.add_subplot(gs[0, 2])
for hi in range(n_heats):
    mask = Heat_enc == hi
    ax3.scatter(np.log10(strain_rate[mask]), y_uts[mask], s=45, alpha=0.8,
                color=HC[hi % len(HC)], edgecolors='none', label=f"Heat {hi}")
ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax3, 'UTS vs Strain Rate  (by Heat)',
    'log₁₀(Strain Rate)  [1/s]', 'UTS (MPa)')

# P4 – Similarity bars
ax4 = fig.add_subplot(gs[1, 0])
plabs = df_sim_act['param'].tolist()
xp    = np.arange(len(plabs))
w     = 0.22
srcs  = [('SINDy(actual)', C1, df_sim_act)]
for nm, col in zip(ML, MC[1:]):
    srcs.append((nm, col, compute_sim(coefs_preds[nm], analytic_ref)))
for i, (lbl, col, df_s) in enumerate(srcs):
    off  = (i - len(srcs)/2 + 0.5) * w
    bars = ax4.bar(xp+off, df_s['similarity'].fillna(0), w,
                   label=lbl, color=col, alpha=0.85, edgecolor=PANEL)
    for bar, val in zip(bars, df_s['similarity']):
        if not np.isnan(val):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+(0.02 if val>=0 else -0.08),
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=7, color=TEXT)
ax4.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.4)
ax4.axhline(0.0, color=C2,  ls='--', lw=1, alpha=0.4)
ax4.set_xticks(xp); ax4.set_xticklabels(plabs, fontsize=8, color=TEXT)
ax4.set_ylim([-0.5, 1.4])
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax4, 'Equation Similarity by Parameter\n(1 = perfect match)', '', 'Similarity')

# P5 – Sensitivity curve
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df_sens['approx_pct_err'], df_sens['mean_sim'], 'o-',
         color=C1, lw=2, ms=6, label='Mean similarity')
if not np.isnan(sl):
    xf = np.linspace(0, df_sens['approx_pct_err'].max(), 200)
    ax5.plot(xf, ic+sl*xf, '--', color=C2, lw=1.5,
             label=f'Trend (d/10%={sl*10:+.4f})')
ax5.axhline(1.0, color=C3, ls=':', alpha=0.5, lw=1, label='Sim=1')
ax5.axhline(0.5, color=C4, ls=':', alpha=0.5, lw=1, label='Sim=0.5')
ax5.set_ylim([-0.5, 1.3])
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax5, 'Error Propagation: Prediction Error → Equation Quality',
    'Approx. Prediction Error (%)', 'Mean Equation Similarity')

# P6 – Sparsity vs noise
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(df_sens['approx_pct_err'], df_sens['n_active'], 's-',
         color=C5, lw=2, ms=6)
ax6.set_ylim([0, len(FEAT_LABELS)+1])
sax(ax6, 'SINDy Sparsity vs Prediction Error',
    'Approx. Prediction Error (%)', 'N Active Terms')

# P7 – Model scatter
ax7 = fig.add_subplot(gs[2, 0])
for i, row in df_models.iterrows():
    ax7.scatter(row['median_err_pct'], row['mean_similarity'],
                s=250, color=MC[i], zorder=5, edgecolors='white', lw=0.8)
    ax7.annotate(row['model'], (row['median_err_pct'], row['mean_similarity']),
                 textcoords='offset points', xytext=(8, 5),
                 fontsize=9, color=MC[i], fontweight='bold')
sax(ax7, 'ML Model: Prediction Error vs Equation Similarity',
    'Median Prediction Error (%)', 'Mean Equation Similarity')

# P8 – Coefficient m and p vs noise
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(PANEL)
pa = df_sens['approx_pct_err'].values
ax8.plot(pa, df_sens['sindy_m'].values, 'o-', color=C3, lw=2, ms=5,
         label='SINDy m (rate)')
ax8.axhline(m_ro, color=C3, ls='--', alpha=0.7, lw=1.5,
            label=f'OLS m={m_ro:.5f}')
ax8b = ax8.twinx(); ax8b.set_facecolor(PANEL)
ax8b.plot(pa, df_sens['sindy_p'].values, 's-', color=C4, lw=2, ms=5,
          label='SINDy p (strain)')
ax8b.axhline(p_ro, color=C4, ls='--', alpha=0.7, lw=1.5,
             label=f'OLS p={p_ro:.5f}')
ax8b.tick_params(colors=TEXT, labelsize=7)
ax8b.set_ylabel('p  (strain exp)', color=C4, fontsize=8)
for sp in ax8.spines.values(): sp.set_edgecolor(GRID)
ax8.tick_params(colors=TEXT, labelsize=8)
ax8.grid(True, color=GRID, alpha=0.5, lw=0.5)
ax8.set_title('Coefficient Stability vs Prediction Error',
              color=C1, fontsize=10, fontweight='bold', pad=7)
ax8.set_xlabel('Approx. Prediction Error (%)', color=TEXT, fontsize=9)
ax8.set_ylabel('m  (rate exp)', color=C3, fontsize=8)
l1, b1 = ax8.get_legend_handles_labels()
l2, b2 = ax8b.get_legend_handles_labels()
ax8.legend(l1+l2, b1+b2, fontsize=7, facecolor=PANEL,
           edgecolor=GRID, labelcolor=TEXT)

# P9 – Text summary
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#0d1117')
for sp in ax9.spines.values(): sp.set_edgecolor(GRID)
ax9.axis('off')
C0s = coefs_act.get('C0',       0)
ms  = coefs_act.get('log(sr)',  0)
ps_ = coefs_act.get('log(eps)', 0)
txt = [
    "Material: Alloy 617  |  Test: Tensile",
    f"Data: {'Real CSV' if REAL else 'Synthetic'}  N={len(y_uts)}",
    f"Target: UTS (MPa)",
    "",
    "-- R-O OLS Reference --------------",
    f"  K  = {K_ro:.2f} MPa",
    f"  m  = {m_ro:.5f}  (rate exponent)",
    f"  p  = {p_ro:.5f}  (strain exponent)",
    f"  UTS = K * sr^m * eps^p",
    "",
    f"-- SINDy actual (R2={r2_act:.4f}) ---",
    f"  C0       = {C0s:+.5g}",
    f"  log(sr)  = {ms:+.5g}",
    f"  log(eps) = {ps_:+.5g}",
    f"  ({n_act_t} total active terms)",
    "",
    "-- Similarity ---------------------",
]
for _, r in df_sim_act.iterrows():
    si = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "nan"
    txt.append(f"  {r['param']:<20}: {si}")
txt += [
    f"  MEAN = {mean_sim_act:.4f}",
    "",
    "-- Error Propagation --------------",
    f"  Pearson r = {corr:.4f}",
    (f"  Dsim/10%err = {sl*10:+.4f}" if not np.isnan(sl) else "  n/a"),
    "",
    "  [PySINDy v2 | STLSQ | IdentityLib]",
]
ax9.text(0.03, 0.97, "\n".join(txt), transform=ax9.transAxes,
         fontsize=7.8, va='top', fontfamily='monospace', color=TEXT,
         bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.9))
ax9.set_title('Alloy 617 Tensile Summary', color=C1,
              fontsize=10, fontweight='bold')

fig.suptitle(
    "Alloy 617 Tensile – SINDy UTS Study  (PySINDy 2.x)\n"
    "Sparse Equation Discovery  |  Ramberg–Osgood Reference  |  Error Propagation",
    fontsize=14, fontweight='bold', color=C1, y=0.998)

plt.savefig(OUT/"sindy_617t_analysis.png", dpi=150,
            bbox_inches=None, facecolor=DARK)
plt.close()
print("  Saved: sindy_617t_analysis.png")

# ── 8. FINAL SUMMARY ──────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY – ALLOY 617 TENSILE (UTS)")
print("="*65)
print(f"\n  {'Real CSV data' if REAL else 'Synthetic data'}  "
      f"({len(y_uts)} specimens)")
print(f"  Best ML model: {best_ml}  "
      f"(R2={RES[best_ml]['te_r2']:.4f}, "
      f"err={RES[best_ml]['med_err']:.1f}%)")
print(f"\n  Ramberg-Osgood OLS reference:")
print(f"    UTS = {K_ro:.2f} * sr^{m_ro:.5f} * eps^{p_ro:.5f}")
print(f"\n  SINDy (STLSQ thresh={best_thresh:.4f}, R2={r2_act:.4f}):")
print(fmt_eq(coefs_act))
print(f"\n  Mean equation similarity = {mean_sim_act:.4f}")
if not np.isnan(corr):
    print(f"  Pearson r (noise vs sim) = {corr:.4f}")
print(f"\n  Literature Alloy 617 tensile (RT):")
print(f"    UTS ~ 690–850 MPa,  strain-rate sensitivity m ~ 0.01–0.05")
print(f"\n  Outputs:")
for f in ["sindy_617t_analysis.png",
          "sindy_617t_equation_similarity.csv",
          "sindy_617t_sensitivity_analysis.csv",
          "sindy_617t_model_comparison.csv",
          "sindy_617t_discovered_equations.csv"]:
    print(f"    {f}")
if REAL:
    print(f"    sindy_617t_specimen_summary.csv")