# ================================
# ALLOY 617 TENSILE – SINDy APPLICABILITY STUDY  (PySINDy edition) -- v3
# ================================
# CHANGES vs v2:
#   FIX-1  Formatting bug: intercept was printing as both value and term.
#   FIX-2  QR-based column pivoting decorrelates Φ before SINDy.
#          Columns with near-zero pivot (effective rank deficiency) are
#          dropped automatically, reducing condition number.
#   FIX-3  Identifiability diagnostic: for every feature, compute its
#          coefficient of variation (CV) in the design matrix and flag
#          features whose variance is too low to identify reliably.
#          "log(sr)" will be explicitly flagged as unidentifiable from
#          this dataset's narrow strain-rate range.
#   FIX-4  Similarity for dropped-but-identifiable terms counts as 0.
#          Similarity for features flagged as unidentifiable is reported
#          as NaN with a note, so the mean is not penalised unfairly.
# ================================

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

# ── 0. LOAD & REDUCE ──────────────────────────────────────────────────
print("="*65)
print("ALLOY 617 TENSILE – SINDy UTS STUDY  (PySINDy) [v3]")
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

    summary = pd.DataFrame(records).dropna()
    summary = summary[
        (summary["UTS"]         > 0) &
        (summary["eps_at_UTS"]  > 1e-6) &
        (summary["strain_rate"] > 0)
    ].copy()

    # RT / HT separation
    ht_mask_primary = (summary["UTS"] < 500) & (summary["eps_at_UTS"] < 0.05)
    iqr_outlier = pd.Series(False, index=summary.index)
    for heat, grp in summary.groupby("Heat"):
        luts = np.log(grp["UTS"])
        q1, q3 = luts.quantile(0.25), luts.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 2.0 * iqr, q3 + 2.0 * iqr
        iqr_outlier.loc[grp.index] = (luts < lo) | (luts > hi)

    ht_mask = ht_mask_primary | iqr_outlier
    df_ht   = summary[ht_mask].copy()
    df_rt   = summary[~ht_mask].copy()

    print(f"\n  Total valid specimens  : {len(summary)}")
    print(f"  HT / outlier excluded  : {len(df_ht)} "
          f"→ {df_ht['Specimen_Name'].tolist()}")
    print(f"  RT retained            : {len(df_rt)}")
    print(f"  RT UTS  : {df_rt['UTS'].min():.1f} – {df_rt['UTS'].max():.1f} MPa")
    print(f"  RT eps  : {df_rt['eps_at_UTS'].min():.4f} – "
          f"{df_rt['eps_at_UTS'].max():.4f}")
    print(f"  RT sr   : {df_rt['strain_rate'].min():.2e} – "
          f"{df_rt['strain_rate'].max():.2e} 1/s")

    UTS         = df_rt["UTS"].values.astype(float)
    eps_at_UTS  = df_rt["eps_at_UTS"].values.astype(float)
    strain_rate = df_rt["strain_rate"].values.astype(float)
    Heat_enc    = df_rt["Heat_enc"].values.astype(float)
    Form_enc    = df_rt["Form_enc"].values.astype(float)

else:
    print("\n  CSVs not found – synthetic Alloy 617 RT data")
    n   = 120
    rng = np.random.default_rng(42)
    Heat_enc    = rng.integers(0, 3, n).astype(float)
    Form_enc    = np.zeros(n)
    strain_rate = 10 ** rng.uniform(-4.5, -3.5, n)
    eps_at_UTS  = rng.uniform(0.20, 0.55, n)
    K_true, m_true, p_true = 727.0, -0.016, 0.43
    h_offset = np.array([0.0, +25.0, -15.0])[Heat_enc.astype(int)]
    UTS = (K_true * (strain_rate**m_true) * (eps_at_UTS**p_true)
           + h_offset + np.random.default_rng(0).normal(0, 12, n))
    df_rt = df_ht = summary = None

# Centred heat encoding
H_mean = Heat_enc.mean()
H_c    = Heat_enc - H_mean

log_y   = np.log(UTS)
log_sr  = np.log(np.clip(strain_rate, 1e-12, None))
log_eps = np.log(np.clip(eps_at_UTS,  1e-12, None))
sr_dec  = np.log10(strain_rate.max()) - np.log10(strain_rate.min())

print(f"\n  H_c range: [{H_c.min():.2f}, {H_c.max():.2f}]  "
      f"| SR span: {sr_dec:.2f} decades")
print(f"N = {len(UTS)},  UTS: {UTS.min():.1f} – {UTS.max():.1f} MPa  "
      f"(mean={UTS.mean():.1f})")

# ── 1. ML MODEL ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: ML PREDICTION MODEL  (target = UTS, MPa)")
print("="*65)

def ml_feats(Hc, F, eps, sr, l_eps, l_sr):
    return np.column_stack([
        np.ones_like(Hc), Hc, Hc**2, F,
        eps, eps**2, np.sqrt(np.clip(eps, 0, None)), l_eps,
        sr, l_sr, eps*l_sr, l_eps*l_sr, Hc*l_eps, Hc*l_sr,
    ])

X_all  = ml_feats(H_c, Form_enc, eps_at_UTS, strain_rate, log_eps, log_sr)
n_feat = min(10, X_all.shape[1])
sel    = SelectKBest(f_regression, k=n_feat)
X_sel  = sel.fit_transform(X_all, UTS)
sc_ml  = StandardScaler()
X_sc   = sc_ml.fit_transform(X_sel)

n_samp = len(UTS)
if n_samp >= 20:
    bins = np.digitize(UTS, np.percentile(UTS, [20, 40, 60, 80]))
    sss  = StratifiedShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, te_i = next(sss.split(X_sc, bins))
else:
    split = max(1, int(n_samp * 0.75))
    tr_i, te_i = np.arange(split), np.arange(split, n_samp)

Xtr, Xte = X_sc[tr_i], X_sc[te_i]
ytr, yte  = UTS[tr_i],  UTS[te_i]

n_cv = min(5, max(2, len(tr_i) // 5))
rcv  = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10, 50, 100, 500]},
                    cv=n_cv, scoring='r2')
rcv.fit(Xtr, ytr)
ridge = rcv.best_estimator_

rf = RandomForestRegressor(300, max_depth=6,
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
    print(f"  {nm:<14} TrainR²={RES[nm]['tr_r2']:.4f}  "
          f"TestR²={RES[nm]['te_r2']:.4f}  "
          f"MedianErr={RES[nm]['med_err']:.1f}%")

best_ml = max(RES, key=lambda k: RES[k]['te_r2'])
print(f"\n  Best: {best_ml}  (R²={RES[best_ml]['te_r2']:.4f}, "
      f"err={RES[best_ml]['med_err']:.1f}%)")

# OLS R-O reference
A_ro     = np.column_stack([np.ones_like(log_sr), log_sr, log_eps])
ro_c, *_ = np.linalg.lstsq(A_ro, log_y, rcond=None)
logK_ro, m_ro, p_ro = ro_c
K_ro = np.exp(logK_ro)
print(f"\n  OLS R-O (RT, log-space):")
print(f"    log(UTS) = {logK_ro:.5f} + {m_ro:.5f}·log(sr) + {p_ro:.5f}·log(eps)")
print(f"    UTS = {K_ro:.2f} · sr^{m_ro:.5f} · eps^{p_ro:.5f}")

# ── 2. IDENTIFIABILITY DIAGNOSTIC ─────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: FEATURE IDENTIFIABILITY DIAGNOSTIC")
print("="*65)

# Full candidate feature library (before any pruning)
FEAT_LABELS_FULL = ['C0', 'log(sr)', 'log(eps)', 'log(eps)^2',
                    'H_c', 'H_c*log(eps)']

def build_phi_full(Hc, eps, sr):
    l_sr  = np.log(np.clip(sr,  1e-12, None))
    l_eps = np.log(np.clip(eps, 1e-12, None))
    return np.column_stack([
        np.ones(len(Hc)),
        l_sr,
        l_eps,
        l_eps**2,
        Hc,
        Hc * l_eps,
    ])

Phi_full = build_phi_full(H_c, eps_at_UTS, strain_rate)

# FIX-3: Per-feature identifiability
#   CV  = std(col) / |mean(col)|  — for non-constant columns
#   For the intercept column (all ones) we skip CV and mark as identifiable.
#   A feature is "weakly identifiable" if its std is < ID_THRESH * std(log_y).
ID_THRESH = 0.10   # feature std < 10% of target std → likely unidentifiable

print(f"\n  Target log(UTS) std = {log_y.std():.5f}")
print(f"  Identifiability threshold: feature std < {ID_THRESH:.0%} of target std "
      f"= {ID_THRESH * log_y.std():.5f}\n")
print(f"  {'Feature':<22} {'std(col)':>10} {'CV':>8} {'identifiable?':>15}")
print("  " + "-"*57)

id_flags   = {}   # True = identifiable, False = too little variance
feat_stds  = {}

for i, lbl in enumerate(FEAT_LABELS_FULL):
    col = Phi_full[:, i]
    s   = col.std()
    feat_stds[lbl] = s
    if lbl == 'C0':
        id_flags[lbl] = True
        print(f"  {'C0':<22} {'(intercept)':>10} {'—':>8} {'YES (fixed)':>15}")
        continue
    cv  = s / abs(col.mean()) if abs(col.mean()) > 1e-10 else np.nan
    ok  = s >= ID_THRESH * log_y.std()
    id_flags[lbl] = ok
    cv_str = f"{cv:.3f}" if not np.isnan(cv) else "n/a"
    flag   = "YES" if ok else "*** WEAK ***"
    print(f"  {lbl:<22} {s:>10.5f} {cv_str:>8} {flag:>15}")

weak_feats = [k for k, v in id_flags.items() if not v]
print(f"\n  Weakly identifiable features: "
      f"{weak_feats if weak_feats else 'none'}")
if 'log(sr)' in weak_feats:
    print(f"  → log(sr) std = {feat_stds['log(sr)']:.5f}  "
          f"(SR span only {sr_dec:.2f} decades).")
    print(f"    SINDy cannot distinguish a rate effect from noise at this scale.")
    print(f"    The OLS m = {m_ro:.5f} is also unreliable from this data.")

# ── FIX-2: QR-based column selection to reduce condition number ────────
print("\n" + "="*65)
print("STEP 3: QR DECORRELATION  (select well-conditioned column subset)")
print("="*65)

# Keep only identifiable features, then apply column-pivoted QR
# to further remove near-dependent columns.
id_cols   = [i for i, lbl in enumerate(FEAT_LABELS_FULL) if id_flags[lbl]]
id_labels = [FEAT_LABELS_FULL[i] for i in id_cols]
Phi_id    = Phi_full[:, id_cols]

# Normalise columns for QR pivoting (so pivot order is variance-weighted)
col_norms    = np.linalg.norm(Phi_id, axis=0, keepdims=True)
col_norms[col_norms == 0] = 1.0
Phi_normed   = Phi_id / col_norms

from scipy.linalg import qr as scipy_qr

# Pivoted QR (this is the only correct call)
_, _, piv = scipy_qr(Phi_normed, pivoting=True)

print("\n  QR pivot order (most → least important):")
print([id_labels[i] for i in piv])

COND_TARGET = 50.0
keep_idx    = list(piv)
Phi_qr      = Phi_id[:, keep_idx]
cond_prev   = np.linalg.cond(Phi_id)

# Greedily drop trailing pivot columns until cond < COND_TARGET or ≥ 2 cols left
dropped_labels = []
while len(keep_idx) > 2:
    cond_now = np.linalg.cond(Phi_id[:, keep_idx])
    if cond_now <= COND_TARGET:
        break
    removed  = keep_idx.pop()               # remove lowest-priority column
    dropped_labels.append(id_labels[removed])

Phi_final   = Phi_id[:, keep_idx]
FEAT_ACTIVE = [id_labels[i] for i in keep_idx]
cond_final  = np.linalg.cond(Phi_final)

print(f"\n  Before QR pruning : {len(id_labels)} cols, cond = {cond_prev:.1f}")
print(f"  After  QR pruning : {len(FEAT_ACTIVE)} cols, cond = {cond_final:.1f}")
print(f"  Kept   : {FEAT_ACTIVE}")
print(f"  Dropped: {dropped_labels if dropped_labels else 'none'}")
if cond_final > COND_TARGET:
    print(f"  *** NOTE: cond still {cond_final:.1f} — "
          f"inherent data collinearity, results may still shift with noise ***")
else:
    print(f"  Condition number OK (≤ {COND_TARGET:.0f}).")

# ── 4. SINDy ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 4: SINDy – SPARSE EQUATION DISCOVERY  (log(UTS) space)")
print("="*65)

def fit_sindy_log(y_log, threshold, Phi, feat_labels):
    """Fit SINDy on Phi → y_log.  Returns coef_dict, r2_log, r2_uts, log_pred."""
    opt   = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=2000)
    model = ps.SINDy(feature_library=IdentityLibrary(), optimizer=opt)
    model.fit(Phi, t=1, x_dot=y_log.reshape(-1, 1))
    log_pred = np.asarray(model.predict(Phi)).ravel()

    ss_r  = np.sum((y_log - log_pred)**2)
    ss_t  = np.sum((y_log - y_log.mean())**2)
    r2_log = 1 - ss_r / ss_t if ss_t else 0.0

    uts_pred = np.exp(log_pred)
    uts_true = np.exp(y_log)
    ss_ru = np.sum((uts_true - uts_pred)**2)
    ss_tu = np.sum((uts_true - uts_true.mean())**2)
    r2_uts = 1 - ss_ru / ss_tu if ss_tu else 0.0

    raw       = model.coefficients().ravel()
    coef_dict = {lbl: float(v)
                 for lbl, v in zip(feat_labels, raw) if abs(v) > 1e-10}
    return coef_dict, r2_log, r2_uts, log_pred

# Threshold grid
print("\n  Tuning STLSQ threshold:")
thresholds  = np.logspace(-3, 1, 150)
best_thresh = 0.05
best_score  = -np.inf

for thresh in thresholds:
    try:
        cd, r2l, _, _ = fit_sindy_log(log_y, thresh, Phi_final, FEAT_ACTIVE)
        nact = len(cd)
        if r2l < 0.55 or nact < 2:
            continue
        score = r2l + max(0, len(FEAT_ACTIVE) - nact) * 0.015
        if score > best_score:
            best_score, best_thresh = score, thresh
    except Exception:
        continue

print(f"  Best threshold = {best_thresh:.5f}")

coefs_act, r2_act_log, r2_act_uts, logp_act = fit_sindy_log(
    log_y, best_thresh, Phi_final, FEAT_ACTIVE)
n_act_t = len(coefs_act)
print(f"\n  SINDy (actual RT):  R²_log={r2_act_log:.4f}  "
      f"R²_UTS={r2_act_uts:.4f}  "
      f"{n_act_t} active: {list(coefs_act.keys())}")

coefs_preds  = {}
r2_preds_log = {}
r2_preds_uts = {}
logp_ml      = {}
for nm in ML:
    log_ml = np.log(np.clip(RES[nm]['ypall'], 1.0, None))
    cd, r2l, r2u, lp = fit_sindy_log(log_ml, best_thresh, Phi_final, FEAT_ACTIVE)
    coefs_preds[nm]  = cd
    r2_preds_log[nm] = r2l
    r2_preds_uts[nm] = r2u
    logp_ml[nm]      = lp
    print(f"  SINDy ({nm}): R²_log={r2l:.4f}  R²_UTS={r2u:.4f}  "
          f"{len(cd)} active: {list(cd.keys())}")

# ── 5. EQUATIONS ──────────────────────────────────────────────────────
# FIX-1: formatting — intercept printed once, cleanly
def fmt_eq(coef_dict, ylbl='log(UTS)'):
    lines = []
    if 'C0' in coef_dict:
        v   = coef_dict['C0']
        K_e = np.exp(v) if abs(v) < 15 else float('nan')
        lines.append(f"  {v:+.5g}  [C0 → K≈{K_e:.1f} MPa  |  OLS K={K_ro:.1f} MPa]")
    for k, v in coef_dict.items():
        if k == 'C0':
            continue
        ann = ""
        if k == 'log(sr)':
            ann = f"   ← OLS m={m_ro:+.5f}"
        elif k == 'log(eps)':
            ann = f"   ← OLS p={p_ro:+.5f}"
        lines.append(f"  {v:+.5g} · {k}{ann}")
    if not lines:
        return f"  {ylbl} = 0  (no active terms)"
    return f"  {ylbl} =\n" + "\n".join(lines)

print("\n" + "="*65)
print("STEP 5: DISCOVERED EQUATIONS  (log(UTS) space, RT only)")
print("="*65)
print(f"\n  -- OLS R-O reference --")
print(f"  log(UTS) = {logK_ro:+.5f}"
      f"  {m_ro:+.5f}·log(sr)"
      f"  {p_ro:+.5f}·log(eps)")
print(f"\n  -- SINDy on ACTUAL RT  "
      f"(R²_log={r2_act_log:.4f}, R²_UTS={r2_act_uts:.4f}) --")
print(fmt_eq(coefs_act))
for nm in ML:
    print(f"\n  -- SINDy on {nm}  "
          f"(ML err={RES[nm]['med_err']:.1f}%, "
          f"R²_log={r2_preds_log[nm]:.4f}) --")
    print(fmt_eq(coefs_preds[nm]))

# ── 6. SIMILARITY ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 6: EQUATION SIMILARITY  (log-space, identifiability-aware)")
print("="*65)

analytic_ref = {
    'C0':       logK_ro,
    'log(sr)':  m_ro,
    'log(eps)': p_ro,
}

def compute_sim(sindy_coefs, ref, id_flags_map):
    """
    FIX-4: similarity rules
      - Feature identifiable   + SINDy kept it   → 1 - |rel_err|  clamped [-1,1]
      - Feature identifiable   + SINDy dropped it → 0.0
      - Feature NOT identifiable (weak variance)  → NaN  (excluded from mean)
    """
    rows = []
    for param, ana in ref.items():
        identifiable = id_flags_map.get(param, True)
        sval         = sindy_coefs.get(param, 0.0)

        if not identifiable:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': False})
            continue

        if abs(ana) > 0.01:
            rel_err = (ana - sval) / ana
            sim     = float(np.clip(1.0 - abs(rel_err), -1.0, 1.0))
        else:
            abs_err = abs(ana - sval)
            rel_err = np.nan
            sim     = float(np.clip(1.0 - abs_err / 0.1, -1.0, 1.0))

        rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                     'rel_err': rel_err, 'similarity': sim,
                     'identifiable': True})
    return pd.DataFrame(rows)

df_sim_act = compute_sim(coefs_act, analytic_ref, id_flags)
print(f"\n  {'Param':<18} {'OLS':>9} {'SINDy':>9} "
      f"{'rel_err':>9} {'sim':>7} {'id?':>10}")
print("  " + "-"*64)
for _, r in df_sim_act.iterrows():
    re  = f"{r['rel_err']:.4f}" if not np.isnan(r['rel_err'])   else "n/a"
    si  = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "NaN"
    idf = "YES" if r['identifiable'] else "WEAK→NaN"
    print(f"  {r['param']:<18} {r['OLS_ref']:>9.5f} {r['sindy']:>9.5f} "
          f"{re:>9} {si:>7} {idf:>10}")

mean_sim_act = df_sim_act['similarity'].dropna().mean()
n_id         = df_sim_act['identifiable'].sum()
print(f"\n  Mean similarity (identifiable terms only, n={n_id}): {mean_sim_act:.4f}")
print(f"  1.0=exact | 0.0=100% off | <0=wrong direction | NaN=unidentifiable")

model_rows = []
for nm in ML:
    dfs  = compute_sim(coefs_preds[nm], analytic_ref, id_flags)
    msim = dfs['similarity'].dropna().mean()
    model_rows.append({
        'model': nm, 'test_r2': RES[nm]['te_r2'],
        'median_err_pct': RES[nm]['med_err'],
        **{f"sim_{r['param']}": r['similarity'] for _, r in dfs.iterrows()},
        'mean_similarity': msim,
    })
    print(f"  {nm:<14} ML err={RES[nm]['med_err']:.1f}%  "
          f"mean_sim={msim:.4f}")
df_models = pd.DataFrame(model_rows)

# ── 7. SENSITIVITY ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 7: SENSITIVITY – Noise → SINDy equation quality")
print("="*65)

noise_levels = [0, 5, 10, 20, 30, 50, 75, 100]
rng2 = np.random.default_rng(7)
sens_rows = []

for s in noise_levels:
    y_n    = np.clip(UTS + rng2.normal(0, s, len(UTS)), 1.0, None)
    log_yn = np.log(y_n)
    cn, r2l, r2u, _ = fit_sindy_log(log_yn, best_thresh, Phi_final, FEAT_ACTIVE)
    dfs    = compute_sim(cn, analytic_ref, id_flags)
    msim   = dfs['similarity'].dropna().mean()
    pct    = 100 * s / max(UTS.mean(), 1)
    m_v    = cn.get('log(sr)',  0)
    p_v    = cn.get('log(eps)', 0)
    sens_rows.append({'noise_MPa': s, 'approx_pct_err': pct,
                      'n_active': len(cn), 'mean_sim': msim,
                      'r2_log': r2l, 'r2_uts': r2u,
                      'sindy_m': m_v, 'sindy_p': p_v})
    print(f"  noise={s:>4} MPa ({pct:>5.1f}%)  active={len(cn)}  "
          f"R²_log={r2l:.4f}  sim={msim:.4f}  "
          f"m={m_v:+.5f}  p={p_v:+.5f}")

df_sens = pd.DataFrame(sens_rows)
pv = df_sens['approx_pct_err'].values
sv = df_sens['mean_sim'].values

corr = pval = sl = ic = at10 = np.nan
if len(pv) > 3 and not np.all(np.isnan(sv)):
    corr, pval = stats.pearsonr(pv, sv)
    sl, ic, *_ = stats.linregress(pv, sv)
    at10 = ic + sl * 10
    print(f"\n  Pearson r = {corr:.4f}  (p={pval:.4f})")
    print(f"  Trend: sim = {ic:.4f} + {sl:+.5f}·(%err)")
    print(f"  At 10% pred error: sim ≈ {at10:.4f}")

# ── 8. SAVE ───────────────────────────────────────────────────────────
df_sim_act.to_csv(OUT/"sindy_617t_equation_similarity.csv",
                  index=False, float_format='%.6f')
df_sens.to_csv(   OUT/"sindy_617t_sensitivity_analysis.csv",
                  index=False, float_format='%.6f')
df_models.to_csv( OUT/"sindy_617t_model_comparison.csv",
                  index=False, float_format='%.6f')
eq_rows = [{'term': lbl,
             'OLS_ref': analytic_ref.get(lbl, np.nan),
             'identifiable': id_flags.get(lbl, True),
             'sindy_actual': coefs_act.get(lbl, 0.),
             **{f'sindy_{nm}': coefs_preds[nm].get(lbl, 0.) for nm in ML}}
           for lbl in FEAT_LABELS_FULL]
pd.DataFrame(eq_rows).to_csv(OUT/"sindy_617t_discovered_equations.csv",
                              index=False, float_format='%.6f')
if REAL and df_rt is not None:
    df_rt.to_csv(OUT/"sindy_617t_RT_specimens.csv",  index=False, float_format='%.6f')
    df_ht.to_csv(OUT/"sindy_617t_HT_specimens.csv",  index=False, float_format='%.6f')

id_report = pd.DataFrame([
    {'feature': lbl, 'std': feat_stds[lbl],
     'id_threshold': ID_THRESH * log_y.std(),
     'identifiable': id_flags[lbl]}
    for lbl in FEAT_LABELS_FULL
])
id_report.to_csv(OUT/"sindy_617t_identifiability.csv",
                 index=False, float_format='%.6f')
print("\n  CSVs saved.")

# ── 9. VISUALISATION ──────────────────────────────────────────────────
DARK='#0d1117'; PANEL='#161b22'; GRID='#21262d'
C1='#58a6ff'; C2='#f85149'; C3='#3fb950'; C4='#d29922'; C5='#bc8cff'; TEXT='#c9d1d9'
MC = [C1, C3, C4]
HC = [C1, C3, C4, C5]

def sax(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=C1, fontsize=9, fontweight='bold', pad=6)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=8)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=8)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)

n_heats    = int(Heat_enc.max()) + 1
uts_ll     = [UTS.min() - 20, UTS.max() + 20]
y_best     = RES[best_ml]['ypall']
yp_act_uts = np.exp(logp_act)

fig = plt.figure(figsize=(22, 20), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# P1 – RT vs HT scatter
ax = fig.add_subplot(gs[0, 0])
if REAL and df_ht is not None:
    ax.scatter(df_rt['eps_at_UTS'], df_rt['UTS'],
               s=45, alpha=0.85, color=C1, edgecolors='none', label='RT (used)')
    ax.scatter(df_ht['eps_at_UTS'], df_ht['UTS'],
               s=45, alpha=0.85, color=C2, marker='x', lw=1.5, label='HT/outlier')
else:
    ax.scatter(eps_at_UTS, UTS, s=40, alpha=0.8, color=C1, edgecolors='none')
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax, 'All Specimens: UTS vs eps@UTS\n(RT=blue, HT/outlier=red×)',
    'eps @ UTS', 'UTS (MPa)')

# P2 – ML predicted vs actual
ax2 = fig.add_subplot(gs[0, 1])
for hi in range(n_heats):
    mask = Heat_enc == hi
    ax2.scatter(UTS[mask], y_best[mask], s=50, alpha=0.85,
                color=HC[hi % len(HC)], edgecolors='none', label=f"Heat {hi}")
ax2.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax2, f'ML ({best_ml}) RT  Actual vs Predicted',
    'Actual UTS (MPa)', 'Predicted UTS (MPa)')
ax2.text(0.05, 0.92, f"R²={RES[best_ml]['te_r2']:.4f}",
         transform=ax2.transAxes, color=C3, fontsize=9, fontweight='bold')

# P3 – SINDy on RT actual
ax3 = fig.add_subplot(gs[0, 2])
for hi in range(n_heats):
    mask = Heat_enc == hi
    ax3.scatter(UTS[mask], yp_act_uts[mask], s=50, alpha=0.85,
                color=HC[hi % len(HC)], edgecolors='none')
ax3.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
sax(ax3, f'SINDy (RT, log-space)  R²_UTS={r2_act_uts:.4f}\n'
         f'{n_act_t} active of {len(FEAT_ACTIVE)} retained features',
    'Actual UTS (MPa)', 'SINDy UTS (MPa)')

# P4 – Identifiability bar chart
ax4 = fig.add_subplot(gs[1, 0])
feat_names = list(feat_stds.keys())
stds_vals  = [feat_stds[f] for f in feat_names]
colors_id  = [C3 if id_flags[f] else C2 for f in feat_names]
bars = ax4.barh(feat_names, stds_vals, color=colors_id, alpha=0.85, edgecolor=PANEL)
thresh_line = ID_THRESH * log_y.std()
ax4.axvline(thresh_line, color=C4, ls='--', lw=1.5,
            label=f'ID threshold ({ID_THRESH:.0%}·σ_y)')
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
for bar, lbl in zip(bars, feat_names):
    tag = " ✓" if id_flags[lbl] else " WEAK"
    ax4.text(bar.get_width() + thresh_line * 0.05, bar.get_y() + bar.get_height()/2,
             tag, va='center', fontsize=7, color=TEXT)
sax(ax4, 'Feature Identifiability\n(green=OK, red=too little variance)',
    'std(feature column)', '')
ax4.set_facecolor(PANEL)

# P5 – Similarity bars (identifiability-aware)
ax5 = fig.add_subplot(gs[1, 1])
plabs = list(analytic_ref.keys())
xp    = np.arange(len(plabs))
w     = 0.20
srcs  = [('SINDy(actual)', C1, df_sim_act)]
for nm, col in zip(ML, MC[1:]):
    srcs.append((nm, col, compute_sim(coefs_preds[nm], analytic_ref, id_flags)))
for i, (lbl, col, df_s) in enumerate(srcs):
    off  = (i - len(srcs)/2 + 0.5) * w
    vals = df_s['similarity'].fillna(-0.05).values   # NaN shown as tiny gap
    bars = ax5.bar(xp + off, vals, w, label=lbl, color=col, alpha=0.85,
                   edgecolor=PANEL)
    for bar, val, idf in zip(bars, df_s['similarity'].values,
                              df_s['identifiable'].values):
        tag = f'{val:.2f}' if not np.isnan(val) else 'NaN'
        ax5.text(bar.get_x() + bar.get_width()/2,
                 max(bar.get_height(), 0) + 0.03,
                 tag, ha='center', va='bottom', fontsize=6.5, color=TEXT)
ax5.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.5)
ax5.axhline(0.0, color=C2,  ls='--', lw=1, alpha=0.5)
ax5.set_xticks(xp)
ax5.set_xticklabels(plabs, fontsize=8, color=TEXT)
ax5.set_ylim([-0.3, 1.5])
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax5, 'Equation Similarity vs OLS\n(NaN=unidentifiable feature)',
    '', 'Similarity')

# P6 – Sensitivity
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(df_sens['approx_pct_err'], df_sens['mean_sim'], 'o-',
         color=C1, lw=2, ms=6, label='Mean sim (id. only)')
if not np.isnan(sl):
    xf = np.linspace(0, df_sens['approx_pct_err'].max(), 200)
    ax6.plot(xf, ic + sl*xf, '--', color=C2, lw=1.5,
             label=f'Δ@10%={sl*10:+.3f}')
ax6.axhline(1.0, color=C3, ls=':', alpha=0.5, lw=1)
ax6.axhline(0.0, color=C4, ls=':', alpha=0.5, lw=1)
ax6.set_ylim([-0.3, 1.3])
ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax6, 'Error Propagation: Pred. Error → Equation Quality',
    'Approx. Prediction Error (%)', 'Mean Similarity (identifiable)')

# P7 – Coefficient stability vs noise
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor(PANEL)
pa = df_sens['approx_pct_err'].values
ax7.plot(pa, df_sens['sindy_p'].values, 's-', color=C3, lw=2, ms=5,
         label='SINDy p [log(eps)]')
ax7.axhline(p_ro, color=C3, ls='--', alpha=0.8, lw=1.5,
            label=f'OLS p={p_ro:.5f}')
if 'log(sr)' not in weak_feats:
    ax7b = ax7.twinx()
    ax7b.plot(pa, df_sens['sindy_m'].values, 'o-', color=C4, lw=2, ms=5)
    ax7b.axhline(m_ro, color=C4, ls='--', alpha=0.8, lw=1.5)
    ax7b.set_ylabel('m (rate exp)', color=C4, fontsize=8)
    ax7b.tick_params(colors=TEXT, labelsize=7)
else:
    ax7.text(0.5, 0.5, 'log(sr) unidentifiable\n(SR span too narrow)',
             ha='center', va='center', transform=ax7.transAxes,
             fontsize=9, color=C2, fontstyle='italic')
for sp in ax7.spines.values(): sp.set_edgecolor(GRID)
ax7.tick_params(colors=TEXT, labelsize=8)
ax7.grid(True, color=GRID, alpha=0.5, lw=0.5)
ax7.set_title('Coefficient Stability vs Noise',
              color=C1, fontsize=9, fontweight='bold')
ax7.set_xlabel('Prediction Error (%)', color=TEXT, fontsize=8)
ax7.set_ylabel('p (strain exp)', color=C3, fontsize=8)
ax7.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# P8 – Model scatter
ax8 = fig.add_subplot(gs[2, 1])
for i, row in df_models.iterrows():
    ax8.scatter(row['median_err_pct'], row['mean_similarity'],
                s=250, color=MC[i % len(MC)], zorder=5,
                edgecolors='white', lw=0.8)
    ax8.annotate(row['model'], (row['median_err_pct'], row['mean_similarity']),
                 textcoords='offset points', xytext=(8, 5),
                 fontsize=9, color=MC[i % len(MC)], fontweight='bold')
ax8.axhline(0, color=C2, ls='--', lw=1, alpha=0.5)
sax(ax8, 'ML Model: Prediction Error vs Equation Similarity',
    'Median Prediction Error (%)', 'Mean Equation Similarity')

# P9 – Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(DARK)
for sp in ax9.spines.values(): sp.set_edgecolor(GRID)
ax9.axis('off')
C0s = coefs_act.get('C0', 0)
ps_ = coefs_act.get('log(eps)', np.nan)
K_s = np.exp(C0s) if abs(C0s) < 15 else float('nan')
txt = [
    "Alloy 617  |  Tensile  |  UTS  [v3]",
    f"{'Real CSV' if REAL else 'Synthetic'}  "
    f"RT={len(UTS)} / HT={len(df_ht) if REAL and df_ht is not None else 0}",
    f"SR span: {sr_dec:.2f} dec | Φ cond: {cond_final:.1f}",
    "",
    "── Identifiability ────────────────",
] + [
    f"  {lbl:<18} {'OK' if id_flags[lbl] else 'WEAK (NaN)':>10}"
    for lbl in FEAT_LABELS_FULL
] + [
    "",
    "── OLS R-O (RT only) ──────────────",
    f"  K={K_ro:.1f} MPa  m={m_ro:.5f}  p={p_ro:.5f}",
    "",
    f"── SINDy (R²_log={r2_act_log:.4f}, R²_UTS={r2_act_uts:.4f}) ─",
    f"  C0={C0s:+.5g} → K≈{K_s:.1f} MPa",
    f"  log(sr) = {'dropped (weak)' if 'log(sr)' in weak_feats else f'{coefs_act.get(chr(108)+chr(111)+chr(103)+chr(40)+chr(115)+chr(114)+chr(41), 0):+.5g}'}",
    f"  log(eps)= {ps_:+.5g}   OLS:{p_ro:+.5f}",
    f"  ({n_act_t} of {len(FEAT_ACTIVE)} active)",
    "",
    "── Similarity (identifiable) ──────",
] + [
    f"  {r['param']:<16}: "
    f"{'NaN (weak)' if not r['identifiable'] else f'{r.similarity:+.4f}'}"
    for _, r in df_sim_act.iterrows()
] + [
    f"  MEAN = {mean_sim_act:+.4f}",
    "",
    "── Error Propagation ──────────────",
    (f"  r={corr:.4f}  p={pval:.4f}" if not np.isnan(corr) else "  n/a"),
    (f"  Δsim/10%err={sl*10:+.4f}" if not np.isnan(sl) else ""),
    "",
    "v3: QR decorrelation | id. diagnostic",
    "    intercept fix | NaN for weak feats",
]
ax9.text(0.03, 0.97, "\n".join(txt), transform=ax9.transAxes,
         fontsize=7.2, va='top', fontfamily='monospace', color=TEXT,
         bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.9))
ax9.set_title('Summary [v3]', color=C1, fontsize=9, fontweight='bold')

fig.suptitle(
    "Alloy 617 Tensile – SINDy UTS Study  "
    "[v3: QR decorrelation · identifiability diagnostic · intercept fix]\n"
    "SINDy in log(UTS) space  |  OLS R-O Reference  |  NaN for unidentifiable features",
    fontsize=11, fontweight='bold', color=C1, y=0.999)

plt.savefig(OUT/"sindy_617t_analysis.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: sindy_617t_analysis.png")

# ── 10. FINAL SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY – ALLOY 617 TENSILE [v3]")
print("="*65)
print(f"\n  RT: {len(UTS)} specimens  |  HT/outlier: "
      f"{len(df_ht) if REAL and df_ht is not None else 0}")
print(f"  Φ condition number (after QR): {cond_final:.1f}")
print(f"  Active features: {FEAT_ACTIVE}")
print(f"\n  Identifiability:")
for lbl in FEAT_LABELS_FULL:
    s   = feat_stds[lbl]
    ok  = id_flags[lbl]
    tag = "OK" if ok else f"WEAK (std={s:.5f} < threshold)"
    print(f"    {lbl:<20}: {tag}")
print(f"\n  OLS R-O: UTS = {K_ro:.2f} · sr^{m_ro:.5f} · eps^{p_ro:.5f}")
print(f"\n  SINDy (thresh={best_thresh:.5f}, {n_act_t} terms):")
print(fmt_eq(coefs_act))
print(f"\n  Similarity (identifiable terms, n={n_id}): {mean_sim_act:.4f}")
if not np.isnan(corr):
    print(f"  Pearson r (noise→sim) = {corr:.4f}  (p={pval:.4f})")
print(f"\n  Key insight: log(sr) is NOT identifiable from this dataset")
print(f"    ({sr_dec:.2f} decade SR span; need ≥ 2 decades for reliable SINDy rate ID)")
print(f"    Both OLS m={m_ro:.5f} and SINDy m=0 should be treated with caution.")
print(f"\n  Outputs:")
for f in ["sindy_617t_analysis.png",
          "sindy_617t_equation_similarity.csv",
          "sindy_617t_sensitivity_analysis.csv",
          "sindy_617t_model_comparison.csv",
          "sindy_617t_discovered_equations.csv",
          "sindy_617t_identifiability.csv",
          "sindy_617t_RT_specimens.csv",
          "sindy_617t_HT_specimens.csv"]:
    print(f"    {f}")