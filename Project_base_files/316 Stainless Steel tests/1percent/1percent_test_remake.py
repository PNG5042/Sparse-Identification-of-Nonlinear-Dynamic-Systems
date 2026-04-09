# ================================
# SS316H CREEP – SINDy APPLICABILITY STUDY  (PySINDy edition) -- v5
# ================================
# FIXES vs v2 (cumulative):
#
#   FIX-V3-A  Feature-relative identifiability threshold (CV>0.01 or std>1e-4).
#   FIX-V3-B  Protected set {C0, log(s), 1/T, H_c} never dropped by QR.
#   FIX-V3-C  Condition target raised 50 → 500.
#   FIX-V3-D/E OLS-relative similarity with near-zero guard.
#   FIX-V3-F  Threshold tuning r2 floor lowered to 0.50.
#   FIX-V4    not_in_library → NaN similarity (library gap ≠ SINDy failure).
#   FIX-V5    Effect-size similarity guard: |coef|·std(feat)/std(y) < 5% → NaN.
#             Replaces raw |OLS|<0.01 check. C0 (intercept) is exempt.
# ================================

import numpy as np
import pandas as pd
import pysindy as ps
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
from scipy.linalg import qr as scipy_qr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
OUT  = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

# ── 0. DATA ───────────────────────────────────────────────────────────
print("="*65)
print("SS316H CREEP – SINDy APPLICABILITY STUDY  [v5]")
print("="*65)

csv_path = BASE / "SS316H-1percent.csv"
REAL = csv_path.exists()

if REAL:
    df = pd.read_csv(csv_path)
    if df["Heat"].dtype == object:
        hmap = {h: i for i, h in enumerate(df["Heat"].unique())}
        df["Heat_encoded"] = df["Heat"].map(hmap)
    elif "Heat_encoded" not in df.columns:
        df["Heat_encoded"] = df["Heat"].astype(float)
    Heat   = df["Heat_encoded"].values.astype(float)
    Temp   = df["Temp (K)"].values.astype(float)
    Stress = df["Stress (Mpa)"].values.astype(float)
    Time   = df["Time (h) to 1% strain"].values.astype(float)
    print(f"  Loaded real data: {len(df)} rows")
else:
    print("  CSV not found – generating synthetic SS316H data")
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
    log_t_true = C_true - n_true * np.log(Stress) + QR_true / Temp
    Time = np.exp(log_t_true + rng.normal(0, 0.25, n))
    print(f"  True: log(t) = {C_true} - {n_true}*log(sigma) + {QR_true:.0f}/T")

# Centred heat encoding
H_mean = Heat.mean()
H_c    = Heat - H_mean

y_log = np.log(np.clip(Time, 1e-12, None))
print(f"\n  N={len(Time)},  log(t): {y_log.min():.2f} – {y_log.max():.2f}")
print(f"  Temp:   {Temp.min():.0f} – {Temp.max():.0f} K")
print(f"  Stress: {Stress.min():.0f} – {Stress.max():.0f} MPa")
print(f"  H_c range: [{H_c.min():.2f}, {H_c.max():.2f}]")

# ── 1. ML PREDICTION MODEL ────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: ML PREDICTION MODEL  (target = log(t))")
print("="*65)

def ml_feats(Hc, T, S):
    return np.column_stack([
        np.ones_like(Hc), Hc, Hc**2,
        T, 1/T, T**2, 1/T**2, np.log(T), T**-0.5,
        S, np.log(S), S**2, S**3, 1/S, 1/S**2, 1/S**3, S**0.5,
        S/T, np.log(S)/T, T*np.log(S), np.log(T)*np.log(S),
        S/T**2, Hc/T, Hc*np.log(S), Hc*S, 1/(S*T),
    ])

X_all  = ml_feats(H_c, Temp, Stress)
n_feat = min(15, X_all.shape[1])
sel    = SelectKBest(f_regression, k=n_feat)
X_sel  = sel.fit_transform(X_all, y_log)
sc_ml  = StandardScaler()
X_sc   = sc_ml.fit_transform(X_sel)

n_samp = len(y_log)
bins   = np.digitize(y_log, np.percentile(y_log, [20, 40, 60, 80]))
if n_samp >= 20:
    sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, te_i = next(sss.split(X_sc, bins))
else:
    split = max(1, int(n_samp * 0.75))
    tr_i, te_i = np.arange(split), np.arange(split, n_samp)

Xtr, Xte = X_sc[tr_i], X_sc[te_i]
ytr, yte  = y_log[tr_i], y_log[te_i]

n_cv = min(5, max(2, len(tr_i) // 10))
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
    err = np.abs((np.exp(yp) - np.exp(yte)) / np.exp(yte)) * 100
    RES[nm] = dict(tr_r2=m.score(Xtr, ytr), te_r2=m.score(Xte, yte),
                   med_err=np.median(err), err=err, ypall=m.predict(X_sc))
    print(f"  {nm:<14} TrainR²={RES[nm]['tr_r2']:.4f}  "
          f"TestR²={RES[nm]['te_r2']:.4f}  "
          f"MedianErr={RES[nm]['med_err']:.1f}%")

best_ml = max(RES, key=lambda k: RES[k]['te_r2'])
print(f"\n  Best: {best_ml}  (R²={RES[best_ml]['te_r2']:.4f}, "
      f"err={RES[best_ml]['med_err']:.1f}%)")

# OLS reference includes heat encoding
A_nb     = np.column_stack([np.ones_like(Temp), np.log(Stress), 1/Temp, H_c])
nb_c, *_ = np.linalg.lstsq(A_nb, y_log, rcond=None)
C_nb, n_nb, QR_nb, h_nb = nb_c
print(f"\n  Norton-Bailey OLS (heat-corrected):")
print(f"    log(t) = {C_nb:.4f} + {n_nb:.4f}·log(σ) + {QR_nb:.2f}/T + {h_nb:.4f}·H_c")

# ── 2. IDENTIFIABILITY DIAGNOSTIC ─────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: FEATURE IDENTIFIABILITY DIAGNOSTIC")
print("="*65)

FEAT_LABELS_FULL = ['C0', '1/T', 'log(s)', 'log(s)^2',
                    '1/s', '(1/T)log(s)', 'log(T)', 'Garofalo', 'H_c']

def build_phi_full(T, S, Hc):
    sinh_arg = np.clip(0.01 * S, 1e-9, 500)
    return np.column_stack([
        np.ones(len(T)),
        1.0 / T,
        np.log(S),
        np.log(S)**2,
        1.0 / S,
        (1.0 / T) * np.log(S),
        np.log(T),
        np.log(np.sinh(sinh_arg)),
        Hc,
    ])

Phi_full = build_phi_full(Temp, Stress, H_c)

# FIX-V3-A: feature-relative identifiability.
# A feature is identifiable if its CV > 0.01  OR  raw std > 1e-4.
# This correctly handles 1/T whose absolute values are tiny but vary
# meaningfully relative to their own scale.
CV_THRESH  = 0.01
STD_THRESH = 1e-4

print(f"\n  Identifiability criteria: CV > {CV_THRESH} OR std > {STD_THRESH}\n")
print(f"  {'Feature':<22} {'std(col)':>10} {'|mean|':>10} {'CV':>8} {'identifiable?':>15}")
print("  " + "-"*67)

id_flags  = {}
feat_stds = {}

for i, lbl in enumerate(FEAT_LABELS_FULL):
    col = Phi_full[:, i]
    s   = col.std()
    feat_stds[lbl] = s
    if lbl == 'C0':
        id_flags[lbl] = True
        print(f"  {'C0':<22} {'(intercept)':>10} {'—':>10} {'—':>8} {'YES (fixed)':>15}")
        continue
    mu  = abs(col.mean())
    cv  = s / mu if mu > 1e-15 else np.nan
    # FIX-V3-A: use CV threshold, with raw std fallback
    if not np.isnan(cv):
        ok = (cv > CV_THRESH) or (s > STD_THRESH)
    else:
        ok = s > STD_THRESH
    id_flags[lbl] = ok
    cv_str = f"{cv:.4f}" if not np.isnan(cv) else "n/a"
    flag   = "YES" if ok else "*** WEAK ***"
    print(f"  {lbl:<22} {s:>10.5f} {mu:>10.5f} {cv_str:>8} {flag:>15}")

weak_feats = [k for k, v in id_flags.items() if not v]
print(f"\n  Weakly identifiable features: "
      f"{weak_feats if weak_feats else 'none'}")

# ── 3. QR DECORRELATION ───────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 3: QR DECORRELATION  (select well-conditioned column subset)")
print("="*65)

# FIX-V3-B: protect core Norton-Bailey terms + H_c from being dropped.
# H_c is identifiable (std=3.37) and physically meaningful (heat-to-heat
# scatter) so it must survive QR pruning even if collinear with others.
# FIX-V3-C: raise condition target to 500 to retain more features.
PROTECTED   = {'C0', 'log(s)', '1/T', 'H_c'}
COND_TARGET = 500.0

id_cols   = [i for i, lbl in enumerate(FEAT_LABELS_FULL) if id_flags[lbl]]
id_labels = [FEAT_LABELS_FULL[i] for i in id_cols]
Phi_id    = Phi_full[:, id_cols]

col_norms = np.linalg.norm(Phi_id, axis=0, keepdims=True)
col_norms[col_norms == 0] = 1.0
Phi_normed = Phi_id / col_norms

_, _, piv = scipy_qr(Phi_normed, pivoting=True)

print("\n  QR pivot order (most → least important):")
print([id_labels[i] for i in piv])
print(f"  Protected features (never dropped): {sorted(PROTECTED)}")
print(f"  Condition target: {COND_TARGET}")

cond_prev      = np.linalg.cond(Phi_id)
keep_idx       = list(range(len(id_labels)))
piv_order      = list(piv)   # least important first (we scan reversed)
dropped_labels = []

# Reverse pivot order = least important last in QR → drop last first
for candidate in reversed(piv_order):
    if len(keep_idx) <= 3:
        break
    cond_now = np.linalg.cond(Phi_id[:, keep_idx])
    if cond_now <= COND_TARGET:
        break
    lbl = id_labels[candidate]
    if lbl in PROTECTED:
        continue   # FIX-V3-B: never drop protected features
    if candidate in keep_idx:
        keep_idx.remove(candidate)
        dropped_labels.append(lbl)

Phi_final   = Phi_id[:, keep_idx]
FEAT_ACTIVE = [id_labels[i] for i in keep_idx]
cond_final  = np.linalg.cond(Phi_final)

print(f"\n  Before QR pruning : {len(id_labels)} cols, cond = {cond_prev:.1f}")
print(f"  After  QR pruning : {len(FEAT_ACTIVE)} cols, cond = {cond_final:.1f}")
print(f"  Kept   : {FEAT_ACTIVE}")
print(f"  Dropped: {dropped_labels if dropped_labels else 'none'}")
if cond_final > COND_TARGET:
    print(f"  *** NOTE: cond still {cond_final:.1f} — "
          f"inherent collinearity; all remaining features are protected ***")
else:
    print(f"  Condition number OK (≤ {COND_TARGET:.0f}).")

# ── 4. SINDy ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 4: SINDy – SPARSE EQUATION DISCOVERY  (log(t) space)")
print("="*65)

def fit_sindy(y_target, threshold, Phi, feat_labels):
    """Fit STLSQ on Phi → y_target. Returns coef_dict, r2, y_pred."""
    opt = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=2000)
    opt.fit(Phi, y_target)
    coeffs = opt.coef_.ravel()
    y_pred = Phi @ coeffs

    ss_r = np.sum((y_target - y_pred)**2)
    ss_t = np.sum((y_target - y_target.mean())**2)
    r2   = 1 - ss_r / ss_t if ss_t else 0.0

    coef_dict = {lbl: float(v)
                 for lbl, v in zip(feat_labels, coeffs) if abs(v) > 1e-10}
    return coef_dict, r2, y_pred

# FIX-V3-F: lower r2 floor to 0.50 so tuner can succeed on noisy real data.
def tune_threshold(y_target, Phi, feat_labels, thresholds, r2_floor=0.50):
    best_thresh = thresholds[len(thresholds)//4]
    best_score  = -np.inf
    n_full      = len(feat_labels)
    for thresh in thresholds:
        try:
            cd, r2, _ = fit_sindy(y_target, thresh, Phi, feat_labels)
            nact = len(cd)
            if r2 < r2_floor or nact < 2 or nact >= n_full:
                continue
            score = r2 + max(0, n_full - nact) * 0.02
            if score > best_score:
                best_score, best_thresh = score, thresh
        except Exception:
            continue
    return best_thresh

thresholds = np.logspace(-3, 2, 200)

print("\n  Tuning STLSQ threshold:")
best_thresh = tune_threshold(y_log, Phi_final, FEAT_ACTIVE, thresholds)
print(f"  Best threshold = {best_thresh:.5f}")

coefs_act, r2_act, yp_act = fit_sindy(y_log, best_thresh, Phi_final, FEAT_ACTIVE)
n_act_t = len(coefs_act)
print(f"\n  SINDy (actual):  R²={r2_act:.4f}  "
      f"{n_act_t} active: {list(coefs_act.keys())}")

coefs_preds = {}
r2_preds    = {}
yp_sindy_ml = {}
for nm in ML:
    cd, r2v, yhat = fit_sindy(RES[nm]['ypall'], best_thresh, Phi_final, FEAT_ACTIVE)
    coefs_preds[nm] = cd
    r2_preds[nm]    = r2v
    yp_sindy_ml[nm] = yhat
    print(f"  SINDy ({nm}): R²={r2v:.4f}  "
          f"{len(cd)} active: {list(cd.keys())}")

# ── 5. EQUATIONS ──────────────────────────────────────────────────────
def fmt_eq(coef_dict, ylbl='log(t)'):
    lines = []
    if 'C0' in coef_dict:
        v = coef_dict['C0']
        lines.append(f"  {v:+.5g}  [intercept  |  OLS C0={C_nb:.4f}]")
    for k, v in coef_dict.items():
        if k == 'C0':
            continue
        ann = ""
        if k == 'log(s)':  ann = f"   ← OLS n={n_nb:+.4f}"
        if k == '1/T':     ann = f"   ← OLS Q/R={QR_nb:+.2f}"
        if k == 'H_c':     ann = f"   ← OLS h={h_nb:+.4f}"
        lines.append(f"  {v:+.5g} · {k}{ann}")
    if not lines:
        return f"  {ylbl} = 0  (no active terms)"
    return f"  {ylbl} =\n" + "\n".join(lines)

print("\n" + "="*65)
print("STEP 5: DISCOVERED EQUATIONS  (log(t) space)")
print("="*65)
print(f"\n  -- OLS Norton-Bailey reference (heat-corrected) --")
print(f"  log(t) = {C_nb:+.5g} + {n_nb:+.5g}·log(σ) + {QR_nb:+.5g}/T + {h_nb:+.5g}·H_c")
print(f"\n  -- SINDy on ACTUAL  (R²={r2_act:.4f}) --")
print(fmt_eq(coefs_act))
for nm in ML:
    print(f"\n  -- SINDy on {nm}  "
          f"(ML err={RES[nm]['med_err']:.1f}%, R²={r2_preds[nm]:.4f}) --")
    print(fmt_eq(coefs_preds[nm]))

# ── 6. SIMILARITY ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 6: EQUATION SIMILARITY  (identifiability-aware)")
print("="*65)

analytic_ref = {
    'C0':     C_nb,
    'log(s)': n_nb,
    '1/T':    QR_nb,
    'H_c':    h_nb,
}

def compute_sim(sindy_coefs, ref, id_flags_map, active_feats=None,
                feat_stds_map=None, y_std=None):
    """
    active_feats   : features present in Phi_final (None = all).
    feat_stds_map  : std of each feature column (for effect-size check).
    y_std          : std of the target y (for effect-size normalisation).

    Similarity is NaN when:
      - feature not identifiable
      - feature not in Phi_final (library gap, not SINDy failure)
      - |OLS_coef| * std(feature) < 5% * std(y)  ← effect too small to
        distinguish from noise; raw-value guard replaced by effect-size guard.
        Example: h_nb=-0.0105 with std(H_c)=3.37 → effect=0.035 vs
        y_std≈2.3 → 1.5% → NaN, correctly treated as not assessable.

    If the feature IS in the library and the effect is large enough,
    similarity = clip(1 - |rel_err|, -1, 1).
    """
    EFF_FRAC = 0.05   # 5% of y_std = minimum detectable effect
    rows = []
    for param, ana in ref.items():
        identifiable = id_flags_map.get(param, True)
        in_library   = (active_feats is None) or (param in active_feats)
        sval         = sindy_coefs.get(param, 0.0)

        if not identifiable:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': False, 'in_library': in_library,
                         'note': 'not_identifiable'})
            continue

        if not in_library:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': np.nan,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': True, 'in_library': False,
                         'note': 'not_in_library'})
            continue

        # Effect-size guard: skip if the feature's contribution to y is
        # too small to be reliably estimated from noisy data.
        # The intercept C0 is exempt — it's scored via absolute error, not effect.
        feat_std = feat_stds_map.get(param, 1.0) if feat_stds_map else 1.0
        y_s      = y_std if y_std else 1.0
        effect   = abs(ana) * feat_std          # units of y
        if param != 'C0' and y_s > 0 and (effect / y_s) < EFF_FRAC:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': True, 'in_library': True,
                         'note': f'effect<{EFF_FRAC:.0%}·σ_y'})
            continue

        if abs(ana) < 1e-10:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': True, 'in_library': True,
                         'note': 'OLS≈0'})
            continue

        rel_err = (ana - sval) / abs(ana)
        sim     = float(np.clip(1.0 - abs(rel_err), -1.0, 1.0))
        rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                     'rel_err': rel_err, 'similarity': sim,
                     'identifiable': True, 'in_library': True,
                     'note': 'ok'})
    return pd.DataFrame(rows)

SIM_KWARGS = dict(feat_stds_map=feat_stds, y_std=y_log.std())

df_sim_act = compute_sim(coefs_act, analytic_ref, id_flags, FEAT_ACTIVE, **SIM_KWARGS)
print(f"\n  {'Param':<18} {'OLS':>12} {'SINDy':>12} "
      f"{'rel_err':>9} {'sim':>7} {'id?':>8} {'note':>20}")
print("  " + "-"*86)
for _, r in df_sim_act.iterrows():
    re  = f"{r['rel_err']:.4f}" if not np.isnan(r['rel_err'])   else "n/a"
    si  = f"{r['similarity']:.4f}" if not np.isnan(r['similarity']) else "NaN"
    idf = "YES" if r['identifiable'] else "WEAK"
    sv_str = f"{r['sindy']:>12.4f}" if not np.isnan(r['sindy']) else f"{'—':>12}"
    print(f"  {r['param']:<18} {r['OLS_ref']:>12.4f} {sv_str} "
          f"{re:>9} {si:>7} {idf:>8} {r['note']:>20}")

mean_sim_act = df_sim_act['similarity'].dropna().mean()
n_id         = int(df_sim_act['identifiable'].sum())
n_scored     = int((df_sim_act['note'] == 'ok').sum())
print(f"\n  Mean similarity (scored terms, n={n_scored}): {mean_sim_act:.4f}")
print(f"  1.0=exact | 0.0=100% off | <0=wrong sign | NaN=not assessable")

model_rows = []
for nm in ML:
    dfs  = compute_sim(coefs_preds[nm], analytic_ref, id_flags, FEAT_ACTIVE, **SIM_KWARGS)
    msim = dfs['similarity'].dropna().mean()
    model_rows.append({
        'model': nm, 'test_r2': RES[nm]['te_r2'],
        'median_err_pct': RES[nm]['med_err'],
        **{f"sim_{r['param']}": r['similarity'] for _, r in dfs.iterrows()},
        'mean_similarity': msim,
    })
    print(f"  {nm:<14} ML err={RES[nm]['med_err']:.1f}%  mean_sim={msim:.4f}")
df_models = pd.DataFrame(model_rows)

# ── 7. SENSITIVITY ────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 7: SENSITIVITY – Noise → SINDy equation quality")
print("="*65)

noise_stds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]
rng2 = np.random.default_rng(7)
sens_rows = []

for s in noise_stds:
    y_n = y_log + rng2.normal(0, s, len(y_log))
    best_t_s = tune_threshold(y_n, Phi_final, FEAT_ACTIVE, thresholds,
                              r2_floor=0.45)
    cn, r2v, _ = fit_sindy(y_n, best_t_s, Phi_final, FEAT_ACTIVE)

    dfs   = compute_sim(cn, analytic_ref, id_flags, FEAT_ACTIVE, **SIM_KWARGS)
    msim  = dfs['similarity'].dropna().mean()
    pct   = 100 * (np.exp(s) - 1)
    nval  = cn.get('log(s)', 0)
    qrval = cn.get('1/T',    0)
    sens_rows.append({'noise_std': s, 'approx_pct_err': pct,
                      'n_active': len(cn), 'mean_sim': msim,
                      'r2': r2v, 'sindy_n': nval, 'sindy_QR': qrval,
                      'thresh_used': best_t_s})
    print(f"  s={s:.2f} (~{pct:>6.0f}% err)  thresh={best_t_s:.5f}  "
          f"active={len(cn):2d}  R²={r2v:.4f}  sim={msim:.4f}  "
          f"n={nval:.4f}  Q/R={qrval:.2f}")

df_sens = pd.DataFrame(sens_rows)
pv = df_sens['approx_pct_err'].values
sv = df_sens['mean_sim'].values

corr = pval = sl = ic = at85 = np.nan
if len(pv) > 3 and not np.all(np.isnan(sv)):
    valid = ~np.isnan(sv)
    if valid.sum() > 3:
        corr, pval = stats.pearsonr(pv[valid], sv[valid])
        sl, ic, *_ = stats.linregress(pv[valid], sv[valid])
        at85 = ic + sl * 85
        print(f"\n  Pearson r = {corr:.4f}  (p={pval:.4f})")
        print(f"  Trend: sim = {ic:.4f} + {sl:+.6f}·(%err)")
        print(f"  At 85% pred error: sim ≈ {at85:.4f}")

# ── 8. SAVE ───────────────────────────────────────────────────────────
df_sim_act.to_csv(OUT/"sindy_equation_similarity.csv",
                  index=False, float_format='%.6f')
df_sens.to_csv(   OUT/"sindy_sensitivity_analysis.csv",
                  index=False, float_format='%.6f')
df_models.to_csv( OUT/"sindy_model_comparison.csv",
                  index=False, float_format='%.6f')
eq_rows = [{'term': lbl,
             'OLS_ref': analytic_ref.get(lbl, np.nan),
             'identifiable': id_flags.get(lbl, True),
             'sindy_actual': coefs_act.get(lbl, 0.),
             **{f'sindy_{nm}': coefs_preds[nm].get(lbl, 0.) for nm in ML}}
           for lbl in FEAT_LABELS_FULL]
pd.DataFrame(eq_rows).to_csv(OUT/"sindy_discovered_equations.csv",
                              index=False, float_format='%.6f')

id_report = pd.DataFrame([
    {'feature': lbl, 'std': feat_stds[lbl],
     'cv_thresh': CV_THRESH, 'std_thresh': STD_THRESH,
     'identifiable': id_flags[lbl]}
    for lbl in FEAT_LABELS_FULL
])
id_report.to_csv(OUT/"sindy_identifiability.csv",
                 index=False, float_format='%.6f')
print("\n  CSVs saved.")

# ── 9. VISUALISATION ──────────────────────────────────────────────────
DARK='#0d1117'; PANEL='#161b22'; GRID='#21262d'
C1='#58a6ff'; C2='#f85149'; C3='#3fb950'; C4='#d29922'; C5='#bc8cff'; TEXT='#c9d1d9'
MC = [C1, C3, C4]

def sax(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=C1, fontsize=9, fontweight='bold', pad=6)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=8)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=8)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)

y_best = RES[best_ml]['ypall']
ll     = [y_log.min()-0.5, y_log.max()+0.5]

fig = plt.figure(figsize=(22, 20), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# P1 – ML prediction
ax = fig.add_subplot(gs[0, 0])
sc = ax.scatter(y_log, y_best, s=15, alpha=0.55, c=Temp,
                cmap='plasma', edgecolors='none')
ax.plot(ll, ll, '--', color=C2, lw=1.5)
plt.colorbar(sc, ax=ax, label='Temp (K)').ax.yaxis.label.set_color(TEXT)
sax(ax, f'ML ({best_ml})  Actual vs Predicted',
    'Actual log(t)', 'Predicted log(t)')
ax.text(0.05, 0.92, f"R²={RES[best_ml]['te_r2']:.4f}",
        transform=ax.transAxes, color=C3, fontsize=9, fontweight='bold')

# P2 – SINDy on actual
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_log, yp_act, s=15, alpha=0.55, color=C3, edgecolors='none')
ax2.plot(ll, ll, '--', color=C2, lw=1.5)
sax(ax2, f'SINDy (actual)  R²={r2_act:.4f}\n{n_act_t} active terms',
    'Actual log(t)', 'SINDy log(t)')

# P3 – SINDy on best ML
ax3 = fig.add_subplot(gs[0, 2])
ysp = yp_sindy_ml[best_ml]
ax3.scatter(y_best, ysp, s=15, alpha=0.55, color=C4, edgecolors='none')
ll3 = [y_best.min()-0.5, y_best.max()+0.5]
ax3.plot(ll3, ll3, '--', color=C2, lw=1.5)
sax(ax3, f'SINDy on {best_ml}\nR²={r2_preds[best_ml]:.4f}',
    'ML Predicted log(t)', 'SINDy log(t)')

# P4 – Identifiability bar chart
ax4 = fig.add_subplot(gs[1, 0])
feat_names = list(feat_stds.keys())
stds_vals  = [feat_stds[f] for f in feat_names]
colors_id  = [C3 if id_flags[f] else C2 for f in feat_names]
bars = ax4.barh(feat_names, stds_vals, color=colors_id, alpha=0.85, edgecolor=PANEL)
# Show CV threshold as dashed line at std=1e-4
ax4.axvline(STD_THRESH, color=C4, ls='--', lw=1.5,
            label=f'std fallback threshold ({STD_THRESH})')
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
for bar, lbl in zip(bars, feat_names):
    tag = " ✓" if id_flags[lbl] else " WEAK"
    ax4.text(bar.get_width() * 1.05 + STD_THRESH * 0.1,
             bar.get_y() + bar.get_height()/2,
             tag, va='center', fontsize=7, color=TEXT)
sax(ax4, 'Feature Identifiability\n(green=OK [CV>0.01 or std>1e-4], red=WEAK)',
    'std(feature column)', '')
ax4.set_facecolor(PANEL)

# P5 – Similarity bars
ax5 = fig.add_subplot(gs[1, 1])
plabs = list(analytic_ref.keys())
xp    = np.arange(len(plabs))
w     = 0.20
srcs  = [('SINDy(actual)', C1, df_sim_act)]
for nm, col in zip(ML, MC[1:]):
    srcs.append((nm, col, compute_sim(coefs_preds[nm], analytic_ref, id_flags, FEAT_ACTIVE, **SIM_KWARGS)))
for i, (lbl, col, df_s) in enumerate(srcs):
    off  = (i - len(srcs)/2 + 0.5) * w
    vals = df_s['similarity'].fillna(-0.05).values
    bars = ax5.bar(xp + off, vals, w, label=lbl, color=col, alpha=0.85,
                   edgecolor=PANEL)
    for bar, val in zip(bars, df_s['similarity'].values):
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
sax(ax5, 'Equation Similarity vs OLS\n(NaN=unidentifiable or |OLS|<0.01)',
    '', 'Similarity')

# P6 – Sensitivity curve
ax6 = fig.add_subplot(gs[1, 2])
valid_mask = ~np.isnan(df_sens['mean_sim'])
ax6.plot(df_sens['approx_pct_err'][valid_mask],
         df_sens['mean_sim'][valid_mask], 'o-',
         color=C1, lw=2, ms=6, label='Mean sim (id. only)')
if not np.isnan(sl):
    xf = np.linspace(0, df_sens['approx_pct_err'].max(), 200)
    ax6.plot(xf, ic + sl*xf, '--', color=C2, lw=1.5,
             label=f'Δ@85%={sl*85:+.3f}')
ax6.axhline(1.0, color=C3, ls=':', alpha=0.5, lw=1)
ax6.axhline(0.0, color=C4, ls=':', alpha=0.5, lw=1)
ax6.set_ylim([-0.3, 1.3])
ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax6, 'Error Propagation: Pred. Error → Equation Quality',
    'Approx. Prediction Error (%)', 'Mean Similarity (identifiable)')

# P7 – Coefficient trajectory vs noise
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor(PANEL)
pa = df_sens['approx_pct_err'].values
ax7.plot(pa, df_sens['sindy_n'].values, 's-', color=C3, lw=2, ms=5,
         label='SINDy n [log(σ)]')
ax7.axhline(n_nb, color=C3, ls='--', alpha=0.8, lw=1.5,
            label=f'OLS n={n_nb:.4f}')
ax7b = ax7.twinx()
ax7b.plot(pa, df_sens['sindy_QR'].values, 'o-', color=C4, lw=2, ms=5)
ax7b.axhline(QR_nb, color=C4, ls='--', alpha=0.8, lw=1.5)
ax7b.set_ylabel('Q/R (K)', color=C4, fontsize=8)
ax7b.tick_params(colors=TEXT, labelsize=7)
for sp in ax7.spines.values(): sp.set_edgecolor(GRID)
ax7.tick_params(colors=TEXT, labelsize=8)
ax7.grid(True, color=GRID, alpha=0.5, lw=0.5)
ax7.set_title('Coefficient Stability vs Noise',
              color=C1, fontsize=9, fontweight='bold')
ax7.set_xlabel('Approx. Prediction Error (%)', color=TEXT, fontsize=8)
ax7.set_ylabel('n (stress exp)', color=C3, fontsize=8)
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
C0s  = coefs_act.get('C0',     0)
ns   = coefs_act.get('log(s)', 0)
QRs  = coefs_act.get('1/T',    0)
Hcs  = coefs_act.get('H_c',    0)
txt = [
    "SS316H Creep | 1% strain  [v5]",
    f"{'Real CSV' if REAL else 'Synthetic'}  N={len(y_log)}",
    f"Φ cond (after QR): {cond_final:.1f}",
    "",
    "── Identifiability (CV>0.01 or std>1e-4) ──",
] + [
    f"  {lbl:<18} {'OK' if id_flags[lbl] else 'WEAK (NaN)':>10}"
    for lbl in FEAT_LABELS_FULL
] + [
    "",
    "── OLS Norton-Bailey (heat-corr.) ─",
    f"  C0={C_nb:.4f}  n={n_nb:.4f}",
    f"  Q/R={QR_nb:.2f}  H_c={h_nb:.4f}",
    "",
    f"── SINDy (R²={r2_act:.4f}) ─────────",
    f"  C0={C0s:+.5g}",
    f"  n={ns:+.5g}   OLS:{n_nb:+.4f}",
    f"  Q/R={QRs:+.5g}   OLS:{QR_nb:+.2f}",
    f"  H_c={Hcs:+.5g}",
    f"  ({n_act_t} of {len(FEAT_ACTIVE)} active)",
    "",
    "── Similarity ─────────────────────",
] + [
    f"  {r['param']:<16}: "
    f"{'NaN' if np.isnan(r['similarity']) else f'{r.similarity:+.4f}'}"
    for _, r in df_sim_act.iterrows()
] + [
    f"  MEAN = {mean_sim_act:+.4f}  (interpretable only)",
    "",
    "── Error Propagation ──────────────",
    (f"  r={corr:.4f}  p={pval:.4f}" if not np.isnan(corr) else "  n/a"),
    (f"  Δsim/85%err={sl*85:+.4f}" if not np.isnan(sl) else ""),
    "",
    "v5: effect-size sim guard",
    "  |coef|·std(feat)/std(y)<5%→NaN",
    "  H_c protected, not_in_lib NaN",
]
ax9.text(0.03, 0.97, "\n".join(txt), transform=ax9.transAxes,
         fontsize=7.2, va='top', fontfamily='monospace', color=TEXT,
         bbox=dict(boxstyle='round', facecolor=DARK, alpha=0.9))
ax9.set_title('Summary [v5]', color=C1, fontsize=9, fontweight='bold')

fig.suptitle(
    "SS316H Creep – SINDy Applicability Study  [v5]\n"
    "Protected NB+H_c terms · Feature-relative identifiability · not_in_library NaN",
    fontsize=11, fontweight='bold', color=C1, y=0.999)

plt.savefig(OUT/"sindy_analysis_v5.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: sindy_analysis_v5.png")

# ── 10. FINAL SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY – SS316H CREEP [v5]")
print("="*65)
print(f"\n  {'Real data' if REAL else 'Synthetic'}  ({len(y_log)} samples)")
print(f"  Φ condition number (after QR): {cond_final:.1f}")
print(f"  Active features: {FEAT_ACTIVE}")
print(f"\n  Identifiability (CV > {CV_THRESH} or std > {STD_THRESH}):")
for lbl in FEAT_LABELS_FULL:
    s   = feat_stds[lbl]
    ok  = id_flags[lbl]
    tag = "OK" if ok else f"WEAK (std={s:.5f})"
    print(f"    {lbl:<22}: {tag}")
print(f"\n  OLS Norton-Bailey (heat-corrected):")
print(f"    log(t) = {C_nb:.4f} + {n_nb:.4f}·log(σ) + {QR_nb:.2f}/T + {h_nb:.4f}·H_c")
print(f"\n  SINDy (thresh={best_thresh:.5f}, {n_act_t} terms):")
print(fmt_eq(coefs_act))
n_scored_final = int((df_sim_act['note'] == 'ok').sum())
print(f"\n  Similarity (scored terms, n={n_scored_final}): {mean_sim_act:.4f}")
if not np.isnan(corr):
    print(f"  Pearson r (noise→sim) = {corr:.4f}  (p={pval:.4f})")
print(f"\n  Outputs:")
for f in ["sindy_analysis_v5.png",
          "sindy_equation_similarity.csv",
          "sindy_sensitivity_analysis.csv",
          "sindy_model_comparison.csv",
          "sindy_discovered_equations.csv",
          "sindy_identifiability.csv"]:
    print(f"    {f}")