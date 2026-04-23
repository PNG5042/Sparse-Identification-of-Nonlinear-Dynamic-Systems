# ================================
# ALLOY 617 TENSILE – SINDy APPLICABILITY STUDY  (PySINDy edition) -- v6
# ================================
# CHANGES vs v5:
#   FIX-H  Instantaneous SR spike rejection at three levels:
#            TIGHT  : keep sr_inst within 10×  of specimen median
#            MOD    : keep sr_inst within 100× of specimen median
#            PCT    : keep sr_inst in global [1st, 99th] percentile
#          All three run through the full point-wise pipeline and results
#          are compared on a dedicated figure page.
#   FIX-I  QR drop reporting upgraded: every run now prints which term was
#          dropped and why (condition contribution), and a cross-filter
#          summary table shows which features survive all three levels.
#   FIX-J  Point-wise SR span now reported per filter level so the user can
#          see how much rate dynamic range survives each cut.
#   FIX-K  Sensitivity analysis runs per filter level (not just level 0).
#   FIX-L  UTS pipeline unchanged from v5 (28 specimens, all RT forms).
#   NOTE   QR is intentionally left free to prune any column; we report
#          what gets dropped and flag physically unexpected outcomes.
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

CSV_FILES = [
    "SGIHX_A1_DETAIL_DATA_Dspec.csv",
    "SGIHX_A1_DETAIL_DATA_RBspec.csv",
    "SGIHX_A1_DETAIL_DATA_Hspec.csv",
]

N_PTS_PER_SPEC = 50   # target points per specimen before SR filtering

# SR filter definitions: (label, method, factor_or_pct)
SR_FILTERS = [
    ('TIGHT',  'factor',  10.0),
    ('MOD',    'factor', 100.0),
    ('PCT',    'pct',   (1.0, 99.0)),
]

# ── colour scheme ──────────────────────────────────────────────────────
DARK ='#0d1117'; PANEL='#161b22'; GRID='#21262d'
C1='#58a6ff'; C2='#f85149'; C3='#3fb950'; C4='#d29922'; C5='#bc8cff'; TEXT='#c9d1d9'
MC = [C1, C3, C4]; HC = [C1, C3, C4, C5]
FILTER_COLORS = {'TIGHT': C1, 'MOD': C3, 'PCT': C4}

def sax(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=C1, fontsize=9, fontweight='bold', pad=6)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=8)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=8)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)

# ══════════════════════════════════════════════════════════════════════
# 0. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════
print("="*65)
print("ALLOY 617 – SINDy v6  (SR filtering × 3 + honest QR)")
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

    # ── per-specimen extraction ────────────────────────────────────────
    specimen_records = []
    pointwise_raw    = []   # all points before SR filter, with spec median sr

    for spec, grp in df_raw.groupby("Specimen_Name"):
        grp = grp.sort_values("Count").dropna(
            subset=["Stress_MPa", "Strain", "Elapsed_Time_Sec"])
        if len(grp) < 3:
            continue

        # specimen-level quantities
        idx_uts     = grp["Stress_MPa"].idxmax()
        uts         = float(grp.loc[idx_uts, "Stress_MPa"])
        eps_uts_val = float(grp.loc[idx_uts, "Strain"])
        dt_s   = grp["Elapsed_Time_Sec"].diff().clip(lower=1e-9)
        deps_s = grp["Strain"].diff()
        sr_s   = (deps_s / dt_s).dropna()
        sr_s   = sr_s[sr_s > 0]
        if len(sr_s) == 0:
            continue
        sr_specimen = float(sr_s.median())

        h_enc = float(grp["Heat_enc"].iloc[0])
        f_enc = float(grp["Form_enc"].iloc[0])

        specimen_records.append({
            "Specimen_Name": spec,
            "UTS":           uts,
            "eps_at_UTS":    eps_uts_val,
            "strain_rate":   sr_specimen,
            "Heat_enc":      h_enc,
            "Form_enc":      f_enc,
            "Heat":          grp["Heat"].iloc[0],
            "Material_Form": grp["Material_Form"].iloc[0],
        })

        # point-wise loading curve (up to UTS)
        stress  = grp["Stress_MPa"].values
        strain  = grp["Strain"].values
        elapsed = grp["Elapsed_Time_Sec"].values
        uts_idx = int(np.argmax(stress))
        if uts_idx < 2:
            continue

        stress_l  = stress[1:uts_idx+1]
        strain_l  = strain[1:uts_idx+1]
        elapsed_l = elapsed[1:uts_idx+1]

        dt_arr   = np.diff(elapsed_l, prepend=elapsed_l[0])
        deps_arr = np.diff(strain_l,  prepend=strain_l[0])
        dt_arr[dt_arr < 1e-9] = 1e-9
        sr_inst  = deps_arr / dt_arr

        valid = (stress_l > 1.0) & (strain_l > 1e-4) & (sr_inst > 0)
        if valid.sum() < 3:
            continue
        sv = stress_l[valid]; ev = strain_l[valid]; sv_r = sr_inst[valid]

        n_v = len(sv)
        idx_sel = (np.arange(n_v) if n_v <= N_PTS_PER_SPEC
                   else np.round(np.linspace(0, n_v-1, N_PTS_PER_SPEC)).astype(int))

        for i in idx_sel:
            pointwise_raw.append({
                "Specimen_Name": spec,
                "Stress_MPa":   sv[i],
                "Strain":       ev[i],
                "sr_inst":      sv_r[i],
                "sr_median":    sr_specimen,   # specimen median — used for filtering
                "Heat_enc":     h_enc,
                "Form_enc":     f_enc,
                "Heat":         grp["Heat"].iloc[0],
                "Material_Form": grp["Material_Form"].iloc[0],
            })

    summary   = pd.DataFrame(specimen_records).dropna()
    summary   = summary[
        (summary["UTS"] > 0) &
        (summary["eps_at_UTS"] > 1e-6) &
        (summary["strain_rate"] > 0)
    ].copy()
    df_pw_all = pd.DataFrame(pointwise_raw)

    # RT / HT separation
    ht_mask_primary = (
        ((summary["UTS"] < 500) & (summary["eps_at_UTS"] < 0.05))
        | (summary["UTS"] < 350)
        | ((summary["UTS"] < 550) & (summary["Material_Form"] == "Plate")
           & (summary["eps_at_UTS"] < 0.20))
    )
    iqr_outlier = pd.Series(False, index=summary.index)
    for heat, grp in summary.groupby("Heat"):
        if len(grp) < 3:
            continue
        luts = np.log(grp["UTS"])
        q1, q3 = luts.quantile(0.25), luts.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 2.0*iqr, q3 + 2.0*iqr
        iqr_outlier.loc[grp.index] = (luts < lo) | (luts > hi)

    ht_mask = ht_mask_primary | iqr_outlier
    df_ht   = summary[ht_mask].copy()
    df_rt   = summary[~ht_mask].copy()
    rt_specs = set(df_rt["Specimen_Name"])

    print(f"\n  Specimens parsed  : {len(summary)}")
    print(f"  HT/outlier        : {len(df_ht)}")
    print(f"  RT retained       : {len(df_rt)}  "
          f"(Bar={( df_rt['Material_Form']=='Bar').sum()}, "
          f"Plate={(df_rt['Material_Form']=='Plate').sum()})")
    print(f"  PW records (pre-filter): {len(df_pw_all)}")

    # keep only RT specimens in point-wise pool
    df_pw_rt = df_pw_all[df_pw_all["Specimen_Name"].isin(rt_specs)].copy()
    print(f"  PW records (RT only)   : {len(df_pw_rt)}")

    # ── Apply SR filters ───────────────────────────────────────────────
    # PCT bounds computed once on the RT pool
    pct_lo = np.percentile(df_pw_rt["sr_inst"], 1.0)
    pct_hi = np.percentile(df_pw_rt["sr_inst"], 99.0)
    print(f"\n  SR global 1–99 pct: [{pct_lo:.3e}, {pct_hi:.3e}] 1/s")

    df_pw_filtered = {}
    for flabel, fmethod, fparam in SR_FILTERS:
        if fmethod == 'factor':
            mask = ((df_pw_rt["sr_inst"] >= df_pw_rt["sr_median"] / fparam) &
                    (df_pw_rt["sr_inst"] <= df_pw_rt["sr_median"] * fparam))
        else:  # pct
            lo_p, hi_p = fparam
            mask = ((df_pw_rt["sr_inst"] >= pct_lo) &
                    (df_pw_rt["sr_inst"] <= pct_hi))
        df_f = df_pw_rt[mask].copy()
        df_pw_filtered[flabel] = df_f
        sr_span = (np.log10(df_f["sr_inst"].max()) -
                   np.log10(df_f["sr_inst"].min())) if len(df_f) > 1 else 0
        print(f"  Filter {flabel:<6}: {len(df_f):>5} pts  |  "
              f"sr=[{df_f['sr_inst'].min():.2e}, {df_f['sr_inst'].max():.2e}]  |  "
              f"span={sr_span:.2f} dec")

    # UTS arrays
    UTS         = df_rt["UTS"].values.astype(float)
    eps_at_UTS  = df_rt["eps_at_UTS"].values.astype(float)
    strain_rate = df_rt["strain_rate"].values.astype(float)
    Heat_enc_u  = df_rt["Heat_enc"].values.astype(float)
    Form_enc_u  = df_rt["Form_enc"].values.astype(float)

else:
    # synthetic fallback
    print("\n  CSVs not found – synthetic data")
    n_spec = 28
    rng = np.random.default_rng(42)
    Heat_enc_u  = rng.integers(0, 3, n_spec).astype(float)
    Form_enc_u  = rng.integers(0, 2, n_spec).astype(float)
    strain_rate = 10 ** rng.uniform(-4.5, -3.5, n_spec)
    eps_at_UTS  = rng.uniform(0.20, 0.55, n_spec)
    UTS = (727. * strain_rate**-0.016 * eps_at_UTS**0.43
           + np.array([0.,25.,-15.])[Heat_enc_u.astype(int)]
           + rng.normal(0, 12, n_spec))
    df_rt = df_ht = summary = None

    n_pw = 1400
    PW_strain_syn = rng.uniform(0.01, 0.60, n_pw)
    PW_sr_syn     = 10 ** rng.uniform(-4.5, -3.5, n_pw)
    PW_He_syn     = rng.integers(0, 3, n_pw).astype(float)
    PW_Fe_syn     = rng.integers(0, 2, n_pw).astype(float)
    PW_stress_syn = np.clip(900.*PW_sr_syn**-0.016 * PW_strain_syn**0.43
                            + rng.normal(0,15,n_pw), 1., None)
    syn_df = pd.DataFrame({'Specimen_Name':'SYN','Stress_MPa':PW_stress_syn,
                           'Strain':PW_strain_syn,'sr_inst':PW_sr_syn,
                           'sr_median':1e-4,'Heat_enc':PW_He_syn,
                           'Form_enc':PW_Fe_syn,'Heat':0,'Material_Form':'Bar'})
    df_pw_filtered = {fl: syn_df.copy() for fl, *_ in SR_FILTERS}
    rt_specs = set(); df_pw_rt = syn_df; pct_lo = PW_sr_syn.min(); pct_hi = PW_sr_syn.max()

H_mean_u = Heat_enc_u.mean(); H_c_u = Heat_enc_u - H_mean_u
log_y_uts = np.log(UTS)
log_sr_u  = np.log(np.clip(strain_rate, 1e-12, None))
log_eps_u = np.log(np.clip(eps_at_UTS,  1e-12, None))
sr_dec_u  = np.log10(strain_rate.max()) - np.log10(strain_rate.min())

# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

def fit_sindy_log(y_log, threshold, Phi, feat_labels):
    opt = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=2000)
    opt.fit(Phi, y_log)
    coeffs   = opt.coef_.ravel()
    log_pred = Phi @ coeffs
    ss_r = np.sum((y_log - log_pred)**2)
    ss_t = np.sum((y_log - y_log.mean())**2)
    r2_log = 1 - ss_r/ss_t if ss_t else 0.0
    y_pred = np.exp(log_pred); y_true = np.exp(y_log)
    ss_ru = np.sum((y_true - y_pred)**2)
    ss_tu = np.sum((y_true - y_true.mean())**2)
    r2_lin = 1 - ss_ru/ss_tu if ss_tu else 0.0
    coef_dict = {lbl: float(v) for lbl, v in zip(feat_labels, coeffs)
                 if abs(v) > 1e-10}
    return coef_dict, r2_log, r2_lin, log_pred

def tune_threshold(y_log, Phi, feat_labels, thresholds, r2_floor=0.55):
    best_thresh = thresholds[0]; best_score = -np.inf
    for thresh in thresholds:
        try:
            cd, r2l, _, _ = fit_sindy_log(y_log, thresh, Phi, feat_labels)
            nact = len(cd)
            if r2l < r2_floor or nact < 2: continue
            score = r2l + max(0, len(feat_labels) - nact) * 0.015
            if score > best_score: best_score, best_thresh = score, thresh
        except Exception: continue
    return best_thresh

def identifiability_check(Phi_full, feat_labels, log_y, id_thresh=0.10, verbose=True):
    id_flags = {}; feat_stds = {}
    thr = id_thresh * log_y.std()
    if verbose:
        print(f"  Target std={log_y.std():.5f}  ID-thr={thr:.5f}")
        print(f"  {'Feature':<22} {'std':>10} {'identifiable?':>14}")
        print("  " + "-"*48)
    for i, lbl in enumerate(feat_labels):
        col = Phi_full[:, i]; s = col.std(); feat_stds[lbl] = s
        if lbl == 'C0':
            id_flags[lbl] = True
            if verbose: print(f"  {'C0':<22} {'(intercept)':>10} {'YES':>14}")
            continue
        ok = s >= thr; id_flags[lbl] = ok
        flag = "YES" if ok else "*** WEAK ***"
        if verbose: print(f"  {lbl:<22} {s:>10.5f} {flag:>14}")
    return id_flags, feat_stds

def qr_prune(Phi_id, id_labels, cond_target=50.0, verbose=True):
    """
    FIX-I: returns (Phi_final, active_labels, cond_final, dropped_labels).
    Prints which column was dropped at each step and reports condition
    contribution so physically unexpected drops are visible.
    """
    col_norms = np.linalg.norm(Phi_id, axis=0, keepdims=True)
    col_norms[col_norms==0] = 1.0
    _, _, piv   = scipy_qr(Phi_id / col_norms, pivoting=True)
    cond_prev   = np.linalg.cond(Phi_id)
    keep_idx    = list(range(len(id_labels)))
    piv_order   = list(piv)
    dropped     = []

    if verbose:
        print(f"  QR pivot order: {[id_labels[i] for i in piv]}")
        print(f"  Initial cond = {cond_prev:.1f}  (target ≤ {cond_target:.0f})")

    while len(keep_idx) > 2:
        cond_now = np.linalg.cond(Phi_id[:, keep_idx])
        if cond_now <= cond_target:
            break
        for cand in reversed(piv_order):
            if cand in keep_idx:
                # measure condition improvement from dropping this column
                trial = [k for k in keep_idx if k != cand]
                cond_trial = np.linalg.cond(Phi_id[:, trial])
                if verbose:
                    print(f"    Drop '{id_labels[cand]}':  "
                          f"cond {cond_now:.1f} → {cond_trial:.1f}"
                          + ("  *** physically primary term ***"
                             if id_labels[cand] in ('log(eps)', 'log(sr_inst)',
                                                     'log(sr)') else ""))
                keep_idx.remove(cand); piv_order.remove(cand)
                dropped.append(id_labels[cand])
                break

    Phi_fin  = Phi_id[:, keep_idx]
    cond_fin = np.linalg.cond(Phi_fin)
    active   = [id_labels[i] for i in keep_idx]
    if verbose:
        print(f"  Final cond={cond_fin:.1f}  "
              f"kept={active}  dropped={dropped if dropped else 'none'}")
    return Phi_fin, active, cond_fin, dropped

def compute_sim(sindy_coefs, ref, id_flags_map, log_y_std):
    rows = []
    for param, ana in ref.items():
        identifiable = id_flags_map.get(param, True)
        sval         = sindy_coefs.get(param, 0.0)
        if not identifiable:
            rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                         'rel_err': np.nan, 'similarity': np.nan,
                         'identifiable': False}); continue
        if abs(ana) > 0.01:
            rel_err = (ana - sval) / ana
            sim     = float(np.clip(1.0 - abs(rel_err), -1.0, 1.0))
        else:
            scale   = max(abs(ana), 0.1 * log_y_std)
            rel_err = np.nan
            sim     = float(np.clip(1.0 - abs(ana-sval)/scale, -1.0, 1.0))
        rows.append({'param': param, 'OLS_ref': ana, 'sindy': sval,
                     'rel_err': rel_err, 'similarity': sim,
                     'identifiable': True})
    return pd.DataFrame(rows)

def fmt_eq(coef_dict, ylbl='log(σ)'):
    if not coef_dict: return f"  {ylbl} = 0"
    lines = []
    if 'C0' in coef_dict:
        v = coef_dict['C0']
        lines.append(f"  {v:+.5g}  [K≈{np.exp(v):.1f} MPa]")
    for k, v in coef_dict.items():
        if k != 'C0': lines.append(f"  {v:+.5g} · {k}")
    return f"  {ylbl} =\n" + "\n".join(lines)

thresholds = np.logspace(-3, 1, 150)
noise_levels = [0, 5, 10, 20, 30, 50, 75, 100]

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS A: POINT-WISE — THREE SR FILTER LEVELS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ANALYSIS A: POINT-WISE  σ(ε, dε/dt)  — 3 SR FILTER LEVELS")
print("="*65)

FEAT_LABELS_PW = ['C0', 'log(eps)', 'log(eps)^2', 'log(sr_inst)', 'H_c', 'F_enc']

def build_phi_pw(Hc, F, l_eps, l_sr):
    return np.column_stack([
        np.ones(len(Hc)), l_eps, l_eps**2, l_sr, Hc, F
    ])

pw_results = {}   # keyed by filter label

rng_sens = np.random.default_rng(7)

for flabel, fmethod, fparam in SR_FILTERS:
    print(f"\n{'─'*65}")
    print(f"  FILTER: {flabel}  "
          + (f"(±{fparam}× specimen median sr)"
             if fmethod=='factor'
             else f"(global 1–99 pct: [{pct_lo:.2e}, {pct_hi:.2e}])"))
    print(f"{'─'*65}")

    df_f = df_pw_filtered[flabel]
    n_f  = len(df_f)
    if n_f < 20:
        print(f"  SKIP: only {n_f} points after filtering")
        continue

    PW_stress   = df_f["Stress_MPa"].values.astype(float)
    PW_strain   = df_f["Strain"].values.astype(float)
    PW_sr       = df_f["sr_inst"].values.astype(float)
    PW_He       = df_f["Heat_enc"].values.astype(float)
    PW_Fe       = df_f["Form_enc"].values.astype(float)

    H_c_pw   = PW_He - PW_He.mean()
    log_y_pw = np.log(PW_stress)
    log_eps  = np.log(np.clip(PW_strain, 1e-12, None))
    log_sr   = np.log(np.clip(PW_sr,    1e-12, None))
    sr_span  = np.log10(PW_sr.max()) - np.log10(PW_sr.min())

    print(f"  N={n_f}  |  sr=[{PW_sr.min():.2e}, {PW_sr.max():.2e}]  "
          f"|  span={sr_span:.2f} dec")

    # Build Φ
    Phi_full = build_phi_pw(H_c_pw, PW_Fe, log_eps, log_sr)

    # Identifiability
    print("\n  [Identifiability]")
    id_flags, feat_stds = identifiability_check(Phi_full, FEAT_LABELS_PW,
                                                log_y_pw, verbose=True)

    # OLS reference
    A_ols = np.column_stack([np.ones(n_f), log_sr, log_eps])
    ro, *_ = np.linalg.lstsq(A_ols, log_y_pw, rcond=None)
    logK_ols, m_ols, p_ols = ro
    K_ols = np.exp(logK_ols)
    print(f"\n  OLS: K={K_ols:.1f}  m={m_ols:.5f}  p={p_ols:.5f}")

    # QR pruning
    print("\n  [QR Decorrelation]")
    id_cols   = [i for i, lbl in enumerate(FEAT_LABELS_PW) if id_flags[lbl]]
    id_labels = [FEAT_LABELS_PW[i] for i in id_cols]
    Phi_id    = Phi_full[:, id_cols]
    Phi_fin, FEAT_ACT, cond_fin, dropped = qr_prune(Phi_id, id_labels,
                                                      verbose=True)

    # flag physically unexpected drops
    primary_dropped = [d for d in dropped
                       if d in ('log(eps)', 'log(sr_inst)')]
    if primary_dropped:
        print(f"  *** WARNING: primary physical terms dropped: "
              f"{primary_dropped} ***")
        print(f"      This means log(eps) and log(eps)^2 are too collinear "
              f"for QR to distinguish them at this filter level.")

    # SINDy
    print("\n  [SINDy Fit]")
    best_thresh = tune_threshold(log_y_pw, Phi_fin, FEAT_ACT,
                                 thresholds, r2_floor=0.50)
    coefs, r2_log, r2_lin, logp = fit_sindy_log(
        log_y_pw, best_thresh, Phi_fin, FEAT_ACT)
    print(f"  Threshold={best_thresh:.5f}  R²_log={r2_log:.4f}  "
          f"R²_lin={r2_lin:.4f}")
    print(f"  Active ({len(coefs)}): {list(coefs.keys())}")
    print(fmt_eq(coefs))

    analytic_ref = {'C0': logK_ols, 'log(sr_inst)': m_ols, 'log(eps)': p_ols}
    df_sim = compute_sim(coefs, analytic_ref, id_flags, log_y_pw.std())
    mean_sim = df_sim['similarity'].dropna().mean()
    print(f"  Mean similarity (identifiable): {mean_sim:.4f}")

    # Sensitivity
    print("\n  [Sensitivity]")
    rng_loc = np.random.default_rng(7)
    sens_rows = []
    for s_mpa in noise_levels:
        y_n    = np.clip(PW_stress + rng_loc.normal(0, s_mpa, n_f), 1., None)
        log_yn = np.log(y_n)
        bt     = tune_threshold(log_yn, Phi_fin, FEAT_ACT,
                                thresholds, r2_floor=0.45)
        cn, r2l, r2u, _ = fit_sindy_log(log_yn, bt, Phi_fin, FEAT_ACT)
        dfs    = compute_sim(cn, analytic_ref, id_flags, log_yn.std())
        msim   = dfs['similarity'].dropna().mean()
        pct    = 100 * s_mpa / max(PW_stress.mean(), 1)
        sens_rows.append({'noise_MPa': s_mpa, 'pct_err': pct,
                          'n_active': len(cn), 'mean_sim': msim,
                          'r2_log': r2l,
                          'sindy_m': cn.get('log(sr_inst)', 0),
                          'sindy_p': cn.get('log(eps)', 0)})
        print(f"    noise={s_mpa:>4} MPa ({pct:>5.1f}%)  "
              f"active={len(cn)}  R²={r2l:.4f}  sim={msim:.4f}  "
              f"m={cn.get('log(sr_inst)',0):+.5f}  "
              f"p={cn.get('log(eps)',0):+.5f}")
    df_sens = pd.DataFrame(sens_rows)
    corr = pval = sl = ic = np.nan
    pv = df_sens['pct_err'].values; sv = df_sens['mean_sim'].values
    if not np.all(np.isnan(sv)) and len(pv) > 3:
        try:
            corr, pval = stats.pearsonr(pv, sv)
            sl, ic, *_ = stats.linregress(pv, sv)
            print(f"    Pearson r={corr:.4f}  p={pval:.4f}")
        except Exception:
            pass

    pw_results[flabel] = dict(
        df=df_f, n=n_f, sr_span=sr_span,
        PW_stress=PW_stress, PW_strain=PW_strain, PW_sr=PW_sr,
        H_c_pw=H_c_pw, PW_Fe=PW_Fe,
        log_y_pw=log_y_pw, log_eps=log_eps, log_sr=log_sr,
        id_flags=id_flags, feat_stds=feat_stds,
        K_ols=K_ols, m_ols=m_ols, p_ols=p_ols, logK_ols=logK_ols,
        FEAT_ACT=FEAT_ACT, Phi_fin=Phi_fin, cond_fin=cond_fin,
        dropped=dropped, primary_dropped=primary_dropped,
        coefs=coefs, r2_log=r2_log, r2_lin=r2_lin,
        logp=logp, df_sim=df_sim, mean_sim=mean_sim,
        df_sens=df_sens, corr=corr, pval=pval, sl=sl, ic=ic,
        analytic_ref=analytic_ref,
    )

# cross-filter summary table
print("\n" + "="*65)
print("  CROSS-FILTER SUMMARY")
print("="*65)
hdr = f"  {'Filter':<8} {'N':>6} {'span':>6} {'cond':>7}  "
hdr += "  ".join(f"{lbl:<14}" for lbl in FEAT_LABELS_PW)
hdr += f"  {'R²_log':>7}  {'mean_sim':>8}"
print(hdr); print("  " + "-"*len(hdr))
for flabel, *_ in SR_FILTERS:
    if flabel not in pw_results: continue
    r = pw_results[flabel]
    row = f"  {flabel:<8} {r['n']:>6} {r['sr_span']:>6.2f} {r['cond_fin']:>7.1f}  "
    for lbl in FEAT_LABELS_PW:
        status = ("KEPT" if lbl in r['FEAT_ACT']
                  else "DROPPED" if lbl in r['id_flags'] and r['id_flags'][lbl]
                  else "WEAK")
        row += f"{status:<16}"
    row += f"  {r['r2_log']:>7.4f}  {r['mean_sim']:>8.4f}"
    print(row)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS B: UTS (unchanged from v5)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ANALYSIS B: UTS  (N=28, all RT forms)  — unchanged from v5")
print("="*65)

FEAT_LABELS_UTS = ['C0', 'log(sr)', 'log(eps)', 'log(eps)^2',
                   'H_c', 'H_c*log(eps)']

def build_phi_uts(Hc, l_eps, l_sr):
    return np.column_stack([
        np.ones(len(Hc)), l_sr, l_eps, l_eps**2, Hc, Hc*l_eps
    ])

Phi_uts_full = build_phi_uts(H_c_u, log_eps_u, log_sr_u)
print("\n  [Identifiability]")
id_flags_uts, feat_stds_uts = identifiability_check(
    Phi_uts_full, FEAT_LABELS_UTS, log_y_uts)

A_ro_u = np.column_stack([np.ones_like(log_sr_u), log_sr_u, log_eps_u])
ro_u, *_ = np.linalg.lstsq(A_ro_u, log_y_uts, rcond=None)
logK_ro, m_ro, p_ro = ro_u; K_ro = np.exp(logK_ro)
print(f"\n  OLS: K={K_ro:.1f}  m={m_ro:.5f}  p={p_ro:.5f}")

print("\n  [QR Decorrelation]")
id_cols_u   = [i for i, lbl in enumerate(FEAT_LABELS_UTS) if id_flags_uts[lbl]]
id_labels_u = [FEAT_LABELS_UTS[i] for i in id_cols_u]
Phi_id_u    = Phi_uts_full[:, id_cols_u]
Phi_fin_u, FEAT_ACT_UTS, cond_u, dropped_u = qr_prune(
    Phi_id_u, id_labels_u, verbose=True)

# ML
print("\n  [ML Model]")
def ml_feats_u(Hc, F, eps, sr, l_eps, l_sr):
    return np.column_stack([
        np.ones_like(Hc), Hc, Hc**2, F, eps, eps**2,
        np.sqrt(np.clip(eps,0,None)), l_eps, sr, l_sr,
        eps*l_sr, l_eps*l_sr, Hc*l_eps, Hc*l_sr,
    ])
X_all_u = ml_feats_u(H_c_u, Form_enc_u, eps_at_UTS, strain_rate,
                     log_eps_u, log_sr_u)
sel_u   = SelectKBest(f_regression, k=min(10, X_all_u.shape[1])).fit(X_all_u, UTS)
X_sc_u  = StandardScaler().fit_transform(sel_u.transform(X_all_u))
n_s_u   = len(UTS)
if n_s_u >= 20:
    bins_u = np.digitize(UTS, np.percentile(UTS,[20,40,60,80]))
    tr_i_u, te_i_u = next(
        StratifiedShuffleSplit(1,test_size=0.25,random_state=42
                               ).split(X_sc_u, bins_u))
else:
    sp = max(1, int(n_s_u*0.75))
    tr_i_u, te_i_u = np.arange(sp), np.arange(sp, n_s_u)
Xtr_u,Xte_u = X_sc_u[tr_i_u],X_sc_u[te_i_u]
ytr_u,yte_u = UTS[tr_i_u],UTS[te_i_u]
n_cv_u = min(5, max(2, len(tr_i_u)//5))
rcv_u  = GridSearchCV(Ridge(),{'alpha':[0.1,1,10,50,100,500]},
                      cv=n_cv_u,scoring='r2').fit(Xtr_u,ytr_u)
ridge_u = rcv_u.best_estimator_
rf_u    = RandomForestRegressor(300,max_depth=6,
              min_samples_split=max(4,len(tr_i_u)//15),
              min_samples_leaf=max(2,len(tr_i_u)//25),
              max_features='sqrt',random_state=42).fit(Xtr_u,ytr_u)
ens_u   = VotingRegressor([('r1',ridge_u),
                            ('r2',Ridge(alpha=rcv_u.best_params_['alpha'])),
                            ('rf',rf_u)]).fit(Xtr_u,ytr_u)
ML_u    = {'Ridge':ridge_u,'RandomForest':rf_u,'Ensemble':ens_u}
RES_u   = {}
for nm,m_est in ML_u.items():
    yp  = m_est.predict(Xte_u)
    err = np.abs((yp-yte_u)/np.clip(np.abs(yte_u),1,None))*100
    RES_u[nm] = dict(tr_r2=m_est.score(Xtr_u,ytr_u),
                     te_r2=m_est.score(Xte_u,yte_u),
                     med_err=np.median(err), ypall=m_est.predict(X_sc_u))
    print(f"  {nm:<14} TrainR²={RES_u[nm]['tr_r2']:.4f}  "
          f"TestR²={RES_u[nm]['te_r2']:.4f}  "
          f"MedianErr={RES_u[nm]['med_err']:.1f}%")
best_ml_u = max(RES_u, key=lambda k: RES_u[k]['te_r2'])

print("\n  [SINDy Fit]")
best_thresh_u = tune_threshold(log_y_uts,Phi_fin_u,FEAT_ACT_UTS,
                               thresholds,r2_floor=0.55)
coefs_u,r2_u_log,r2_u_lin,logp_u = fit_sindy_log(
    log_y_uts,best_thresh_u,Phi_fin_u,FEAT_ACT_UTS)
print(f"  Threshold={best_thresh_u:.5f}  R²_log={r2_u_log:.4f}  R²_UTS={r2_u_lin:.4f}")
print(fmt_eq(coefs_u,'log(UTS)'))

analytic_ref_uts = {'C0':logK_ro,'log(sr)':m_ro,'log(eps)':p_ro}
df_sim_u  = compute_sim(coefs_u,analytic_ref_uts,id_flags_uts,log_y_uts.std())
mean_sim_u = df_sim_u['similarity'].dropna().mean()
print(f"  Mean similarity: {mean_sim_u:.4f}")

print("\n  [Sensitivity]")
rng3 = np.random.default_rng(11)
sens_rows_u = []
for s_mpa in noise_levels:
    y_n = np.clip(UTS+rng3.normal(0,s_mpa,len(UTS)),1.,None)
    log_yn = np.log(y_n)
    bt = tune_threshold(log_yn,Phi_fin_u,FEAT_ACT_UTS,thresholds,r2_floor=0.50)
    cn,r2l,r2u,_ = fit_sindy_log(log_yn,bt,Phi_fin_u,FEAT_ACT_UTS)
    dfs = compute_sim(cn,analytic_ref_uts,id_flags_uts,log_yn.std())
    msim = dfs['similarity'].dropna().mean()
    pct  = 100*s_mpa/max(UTS.mean(),1)
    sens_rows_u.append({'noise_MPa':s_mpa,'pct_err':pct,
                        'n_active':len(cn),'mean_sim':msim,
                        'r2_log':r2l,'sindy_m':cn.get('log(sr)',0),
                        'sindy_p':cn.get('log(eps)',0)})
    print(f"  noise={s_mpa:>4} MPa ({pct:>5.1f}%)  active={len(cn)}  "
          f"R²={r2l:.4f}  sim={msim:.4f}")
df_sens_u = pd.DataFrame(sens_rows_u)
corr_u = pval_u = sl_u = ic_u = np.nan
pv_u = df_sens_u['pct_err'].values; sv_u = df_sens_u['mean_sim'].values
if not np.all(np.isnan(sv_u)) and len(pv_u) > 3:
    try:
        corr_u,pval_u = stats.pearsonr(pv_u,sv_u)
        sl_u,ic_u,*_ = stats.linregress(pv_u,sv_u)
        print(f"  Pearson r={corr_u:.4f}  p={pval_u:.4f}")
    except Exception: pass

# ══════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ══════════════════════════════════════════════════════════════════════
print("\n  Saving CSVs...")
for flabel in pw_results:
    r = pw_results[flabel]
    r['df_sim'].to_csv(
        OUT/f"sindy_617t_pw_{flabel.lower()}_similarity.csv",
        index=False, float_format='%.6f')
    r['df_sens'].to_csv(
        OUT/f"sindy_617t_pw_{flabel.lower()}_sensitivity.csv",
        index=False, float_format='%.6f')
df_sim_u.to_csv(OUT/"sindy_617t_uts_similarity_v6.csv",
                index=False, float_format='%.6f')
df_sens_u.to_csv(OUT/"sindy_617t_uts_sensitivity_v6.csv",
                 index=False, float_format='%.6f')
if REAL and df_rt is not None:
    df_rt.to_csv(OUT/"sindy_617t_RT_specimens_v6.csv",
                 index=False, float_format='%.6f')
print("  CSVs saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 – SR FILTER COMPARISON (point-wise)
# ══════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(22, 18), facecolor=DARK)
gs1  = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.52, wspace=0.38)
fig1.suptitle(
    "Alloy 617 – Point-Wise Analysis  |  SR Filter Comparison  [v6]\n"
    "TIGHT=±10×  |  MOD=±100×  |  PCT=global 1–99th percentile",
    fontsize=11, fontweight='bold', color=C1, y=0.999)

flist = [fl for fl, *_ in SR_FILTERS if fl in pw_results]

for row_i, flabel in enumerate(flist):
    r   = pw_results[flabel]
    col = FILTER_COLORS[flabel]

    # col 0 – stress vs strain scatter
    ax = fig1.add_subplot(gs1[row_i, 0])
    ax.scatter(r['PW_strain'], r['PW_stress'], s=5, alpha=0.35,
               color=col, edgecolors='none')
    sax(ax, f"{flabel}: Stress-Strain  (N={r['n']})\n"
            f"SR span={r['sr_span']:.2f} dec  cond={r['cond_fin']:.1f}",
        'Strain', 'Stress (MPa)')
    # annotate dropped terms
    if r['dropped']:
        ax.text(0.02, 0.08, f"QR dropped: {r['dropped']}",
                transform=ax.transAxes, fontsize=7.5, color=C2,
                fontstyle='italic')

    # col 1 – SINDy predicted vs actual
    ax2 = fig1.add_subplot(gs1[row_i, 1])
    pred = np.exp(r['logp'])
    sl_lim = [r['PW_stress'].min()-20, r['PW_stress'].max()+20]
    ax2.scatter(r['PW_stress'], pred, s=5, alpha=0.35,
                color=col, edgecolors='none')
    ax2.plot(sl_lim, sl_lim, '--', color=C2, lw=1.5)
    ax2.text(0.04, 0.90,
             f"R²_log={r['r2_log']:.4f}\nR²_lin={r['r2_lin']:.4f}\n"
             f"sim={r['mean_sim']:.4f}",
             transform=ax2.transAxes, color=col, fontsize=8.5,
             fontweight='bold')
    # annotate active equation terms
    eq_str = "  ".join(
        f"{k}={v:+.3f}" for k, v in r['coefs'].items() if k != 'C0')
    ax2.text(0.04, 0.04, eq_str, transform=ax2.transAxes,
             fontsize=7, color=TEXT)
    sax(ax2, f"{flabel}: SINDy Actual vs Predicted",
        'Actual σ (MPa)', 'Predicted σ (MPa)')

    # col 2 – sensitivity curve
    ax3 = fig1.add_subplot(gs1[row_i, 2])
    ds  = r['df_sens']
    ax3.plot(ds['pct_err'], ds['mean_sim'], 'o-',
             color=col, lw=2, ms=5, label='mean sim')
    ax3_r = ax3.twinx()
    ax3_r.plot(ds['pct_err'], ds['r2_log'], 's--',
               color=C5, lw=1.5, ms=4, alpha=0.7, label='R²_log')
    ax3_r.set_ylabel('R²_log', color=C5, fontsize=8)
    ax3_r.tick_params(colors=TEXT, labelsize=7)
    for sp in ax3_r.spines.values(): sp.set_edgecolor(GRID)
    if not np.isnan(r['sl']):
        xf = np.linspace(0, ds['pct_err'].max(), 200)
        ax3.plot(xf, r['ic'] + r['sl']*xf, '--', color=C2, lw=1,
                 alpha=0.7, label=f"r={r['corr']:.3f}")
    ax3.axhline(1.0, color=TEXT, ls=':', alpha=0.3)
    ax3.axhline(0.0, color=C2,  ls=':', alpha=0.3)
    ax3.set_ylim([-0.3, 1.3])
    ax3.legend(fontsize=6.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    sax(ax3, f"{flabel}: Noise Sensitivity",
        'Noise (% of mean σ)', 'Mean Similarity')

plt.savefig(OUT/"sindy_617t_v6_filter_comparison.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: sindy_617t_v6_filter_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 – CROSS-FILTER DIAGNOSTICS  (SR span, m/p stability, similarity)
# ══════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(22, 12), facecolor=DARK)
gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.50, wspace=0.40)
fig2.suptitle(
    "Alloy 617 – Cross-Filter Diagnostics  [v6]\n"
    "How SR spike removal changes identifiability, QR outcome, and equation quality",
    fontsize=11, fontweight='bold', color=C1, y=0.999)

flist_valid = [fl for fl, *_ in SR_FILTERS if fl in pw_results]
x_pos = np.arange(len(flist_valid))
bar_w = 0.32

# P1 – N points and SR span per filter
ax_a = fig2.add_subplot(gs2[0, 0])
ns   = [pw_results[fl]['n']       for fl in flist_valid]
spans= [pw_results[fl]['sr_span'] for fl in flist_valid]
bars_n = ax_a.bar(x_pos - bar_w/2, ns, bar_w,
                   color=[FILTER_COLORS[fl] for fl in flist_valid],
                   alpha=0.85, edgecolor=PANEL, label='N points')
ax_a2 = ax_a.twinx()
ax_a2.bar(x_pos + bar_w/2, spans, bar_w, color=C5, alpha=0.6,
          edgecolor=PANEL, label='SR span (dec)')
ax_a2.set_ylabel('SR span (decades)', color=C5, fontsize=8)
ax_a2.tick_params(colors=TEXT, labelsize=7)
for sp in ax_a2.spines.values(): sp.set_edgecolor(GRID)
ax_a.set_xticks(x_pos); ax_a.set_xticklabels(flist_valid, color=TEXT, fontsize=9)
for bar, n in zip(bars_n, ns):
    ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
              str(n), ha='center', va='bottom', fontsize=8, color=TEXT)
sax(ax_a, 'Points Retained vs SR Span per Filter',
    'Filter Level', 'N points')

# P2 – QR drop outcome per filter
ax_b = fig2.add_subplot(gs2[0, 1])
ax_b.set_facecolor(PANEL)
ax_b.axis('off')
rows_tbl = []
for fl in flist_valid:
    r = pw_results[fl]
    for lbl in FEAT_LABELS_PW:
        if lbl == 'C0': continue
        if not r['id_flags'].get(lbl, True):
            status = 'WEAK'
        elif lbl in r['FEAT_ACT']:
            status = 'KEPT'
        else:
            status = 'DROPPED'
        rows_tbl.append({'Filter': fl, 'Feature': lbl, 'Status': status})
df_tbl = pd.DataFrame(rows_tbl).pivot(index='Feature', columns='Filter',
                                       values='Status')
# draw as text table
col_x = {fl: 0.18 + i*0.27 for i, fl in enumerate(flist_valid)}
ax_b.text(0.02, 0.97, 'QR Outcome per Filter', transform=ax_b.transAxes,
          fontsize=9, color=C1, fontweight='bold', va='top')
for i, fl in enumerate(flist_valid):
    ax_b.text(col_x[fl], 0.88, fl, transform=ax_b.transAxes,
              fontsize=8.5, color=FILTER_COLORS[fl], fontweight='bold',
              ha='center')
for j, feat in enumerate(df_tbl.index):
    y = 0.78 - j*0.13
    ax_b.text(0.02, y, feat, transform=ax_b.transAxes,
              fontsize=8, color=TEXT, va='center')
    for fl in flist_valid:
        stat = df_tbl.loc[feat, fl] if fl in df_tbl.columns else '—'
        col_stat = (C3 if stat=='KEPT' else
                    C2 if stat=='DROPPED' else C4)
        ax_b.text(col_x[fl], y, stat, transform=ax_b.transAxes,
                  fontsize=8, color=col_stat, ha='center', va='center',
                  fontweight='bold')
ax_b.set_title('QR Feature Outcome Table',
               color=C1, fontsize=9, fontweight='bold', pad=6)
for sp in ax_b.spines.values(): sp.set_edgecolor(GRID)

# P3 – R² and mean_sim per filter
ax_c = fig2.add_subplot(gs2[0, 2])
r2s  = [pw_results[fl]['r2_log']  for fl in flist_valid]
sims = [pw_results[fl]['mean_sim'] for fl in flist_valid]
ax_c.bar(x_pos - bar_w/2, r2s,  bar_w,
         color=[FILTER_COLORS[fl] for fl in flist_valid],
         alpha=0.85, edgecolor=PANEL, label='R²_log')
ax_c.bar(x_pos + bar_w/2, sims, bar_w, color=C5, alpha=0.7,
         edgecolor=PANEL, label='mean_sim')
ax_c.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.4)
ax_c.set_xticks(x_pos); ax_c.set_xticklabels(flist_valid, color=TEXT, fontsize=9)
ax_c.set_ylim([0, 1.2])
ax_c.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
for i, (r2, sm) in enumerate(zip(r2s, sims)):
    ax_c.text(i-bar_w/2, r2+0.02, f'{r2:.3f}', ha='center', fontsize=7.5, color=TEXT)
    ax_c.text(i+bar_w/2, sm+0.02, f'{sm:.3f}', ha='center', fontsize=7.5, color=C5)
sax(ax_c, 'Fit Quality vs Equation Similarity\nper SR Filter', 'Filter', 'Score')

# P4 – m (rate exponent) stability across noise, all filters
ax_d = fig2.add_subplot(gs2[1, 0])
for fl in flist_valid:
    ds = pw_results[fl]['df_sens']
    ax_d.plot(ds['pct_err'], ds['sindy_m'], 'o-',
              color=FILTER_COLORS[fl], lw=2, ms=5, label=fl)
    ax_d.axhline(pw_results[fl]['m_ols'], color=FILTER_COLORS[fl],
                 ls='--', lw=1, alpha=0.5)
ax_d.axhline(0, color=TEXT, ls=':', lw=1, alpha=0.4)
ax_d.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_d, 'Rate Exponent m Stability vs Noise\n(dashed = OLS reference)',
    'Noise (% of mean σ)', 'SINDy m [log(sr_inst)]')

# P5 – p (strain exponent) stability across noise, all filters
ax_e = fig2.add_subplot(gs2[1, 1])
for fl in flist_valid:
    ds = pw_results[fl]['df_sens']
    ax_e.plot(ds['pct_err'], ds['sindy_p'], 'o-',
              color=FILTER_COLORS[fl], lw=2, ms=5, label=fl)
    ax_e.axhline(pw_results[fl]['p_ols'], color=FILTER_COLORS[fl],
                 ls='--', lw=1, alpha=0.5)
ax_e.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_e, 'Strain Exponent p Stability vs Noise\n(dashed = OLS reference)',
    'Noise (% of mean σ)', 'SINDy p [log(eps)]')

# P6 – mean sim vs noise, all filters + UTS overlay
ax_f = fig2.add_subplot(gs2[1, 2])
for fl in flist_valid:
    ds = pw_results[fl]['df_sens']
    ax_f.plot(ds['pct_err'], ds['mean_sim'], 'o-',
              color=FILTER_COLORS[fl], lw=2, ms=5, label=f'PW-{fl}')
ax_f.plot(df_sens_u['pct_err'], df_sens_u['mean_sim'], 's--',
          color=C2, lw=2, ms=5, label='UTS')
ax_f.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.3)
ax_f.axhline(0.0, color=C4,  ls=':', lw=1, alpha=0.3)
ax_f.set_ylim([-0.3, 1.3])
ax_f.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_f, 'Noise Sensitivity Comparison\n(all filters + UTS)',
    'Noise (% of mean)', 'Mean Similarity')

plt.savefig(OUT/"sindy_617t_v6_cross_filter.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: sindy_617t_v6_cross_filter.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 – UTS ANALYSIS (same layout as v5)
# ══════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(22, 10), facecolor=DARK)
gs3  = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.50, wspace=0.38)
fig3.suptitle(
    f"Alloy 617 – UTS Specimen-Level Analysis  [v6]  N={len(UTS)} RT specimens\n"
    f"SR span={sr_dec_u:.2f} dec  |  OLS: K={K_ro:.1f} m={m_ro:.4f} p={p_ro:.4f}",
    fontsize=11, fontweight='bold', color=C1, y=0.999)

ax_u1 = fig3.add_subplot(gs3[0, 0])
if REAL and df_ht is not None:
    ax_u1.scatter(df_rt['eps_at_UTS'], df_rt['UTS'],
                  s=45, alpha=0.85, color=C1, edgecolors='none', label='RT')
    ax_u1.scatter(df_ht['eps_at_UTS'], df_ht['UTS'],
                  s=45, alpha=0.85, color=C2, marker='x', lw=1.5,
                  label='HT/outlier')
else:
    ax_u1.scatter(eps_at_UTS, UTS, s=40, alpha=0.8, color=C1, edgecolors='none')
ax_u1.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_u1, 'All Specimens: UTS vs eps@UTS', 'eps @ UTS', 'UTS (MPa)')

ax_u2 = fig3.add_subplot(gs3[0, 1])
y_best_u = RES_u[best_ml_u]['ypall']
uts_ll   = [UTS.min()-20, UTS.max()+20]
n_heats_u = int(Heat_enc_u.max()) + 1
for hi in range(n_heats_u):
    mask = Heat_enc_u == hi
    ax_u2.scatter(UTS[mask], y_best_u[mask], s=55, alpha=0.85,
                  color=HC[hi%len(HC)], edgecolors='none', label=f"Heat {hi}")
ax_u2.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
ax_u2.text(0.05, 0.91, f"R²={RES_u[best_ml_u]['te_r2']:.4f}",
           transform=ax_u2.transAxes, color=C3, fontsize=9, fontweight='bold')
ax_u2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_u2, f'ML ({best_ml_u}) Actual vs Predicted', 'Actual UTS', 'Predicted UTS')

ax_u3 = fig3.add_subplot(gs3[0, 2])
yp_uts = np.exp(logp_u)
for hi in range(n_heats_u):
    mask = Heat_enc_u == hi
    ax_u3.scatter(UTS[mask], yp_uts[mask], s=55, alpha=0.85,
                  color=HC[hi%len(HC)], edgecolors='none')
ax_u3.plot(uts_ll, uts_ll, '--', color=C2, lw=1.5)
ax_u3.text(0.05, 0.91,
           f"R²_log={r2_u_log:.4f}\nR²_UTS={r2_u_lin:.4f}\n"
           f"sim={mean_sim_u:.4f}",
           transform=ax_u3.transAxes, color=C3, fontsize=9, fontweight='bold')
sax(ax_u3, f'SINDy (UTS): Actual vs Predicted', 'Actual UTS', 'SINDy UTS')

ax_u4 = fig3.add_subplot(gs3[1, 0])
fn_u = list(feat_stds_uts.keys())
col_id_u = [C3 if id_flags_uts[f] else C2 for f in fn_u]
bars_u = ax_u4.barh(fn_u, [feat_stds_uts[f] for f in fn_u],
                    color=col_id_u, alpha=0.85, edgecolor=PANEL)
thr_u = 0.10 * log_y_uts.std()
ax_u4.axvline(thr_u, color=C4, ls='--', lw=1.5, label='ID thresh')
ax_u4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
for bar, lbl in zip(bars_u, fn_u):
    tag = " ✓" if id_flags_uts[lbl] else " WEAK"
    ax_u4.text(bar.get_width()+thr_u*0.05, bar.get_y()+bar.get_height()/2,
               tag, va='center', fontsize=7, color=TEXT)
sax(ax_u4, 'Feature Identifiability (UTS)', 'std(feature)', '')

ax_u5 = fig3.add_subplot(gs3[1, 1])
plabs_u = list(analytic_ref_uts.keys()); xp_u = np.arange(len(plabs_u))
bars_u5 = ax_u5.bar(xp_u,
                     df_sim_u['similarity'].fillna(-0.05).values,
                     color=[C3 if v else C4
                            for v in df_sim_u['identifiable'].values],
                     alpha=0.85, edgecolor=PANEL)
ax_u5.axhline(1.0, color=TEXT, ls=':', lw=1, alpha=0.5)
ax_u5.axhline(0.0, color=C2,  ls='--', lw=1, alpha=0.5)
for bar, val in zip(bars_u5, df_sim_u['similarity'].values):
    tag = f'{val:.3f}' if not np.isnan(val) else 'NaN'
    ax_u5.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),0)+0.03,
               tag, ha='center', va='bottom', fontsize=8, color=TEXT)
ax_u5.set_xticks(xp_u); ax_u5.set_xticklabels(plabs_u, fontsize=8, color=TEXT)
ax_u5.set_ylim([-0.3, 1.5])
sax(ax_u5, f'Equation Similarity vs OLS (UTS)\nMean={mean_sim_u:.4f}', '', 'Similarity')

ax_u6 = fig3.add_subplot(gs3[1, 2])
ax_u6.plot(df_sens_u['pct_err'], df_sens_u['mean_sim'], 's-',
           color=C2, lw=2, ms=5, label='UTS sim')
ax_u6.plot(df_sens_u['pct_err'], df_sens_u['sindy_p'], '^--',
           color=C3, lw=1.5, ms=4, label='SINDy p')
ax_u6.axhline(p_ro, color=C3, ls=':', lw=1, alpha=0.6,
              label=f'OLS p={p_ro:.4f}')
ax_u6.set_ylim([-0.3, 1.3])
ax_u6.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
sax(ax_u6, 'UTS Noise Sensitivity', 'Noise (% of mean UTS)', 'Value')

plt.savefig(OUT/"sindy_617t_v6_uts.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: sindy_617t_v6_uts.png")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FINAL SUMMARY – ALLOY 617  [v6]")
print("="*65)
print(f"\n  UTS: N={len(UTS)} specimens, SR span={sr_dec_u:.2f} dec, "
      f"cond={cond_u:.1f}")
print(f"  OLS: UTS = {K_ro:.1f} · sr^{m_ro:.5f} · eps^{p_ro:.5f}")
print(fmt_eq(coefs_u, 'log(UTS)'))
print(f"  R²_log={r2_u_log:.4f}  R²_UTS={r2_u_lin:.4f}  sim={mean_sim_u:.4f}")

print(f"\n  Point-wise pipeline:")
print(f"  {'Filter':<8}  {'N':>5}  {'span':>5}  {'cond':>6}  "
      f"{'QR kept':<35}  {'R²_log':>7}  {'m_OLS':>8}  {'m_SINDy':>8}  {'sim':>6}")
print("  " + "-"*105)
for fl in flist_valid:
    r = pw_results[fl]
    m_s = r['coefs'].get('log(sr_inst)', 0)
    print(f"  {fl:<8}  {r['n']:>5}  {r['sr_span']:>5.2f}  "
          f"{r['cond_fin']:>6.1f}  {str(r['FEAT_ACT']):<35}  "
          f"{r['r2_log']:>7.4f}  {r['m_ols']:>8.5f}  {m_s:>8.5f}  "
          f"{r['mean_sim']:>6.4f}")

print(f"\n  Key question: does any filter level recover m ≈ OLS m?")
for fl in flist_valid:
    r = pw_results[fl]
    m_s = r['coefs'].get('log(sr_inst)', 0)
    m_o = r['m_ols']
    match = abs(m_s - m_o) < 0.05
    print(f"  {fl}: SINDy m={m_s:+.5f}  OLS m={m_o:+.5f}  "
          f"{'✓ MATCH' if match else '✗ off by ' + f'{abs(m_s-m_o):.5f}'}")

print(f"\n  Outputs → {OUT}")
for f in ["sindy_617t_v6_filter_comparison.png",
          "sindy_617t_v6_cross_filter.png",
          "sindy_617t_v6_uts.png"]:
    print(f"    {f}")