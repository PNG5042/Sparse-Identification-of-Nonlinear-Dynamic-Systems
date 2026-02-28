import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CÁC TÊN CỘT CHÍNH
# =========================================================
SPECIMEN_COL = "Specimen_Name"
RATE_COL     = "Nominal_Strain_Rate"
STRAIN_COL   = "Strain_percent"
STRESS_COL   = "Stress_MPa"

DROP_NEGATIVE_STRAIN   = True
MIN_POINTS_PER_SEGMENT = 50      # bỏ các segment quá ngắn
GAP_TOL_PCT            = 0.02    # %strain cho phép giữa 2 segment
AGG_METHOD             = "median"  # median hoặc mean để gộp m

# =========================================================
# LOAD DATA
# =========================================================
detail = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_DETAIL_DATA.csv")
spec   = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\Sparse-Identification-of-Nonlinear-Dynamic-Systems\Project_base_files\Alloy 617 Test\Strain-rate-sensitivity\SGIHX_A5_SPECIMEN_LIST.csv")

# ---- kiểm tra cột bắt buộc ----
for col in [SPECIMEN_COL, RATE_COL, STRAIN_COL, STRESS_COL]:
    if col not in detail.columns:
        raise KeyError(f"DETAIL file thiếu cột {col}. Available: {list(detail.columns)}")

if SPECIMEN_COL not in spec.columns:
    raise KeyError(f"SPECIMEN_LIST file thiếu cột {SPECIMEN_COL}. Available: {list(spec.columns)}")

# ---- ép kiểu & lọc DETAIL ----
detail = detail[[SPECIMEN_COL, RATE_COL, STRAIN_COL, STRESS_COL]].copy()
detail[RATE_COL]   = pd.to_numeric(detail[RATE_COL], errors="coerce")
detail[STRAIN_COL] = pd.to_numeric(detail[STRAIN_COL], errors="coerce")
detail[STRESS_COL] = pd.to_numeric(detail[STRESS_COL], errors="coerce")
detail = detail.dropna(subset=[SPECIMEN_COL, RATE_COL, STRAIN_COL, STRESS_COL])

# log() cần rate>0 & stress>0
detail = detail[(detail[RATE_COL] > 0) & (detail[STRESS_COL] > 0)].copy()
if DROP_NEGATIVE_STRAIN:
    detail = detail[detail[STRAIN_COL] >= 0].copy()

print("=== DETAIL summary ===")
print(f"Rows: {len(detail):,}")
print(f"Specimens in DETAIL: {detail[SPECIMEN_COL].nunique()}")
print(f"Strain range (%): {detail[STRAIN_COL].min():.6g} – {detail[STRAIN_COL].max():.6g}")
print(f"Strain rates: {sorted(detail[RATE_COL].unique())}")

print("\n=== SPECIMEN_LIST summary ===")
print(f"Rows: {len(spec):,}")
print(f"Specimens in SPECIMEN_LIST: {spec[SPECIMEN_COL].nunique()}")
print("Some metadata columns:", [c for c in spec.columns if c != SPECIMEN_COL][:10])

# =========================================================
# HÀM PHỤ
# =========================================================
def _interp(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Nội suy tuyến tính y(xq) với x đã sort."""
    if x.size < 2:
        raise ValueError("Not enough points for interpolation.")
    if xq < x.min() or xq > x.max():
        raise ValueError("Query strain outside segment range.")
    return float(np.interp(xq, x, y))

def build_segments_for_specimen(df_one: pd.DataFrame) -> pd.DataFrame:
    """
    Cho 1 specimen, chia thành các segment theo strain-rate.
    Mỗi segment: (rate, smin, smax, x-array, y-array)
    """
    segs = []
    for rate, g in df_one.groupby(RATE_COL, sort=False):
        g2 = g.sort_values(STRAIN_COL)
        x = g2[STRAIN_COL].to_numpy(float)
        y = g2[STRESS_COL].to_numpy(float)

        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]

        if len(x_u) < MIN_POINTS_PER_SEGMENT:
            continue

        segs.append({
            "rate": float(rate),
            "smin": float(x_u.min()),
            "smax": float(x_u.max()),
            "x": x_u,
            "y": y_u
        })

    seg_df = pd.DataFrame(segs)
    if seg_df.empty:
        return seg_df
    # sắp theo thứ tự strain (0→ lớn)
    return seg_df.sort_values("smin").reset_index(drop=True)

def choose_jump_strain(segA, segB) -> float:
    """
    Chọn strain để so sánh trước/sau jump.
    - Nếu 2 segment overlap: lấy midpoint overlap
    - Nếu không overlap nhưng gap nhỏ: lấy biên gần
    """
    a_min, a_max = segA["smin"], segA["smax"]
    b_min, b_max = segB["smin"], segB["smax"]

    ov_min = max(a_min, b_min)
    ov_max = min(a_max, b_max)
    if ov_min <= ov_max:        # có overlap
        return 0.5 * (ov_min + ov_max)

    gap = b_min - a_max
    if gap > GAP_TOL_PCT:
        raise ValueError(f"Gap too large between segments: {gap:.6g}%")

    s = a_max
    if not (b_min <= s <= b_max):
        s = b_min
    return float(s)

def compute_m_jumps_for_specimen(df_one: pd.DataFrame) -> pd.DataFrame:
    """
    Tính m cho từng jump (giữa 2 segment kề nhau) của 1 specimen:
        m = ln(sigma2/sigma1)/ln(rate2/rate1)
    """
    seg_df = build_segments_for_specimen(df_one)
    if len(seg_df) < 2:
        return pd.DataFrame()

    rows = []
    this_name = df_one[SPECIMEN_COL].iloc[0]

    for i in range(len(seg_df) - 1):
        A = seg_df.iloc[i]
        B = seg_df.iloc[i + 1]
        try:
            s_jump = choose_jump_strain(A, B)
            sigA = _interp(A["x"], A["y"], s_jump)
            sigB = _interp(B["x"], B["y"], s_jump)

            m = np.log(sigB / sigA) / np.log(B["rate"] / A["rate"])

            rows.append({
                SPECIMEN_COL: this_name,
                "jump_index": i + 1,
                "strain_jump_pct": s_jump,
                "rate_before": A["rate"],
                "rate_after": B["rate"],
                "sigma_before_MPa": sigA,
                "sigma_after_MPa": sigB,
                "m": float(m)
            })
        except Exception as e:
            rows.append({
                SPECIMEN_COL: this_name,
                "jump_index": i + 1,
                "strain_jump_pct": np.nan,
                "rate_before": A["rate"],
                "rate_after": B["rate"],
                "sigma_before_MPa": np.nan,
                "sigma_after_MPa": np.nan,
                "m": np.nan,
                "note": str(e)
            })

    return pd.DataFrame(rows)

# =========================================================
# TÍNH m CHO TỪNG SPECIMEN, SAU ĐÓ MERGE METADATA
# =========================================================
rows = []
for name, g in detail.groupby(SPECIMEN_COL, sort=False):
    mj = compute_m_jumps_for_specimen(g)
    if not mj.empty:
        rows.append(mj)

m_by_spec = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
m_by_spec = m_by_spec.dropna(subset=["m"]).copy()

print("\n=== m_by_spec (head) ===")
print(m_by_spec.head(10).to_string(index=False))

# ---- MERGE METADATA từ SPECIMEN_LIST ----
meta_cols = [c for c in spec.columns if c != SPECIMEN_COL]
m_with_meta = m_by_spec.merge(
    spec[[SPECIMEN_COL] + meta_cols],
    on=SPECIMEN_COL,
    how="left",
)

print("\n=== m_with_meta (head) ===")
print(m_with_meta.head(10).to_string(index=False))

# =========================================================
# TỔNG HỢP m THEO JUMP + (OPTIONAL) THEO NHIỆT ĐỘ
# =========================================================
group_cols = ["rate_before", "rate_after"]

# nếu trong SPECIMEN_LIST có nhiệt độ, ta thêm nó vào group
temp_col_candidates = ["Nominal_Temperature_C", "Temperature_C"]
temp_cols_found = [c for c in temp_col_candidates if c in m_with_meta.columns]
if temp_cols_found:
    TEMP_COL = temp_cols_found[0]
    group_cols.append(TEMP_COL)
    print(f"\nGrouping by jump + temperature column: {TEMP_COL}")
else:
    TEMP_COL = None
    print("\nNo temperature column found; grouping only by rate_before, rate_after.")

if AGG_METHOD == "median":
    agg = (m_with_meta
           .groupby(group_cols)
           .agg(
               m_median=("m", "median"),
               m_mean=("m", "mean"),
               m_std=("m", "std"),
               n=("m", "count"),
               strain_jump_median=("strain_jump_pct", "median")
           )
           .reset_index()
           .sort_values(group_cols)
           .reset_index(drop=True))
else:
    agg = (m_with_meta
           .groupby(group_cols)
           .agg(
               m_mean=("m", "mean"),
               m_median=("m", "median"),
               m_std=("m", "std"),
               n=("m", "count"),
               strain_jump_median=("strain_jump_pct", "median")
           )
           .reset_index()
           .sort_values(group_cols)
           .reset_index(drop=True))

print("\n=== Aggregated m by jump (and temperature if available) ===")
print(agg.to_string(index=False))

# =========================================================
# PLOTS (m + metadata)
# =========================================================

# 1) m theo strain_jump (từng specimen)
plt.figure(figsize=(7, 5))
plt.scatter(m_with_meta["strain_jump_pct"], m_with_meta["m"])
plt.axhline(0, linestyle="--", linewidth=0.9)
plt.xlabel("Strain at jump (%)")
plt.ylabel("m = ln(σ2/σ1) / ln(ė2/ė1)")
plt.title("Alloy 617 – m per specimen (DETAIL + SPECIMEN_LIST)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) nếu có nhiệt độ: vẽ m vs T (mỗi jump là 1 điểm)
if TEMP_COL is not None:
    plt.figure(figsize=(7, 5))
    plt.scatter(m_with_meta[TEMP_COL], m_with_meta["m"])
    plt.axhline(0, linestyle="--", linewidth=0.9)
    plt.xlabel(f"{TEMP_COL} (°C)")
    plt.ylabel("m")
    plt.title("m vs temperature (per specimen)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3) aggregated m vs strain_jump_median
plt.figure(figsize=(7, 5))
x = agg["strain_jump_median"].to_numpy(float)
y = agg["m_median"].to_numpy(float)
yerr = agg["m_std"].to_numpy(float)
plt.errorbar(x, y, yerr=yerr, fmt="o")
plt.axhline(0, linestyle="--", linewidth=0.9)
plt.xlabel("Median strain at jump (%)")
plt.ylabel("Aggregated m (median ± std)")
plt.title("Aggregated m by jump (DETAIL + SPECIMEN_LIST)")
plt.grid(True)
plt.tight_layout()
plt.show()