from pathlib import Path
import numpy as np
import pandas as pd

from creep_rupture_sindy import (
    standardize_columns,
    fit_sindy_lmp_model,
    optimize_C_parameter,
    print_model_summary,
    plot_c_optimization,
    plot_model_performance,
)

# ------------------------------
# Multi-file loader
# ------------------------------
def load_alloy617_from_excels(
    folder_or_files,
    sheet_name="Rupture",
    file_glob="*.csv",
    allow_all_sheets_fallback=True,
):
    """
    Loads one or many Excel files, standardizes columns (T, sigma, t_r),
    concatenates into a single DataFrame.

    Parameters
    ----------
    folder_or_files : str | Path | list[str|Path]
        Either a folder path (we'll glob *.xlsx) or an explicit list of files.
    sheet_name : str | int | None
        Target sheet name. If not found and allow_all_sheets_fallback=True,
        we will try all sheets and keep those that match required columns.
    file_glob : str
        Pattern used when folder_or_files is a folder.
    allow_all_sheets_fallback : bool
        If True, tries all sheets when the requested sheet isn't present.

    Returns
    -------
    df_all : pd.DataFrame
        Concatenated data with columns: T, sigma, t_r plus 'source_file' and 'source_sheet'.
    """

    # Resolve list of files
    if isinstance(folder_or_files, (str, Path)):
        p = Path(folder_or_files)
        if p.is_dir():
            files = sorted(p.glob(file_glob))
        else:
            files = [p]
    else:
        files = [Path(f) for f in folder_or_files]

    if not files:
        raise FileNotFoundError("No Excel files found. Check your folder/path/glob.")

    frames = []
    for f in files:
        try:
            xls = pd.ExcelFile(f)
        except Exception as e:
            print(f"[SKIP] Could not open {f.name}: {e}")
            continue

        def try_sheet(sh):
            df = pd.read_excel(f, sheet_name=sh)
            df = standardize_columns(df)  # expects T, sigma, t_r after mapping
            df["source_file"] = f.name
            df["source_sheet"] = str(sh)
            return df

        # First try the requested sheet
        if sheet_name is not None and sheet_name in xls.sheet_names:
            try:
                frames.append(try_sheet(sheet_name))
                continue
            except Exception as e:
                print(f"[WARN] {f.name} sheet '{sheet_name}' failed standardization: {e}")

        # Optional fallback: try all sheets and keep those that work
        if allow_all_sheets_fallback:
            kept_any = False
            for sh in xls.sheet_names:
                try:
                    frames.append(try_sheet(sh))
                    kept_any = True
                except Exception:
                    pass
            if not kept_any:
                print(f"[SKIP] No usable sheets in {f.name}. Sheets: {xls.sheet_names}")
        else:
            print(f"[SKIP] Sheet '{sheet_name}' not found in {f.name}. Sheets: {xls.sheet_names}")

    if not frames:
        raise ValueError("No usable data found across the provided Excel files.")

    df_all = pd.concat(frames, ignore_index=True)

    # Clean up
    df_all = df_all.dropna(subset=["T", "sigma", "t_r"])
    df_all = df_all[(df_all["T"] > 0) & (df_all["sigma"] > 0) & (df_all["t_r"] > 0)]

    return df_all


# ------------------------------
# Main workflow
# ------------------------------
def main():
    # Point this at your Alloy 617 Excel folder (or provide a list of files)
    data_path = Path("Alloy617_excels")  # <-- change to your folder

    # 1) Load + merge multiple files
    df = load_alloy617_from_excels(
        data_path,
        sheet_name="Rupture",      # change if your sheet name differs
        file_glob="*.xlsx",
        allow_all_sheets_fallback=True,
    )
    print(f"\nLoaded {len(df)} rows from {df['source_file'].nunique()} files.")
    print("Columns:", list(df.columns))

    # Optional: save the merged dataset
    df.to_csv("alloy617_rupture_merged.csv", index=False)
    print("Saved merged data: alloy617_rupture_merged.csv")

    # 2) Fit a baseline model
    model = fit_sindy_lmp_model(df, C=20.0, poly_degree=3, threshold=0.01)
    print_model_summary(model)

    # 3) Optional: optimize C
    C_grid = np.linspace(18, 22, 21)
    best_model, results = optimize_C_parameter(
        df, C_grid=C_grid, poly_degree=3, threshold=0.01
    )
    print("\nBest model after C optimization:")
    print_model_summary(best_model)
    plot_c_optimization(results)

    # 4) Diagnostics
    plot_model_performance(df, best_model, save_path="alloy617_diagnostics.png")
    print("Saved diagnostics: alloy617_diagnostics.png")


if __name__ == "__main__":
    main()
