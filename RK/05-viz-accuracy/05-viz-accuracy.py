import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# SECTION 1: Define input/output directories
# ============================================================
RK_NUM = 1  # Change this to 2 or 3 if you want to compare RK2 or RK3 instead of RK1
OUT_DIR = f"../03-rk-accuracy/RK{RK_NUM}_outputs"
MLP_BASE_DIR = "../04-mlp-accuracy"
PLOT_DIR = f"./RK{RK_NUM}_plots"

# Create output folders if they do not already exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# SECTION 1A: Find all MLP result folders
# Expected:
#   mlp-0032-2-results
#   mlp-0032-3-results
#   ...
#   mlp-1024-7-results
# ============================================================
mlp_dirs = sorted(glob.glob(os.path.join(MLP_BASE_DIR, "mlp-*-results")))

if not mlp_dirs:
    raise FileNotFoundError(f"No MLP result folders found in {MLP_BASE_DIR}")

# Store (mlp_id, mlp_dir)
mlp_models = []
for mlp_dir in mlp_dirs:
    mlp_name = os.path.basename(mlp_dir)
    mlp_id = mlp_name.replace("mlp-", "").replace("-results", "")
    mlp_models.append((mlp_id, mlp_dir))

print(f"Found {len(mlp_models)} MLP result folders")
for mlp_id, _ in mlp_models:
    print(f"  {mlp_id}")


# ============================================================
# SECTION 1B: Solar charging helper
# ============================================================
def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    # I_sun = (Irr * Area * eff) / Vmp
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0


# ============================================================
# SECTION 2: Path helper
# Try .npy first, then fallback to .csv with same base name
# ============================================================
def resolve_with_csv_fallback(path):
    """
    Return the existing file path.
    Priority:
      1. original path
      2. same path but with .csv extension
    Returns None if neither exists.
    """
    if os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    csv_path = root + ".csv"

    if os.path.exists(csv_path):
        print(f"Fallback to CSV: {csv_path}")
        return csv_path

    return None


# ============================================================
# SECTION 3: Helper function to load RK output files
# Supports both .npy and .csv formats
# ============================================================
def load_rk_output(path):
    resolved_path = resolve_with_csv_fallback(path)

    if resolved_path is None:
        raise FileNotFoundError(f"Neither .npy nor .csv found for: {path}")

    ext = os.path.splitext(resolved_path)[1].lower()

    # If the file is a NumPy array, assume:
    # column 0 = time
    # column 1 = voltage difference / error
    if ext == ".npy":
        arr = np.load(resolved_path)
        return arr[:, 0], arr[:, 1]

    # If the file is a CSV, try named columns first,
    # otherwise fallback to first two columns
    elif ext == ".csv":
        df = pd.read_csv(resolved_path)

        if "Time (s)" in df.columns and "Voltage Difference (V)" in df.columns:
            return df["Time (s)"].to_numpy(), df["Voltage Difference (V)"].to_numpy()

        elif len(df.columns) >= 2:
            return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()

        else:
            raise ValueError(f"CSV file does not have at least 2 columns: {resolved_path}")

    else:
        raise ValueError(f"Unsupported file type: {resolved_path}")


# ============================================================
# SECTION 4: Helper to load state files (.npy or .csv)
# Expects two columns: time and state label
# ============================================================
def load_state_output(path):
    resolved_path = resolve_with_csv_fallback(path)

    if resolved_path is None:
        raise FileNotFoundError(f"Neither .npy nor .csv found for: {path}")

    ext = os.path.splitext(resolved_path)[1].lower()

    if ext == ".npy":
        data = np.load(resolved_path, allow_pickle=True)

        # structured array case
        if getattr(data, "dtype", None) is not None and data.dtype.names is not None:
            time_field = data.dtype.names[0]
            state_field = data.dtype.names[1]
            values = np.array(data[time_field])
            labels = np.array(data[state_field]).astype(str)
            return values, labels

        # plain 2-column array case
        data = np.asarray(data)
        return data[:, 0], data[:, 1].astype(str)

    elif ext == ".csv":
        df = pd.read_csv(resolved_path)

        # Try common state column names first
        possible_time_cols = ["t_s", "time", "Time (s)", df.columns[0]]
        possible_state_cols = ["state", "State", "label", df.columns[1] if len(df.columns) > 1 else None]

        time_col = next((c for c in possible_time_cols if c in df.columns), None)
        state_col = next((c for c in possible_state_cols if c in df.columns), None)

        if time_col is None or state_col is None:
            if len(df.columns) >= 2:
                return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].astype(str).to_numpy()
            raise ValueError(f"CSV state file does not have enough columns: {resolved_path}")

        return df[time_col].to_numpy(), df[state_col].astype(str).to_numpy()

    else:
        raise ValueError(f"Unsupported file type: {resolved_path}")


# ============================================================
# SECTION 5: Find all configuration folders
# These are expected to look like:
#   000_cfg, 001_cfg, 002_cfg, ...
# ============================================================
cfg_dirs = sorted(glob.glob(os.path.join(OUT_DIR, "*_cfg")))

if not cfg_dirs:
    raise FileNotFoundError(f"No *_cfg folders found in {OUT_DIR}")

total_plots = 0


# ============================================================
# SECTION 6: Loop through each configuration folder
# and generate one combined RK vs ALL MLP plot
# ============================================================
for cfg_dir in cfg_dirs:
    # Extract folder name and numeric config index
    cfg_name = os.path.basename(cfg_dir)
    cfg_idx = cfg_name.split("_")[0]

    # ------------------------------------------------------------
    # SECTION 6A: Find all RK result files inside this cfg folder
    # Accept both .npy and .csv files
    # ------------------------------------------------------------
    rk_files = sorted(
        glob.glob(os.path.join(cfg_dir, "*.npy")) +
        glob.glob(os.path.join(cfg_dir, "*.csv"))
    )

    if not rk_files:
        print(f"Skipping empty folder: {cfg_name}")
        continue

    print(f"\nProcessing {cfg_name} ({len(rk_files)} RK files)")

    # ------------------------------------------------------------
    # SECTION 6B: Map cfg index to the correct RK run folder
    # ------------------------------------------------------------
    cfg_num = int(cfg_idx)

    runs_per_cfg = 13
    rk_base = cfg_num * runs_per_cfg + 1

    state_path = f"../02-run-rk/RK{RK_NUM}/output/r{rk_base:02d}/log-states.npy"
    resolved_state_path = resolve_with_csv_fallback(state_path)

    if resolved_state_path is None:
        print(f"Skipping {cfg_name}: missing {state_path} and CSV fallback")
        continue

    # ------------------------------------------------------------
    # SECTION 6C: Load state data and extract VHI / VLO times
    # ------------------------------------------------------------
    try:
        values, labels = load_state_output(state_path)
    except Exception as e:
        print(f"Skipping {cfg_name}: could not load state file ({e})")
        continue

    vhi_vals = values[labels == 'VHI']
    vlo_vals = values[labels == 'VLO']

    if len(vhi_vals) == 0 or len(vlo_vals) == 0:
        print(f"Skipping {cfg_name}: missing VHI or VLO in state file")
        continue

    # Use the first VLO event as the x-axis upper bound
    x_start = vhi_vals[0]
    x_end = vlo_vals[0]

    # ------------------------------------------------------------
    # SECTION 6D: Load config values for charge-time shift
    # ------------------------------------------------------------
    cfg_path = "../01-gen-rk-data/test.csv"
    cfgs = pd.read_csv(cfg_path)
    row = cfgs.iloc[cfg_num]

    c_f = row['c_f']
    vhi = row['vhi']
    vlo = row['vlo']
    irr_w_m2 = 1366.1
    sa_m2 = row['sa_m2']
    eff = row['eff']
    vmp = row['vmp']

    i_charge = calc_solar_current(irr_w_m2, sa_m2, eff, vmp)

    if i_charge <= 0:
        print(f"Skipping {cfg_name}: computed charging current is <= 0")
        continue

    t_charge = c_f * (vhi - vlo) / i_charge

    # ------------------------------------------------------------
    # SECTION 6E: Create one figure for this configuration
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------
    # SECTION 6F: Plot all RK error curves
    # ------------------------------------------------------------
    rk_files_sorted = sorted(rk_files)
    rk_cmap = cm.viridis
    N_rk = len(rk_files_sorted)
    rk_colors = rk_cmap(np.linspace(0, 1, N_rk))
    linestyles = ["-", "--", "-.", ":"]

    base_dt_for_title = None

    for i, rk_path in enumerate(rk_files_sorted):
        try:
            rk_t, rk_err = load_rk_output(rk_path)
        except Exception as e:
            print(f"Skipping RK file {rk_path}: {e}")
            continue

        stem = os.path.splitext(os.path.basename(rk_path))[0]
        parts = stem.split("_")

        try:
            base_dt = float(parts[1])
            next_dt = float(parts[3])
        except (IndexError, ValueError):
            print(f"Skipping badly named RK file: {rk_path}")
            continue

        if base_dt_for_title is None:
            base_dt_for_title = base_dt

        ax.plot(
            rk_t,
            rk_err,
            color=rk_colors[i],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.5,
            label=f"RK dt={next_dt}"
        )

    # ------------------------------------------------------------
    # SECTION 6G: Plot ALL MLP error curves
    # ------------------------------------------------------------
    mlp_cmap = cm.plasma
    N_mlp = len(mlp_models)
    mlp_colors = mlp_cmap(np.linspace(0, 1, N_mlp))

    plotted_mlp_count = 0

    for j, (mlp_id, mlp_dir) in enumerate(mlp_models):
        mlp_path = os.path.join(mlp_dir, f"{cfg_idx}_error.npy")

        if not os.path.exists(mlp_path):
            print(f"Missing MLP file for {mlp_id}: {mlp_path}")
            continue

        try:
            mlp_data = np.load(mlp_path)
            mlp_t = mlp_data[:, 0]
            mlp_err = mlp_data[:, 1]
        except Exception as e:
            print(f"Skipping MLP file {mlp_path}: {e}")
            continue

        mlp_t_shifted = mlp_t - t_charge
        mask = mlp_t_shifted >= 0
        mlp_t_plot = mlp_t_shifted[mask]
        mlp_err_plot = mlp_err[mask]

        if len(mlp_t_plot) == 0:
            print(f"Skipping MLP {mlp_id} for {cfg_name}: no data after shift")
            continue

        ax.plot(
            mlp_t_plot,
            mlp_err_plot,
            color=mlp_colors[j],
            linewidth=2,
            label=f"MLP ({mlp_id})"
        )

        plotted_mlp_count += 1

    if plotted_mlp_count == 0:
        print(f"No MLP curves plotted for {cfg_name}")

    # ------------------------------------------------------------
    # SECTION 6H: Format plot
    # ------------------------------------------------------------
    ax.set_xlim(0, x_end)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Error| (V)")
    ax.set_title(f"RK{RK_NUM} (DT {base_dt_for_title}) vs All MLPs ({cfg_idx})")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    # ------------------------------------------------------------
    # SECTION 6I: Save plot to file
    # ------------------------------------------------------------
    out_png = os.path.join(
        PLOT_DIR,
        f"{cfg_name}_RK{RK_NUM}_ALL_MLPs.png"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    total_plots += 1


# ============================================================
# SECTION 7: Print summary
# ============================================================
print(f"\nSaved {total_plots} plots to {PLOT_DIR}/")