import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 1: Define input/output directories
# ============================================================
OUT_DIR = "../03-rk-accuracy/outputs"
MLP_DIR = "../04-mlp-accuracy/mlp-0256-5-results"
PLOT_DIR = "./plots"

# Create output folders if they do not already exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Extract MLP model ID from folder name
# Example:
#   "../04-mlp-accuracy/mlp-0256-5-results" -> "0256-5"
mlp_name = os.path.basename(MLP_DIR)
mlp_id = mlp_name.replace("mlp-", "").replace("-results", "")


# ============================================================
# SECTION 2: Helper function to load RK output files
# Supports both .npy and .csv formats
# ============================================================
def load_rk_output(path):
    ext = os.path.splitext(path)[1].lower()

    # If the file is a NumPy array, assume:
    # column 0 = time
    # column 1 = voltage difference / error
    if ext == ".npy":
        arr = np.load(path)
        return arr[:, 0], arr[:, 1]

    # If the file is a CSV, read named columns
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df["Time (s)"].to_numpy(), df["Voltage Difference (V)"].to_numpy()

    # Reject unsupported file types
    else:
        raise ValueError(f"Unsupported file type: {path}")


# ============================================================
# SECTION 3: Find all configuration folders
# These are expected to look like:
#   000_cfg, 001_cfg, 002_cfg, ...
# ============================================================
cfg_dirs = sorted(glob.glob(os.path.join(OUT_DIR, "*_cfg")))

if not cfg_dirs:
    raise FileNotFoundError(f"No *_cfg folders found in {OUT_DIR}")

total_plots = 0


# ============================================================
# SECTION 4: Loop through each configuration folder
# and generate one combined RK vs MLP plot
# ============================================================
for cfg_dir in cfg_dirs:
    # Extract folder name and numeric config index
    # Example:
    #   cfg_name = "000_cfg"
    #   cfg_idx  = "000"
    cfg_name = os.path.basename(cfg_dir)
    cfg_idx = cfg_name.split("_")[0]

    # ------------------------------------------------------------
    # SECTION 4A: Load corresponding MLP error file
    # Example:
    #   000_cfg -> 000_error.npy
    # ------------------------------------------------------------
    mlp_path = os.path.join(MLP_DIR, f"{cfg_idx}_error.npy")
    if not os.path.exists(mlp_path):
        print(f"Skipping {cfg_name}: missing MLP file {mlp_path}")
        continue

    mlp_data = np.load(mlp_path)
    mlp_t = mlp_data[:, 0]
    mlp_err = mlp_data[:, 1]

    # ------------------------------------------------------------
    # SECTION 4B: Find all RK result files inside this cfg folder
    # Accept both .npy and .csv files
    # ------------------------------------------------------------
    rk_files = sorted(
        glob.glob(os.path.join(cfg_dir, "*.npy")) +
        glob.glob(os.path.join(cfg_dir, "*.csv"))
    )

    if not rk_files:
        print(f"Skipping empty folder: {cfg_name}")
        continue

    print(f"\nProcessing {cfg_name} ({len(rk_files)} files)")
    print(mlp_path)

    # ============================================================
    # SECTION 4C: Map cfg index to the correct RK run folder
    # so we can load log-states.npy and determine x-axis limits
    #
    # Example:
    #   cfg_idx = "000" -> cfg_num = 0
    #   runs_per_cfg = 13
    #   rk_base = 0*13 + 1 = 1
    #   state_path = "../02-run-rk/RK1/r01/log-states.npy"
    # ============================================================
    cfg_num = int(cfg_idx)

    runs_per_cfg = 13
    rk_base = cfg_num * runs_per_cfg + 1

    state_path = f"../02-run-rk/RK1/r{rk_base:02d}/log-states.npy"

    if not os.path.exists(state_path):
        print(f"Skipping {cfg_name}: missing {state_path}")
        continue

    # ------------------------------------------------------------
    # SECTION 4D: Load state data and extract VHI / VLO times
    # These values are used to set the x-axis range
    # ------------------------------------------------------------
    data = np.load(state_path)

    # Assumes:
    #   x[0] = time/value
    #   x[1] = state label
    values = np.array([x[0] for x in data])
    labels = np.array([x[1] for x in data])

    # Find all times labeled VHI and VLO
    vhi_vals = values[labels == 'VHI']
    vlo_vals = values[labels == 'VLO']

    if len(vhi_vals) == 0 or len(vlo_vals) == 0:
        print(f"Skipping {cfg_name}: missing VHI or VLO in log-states.npy")
        continue

    # Use the first VHI and first VLO event as x-axis bounds
    x_start = vhi_vals[0]
    x_end = vlo_vals[0]

    # ------------------------------------------------------------
    # SECTION 4E: Create one figure for this configuration
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------
    # SECTION 4F: Plot all RK error curves for this config
    # ------------------------------------------------------------
    for rk_path in rk_files:
        rk_t, rk_err = load_rk_output(rk_path)

        # Clean up file name for legend display
        stem = os.path.splitext(os.path.basename(rk_path))[0]
        stem_clean = stem.replace("_", " ")

        ax.plot(rk_t, rk_err, label=stem_clean)

    # ------------------------------------------------------------
    # SECTION 4G: Plot MLP error curve
    # ------------------------------------------------------------
    ax.plot(mlp_t, mlp_err, label=f"MLP ({cfg_idx})", linewidth=2)

    # ------------------------------------------------------------
    # SECTION 4H: Format plot
    # ------------------------------------------------------------
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Error| (V)")
    ax.set_title(f"RK vs MLP ({mlp_id}, {cfg_idx})")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------
    # SECTION 4I: Save plot to file
    # ------------------------------------------------------------
    out_png = os.path.join(
        PLOT_DIR,
        f"{cfg_name}_all_rk_vs_mlp_{mlp_id}.png"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    total_plots += 1


# ============================================================
# SECTION 5: Print summary
# ============================================================
print(f"\nSaved {total_plots} plots to {PLOT_DIR}/")