import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "./outputs"
MLP_DIR = "./mlp-0256-5-results"
PLOT_DIR = "./plots"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_rk_output(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        return arr[:, 0], arr[:, 1]

    elif ext == ".csv":
        df = pd.read_csv(path)
        return df["Time (s)"].to_numpy(), df["Voltage Difference (V)"].to_numpy()

    else:
        raise ValueError(f"Unsupported file type: {path}")

# Loop through all cfg folders (000_cfg, 001_cfg, etc.)

cfg_dirs = sorted(glob.glob(os.path.join(OUT_DIR, "*_cfg")))

if not cfg_dirs:
    raise FileNotFoundError(f"No *_cfg folders found in {OUT_DIR}")

total_plots = 0

for cfg_dir in cfg_dirs:

    cfg_name = os.path.basename(cfg_dir)   # e.g. "000_cfg"

    # Build matching MLP file (000_error.npy, etc.)

    cfg_idx = cfg_name.split("_")[0]

    mlp_path = os.path.join(MLP_DIR, f"{cfg_idx}_error.npy")

    if not os.path.exists(mlp_path):
        print(f"Skipping {cfg_name}: missing MLP file {mlp_path}")
        continue

    # Load MLP data for this group
    mlp_data = np.load(mlp_path)
    mlp_t = mlp_data[:, 0]
    mlp_err = mlp_data[:, 1]

    # Find RK files inside this cfg folder
    
    rk_files = sorted(
        glob.glob(os.path.join(cfg_dir, "*.npy")) +
        glob.glob(os.path.join(cfg_dir, "*.csv"))
    )

    if not rk_files:
        print(f"Skipping empty folder: {cfg_name}")
        continue

    print(f"\nProcessing {cfg_name} ({len(rk_files)} files)")

    # Create plot subdirectory
    plot_subdir = os.path.join(PLOT_DIR, cfg_name)
    os.makedirs(plot_subdir, exist_ok=True)

    
    # Loop through RK files
    
    for rk_path in rk_files:

        rk_t, rk_err = load_rk_output(rk_path)

        stem = os.path.splitext(os.path.basename(rk_path))[0]
        stem_clean = stem.replace("_", " ")
        rk_method = stem.split("_")[0]

        plt.figure()
        plt.plot(rk_t, rk_err, label=stem_clean)
        plt.plot(mlp_t, mlp_err, label=f"MLP ({cfg_idx})")

        plt.xlabel("Time (s)")
        plt.ylabel("|Error| (V)")
        plt.legend()
        plt.title(f"{rk_method} vs MLP ({cfg_name})")

        out_png = os.path.join(
            plot_subdir,
            f"{stem}_vs_mlp_{cfg_idx}.png"
        )

        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()

        total_plots += 1

print(f"\nSaved {total_plots} plots to {PLOT_DIR}/")