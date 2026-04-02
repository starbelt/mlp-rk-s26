import numpy as np
import os
import pandas as pd
import shutil

# ============================================================
# Helper: Load .npy or fallback to .csv
# ============================================================

def load_array_with_fallback(path):
    """
    Try loading a .npy file. If it doesn't exist,
    try loading a .csv file with the same name.
    Returns: numpy array with shape (N, 2)
    """

    # Case 1: .npy exists
    if os.path.exists(path):
        return np.load(path)

    # Case 2: try .csv fallback
    csv_path = path.replace(".npy", ".csv")

    if os.path.exists(csv_path):
        print(f"Fallback to CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Try common column formats
        if df.shape[1] >= 2:
            return df.iloc[:, :2].to_numpy()
        else:
            raise ValueError(f"CSV file does not have at least 2 columns: {csv_path}")

    # Case 3: nothing found
    raise FileNotFoundError(f"Neither .npy nor .csv found for: {path}")

# ============================================================
# Setup Output Directory
# ============================================================

RK_VALUE = 4  # Change this if you want to compare RK2 or RK3 instead of RK1
# Remove old outputs folder if it exists (clean slate)
if os.path.exists(f"RK{RK_VALUE}_outputs"):
    shutil.rmtree(f"RK{RK_VALUE}_outputs")

# Create fresh outputs directory
os.makedirs(f"RK{RK_VALUE}_outputs", exist_ok=True)

# ============================================================
# Load Configuration Data
# ============================================================

# Read RK configuration CSV (contains dt_s and multiplier info)
cfg = pd.read_csv("../01-gen-rk-data/rk_configs.csv")

# Total number of simulation runs
num_runs = len(cfg)

# Extract multiplier and timestep arrays for quick indexing
multipliers = cfg["multiplier"].astype(int).values
rk_dt = cfg["dt_s"].values

# Group index for naming folders (000_cfg, 001_cfg, ...)
group_idx = 0

# ============================================================
# Loop Over Groups (13 runs per configuration)
# ============================================================

# Each group corresponds to 13 dt_s sweeps from one base config
for base_k in range(1, num_runs + 1, 13):

    # Create group folder (e.g., outputs/000_cfg)
    group_name = f"{group_idx:03d}_cfg"
    group_dir = os.path.join(f"RK{RK_VALUE}_outputs", group_name)
    os.makedirs(group_dir, exist_ok=True)

    # Path to the "base" RK output (smallest dt_s in group)
    base_path = f"../02-run-rk/RK{RK_VALUE}/output/r{base_k:02d}/log-buff-v.npy"

    # Skip group if base file is missing
    try:
        base_arr = load_array_with_fallback(base_path)
    except FileNotFoundError:
        print(f"Skipping missing base file: {base_path}")
        group_idx += 1
        continue

    # Load base simulation results
    base_t = base_arr[:, 0]  # time
    base_v = base_arr[:, 1]  # buffer voltage

    # Get base multiplier and timestep
    base_multiplier = multipliers[base_k - 1]
    base_dt = rk_dt[base_k - 1]

    # ========================================================
    # Compare Base Run Against Other Runs in Group
    # ========================================================

    # Loop over the 13 runs in this group
    for offset in range(1, 14):

        # Compute index of comparison run
        k = base_k + (offset - 1)

        # Stop if we exceed total runs
        if k > num_runs:
            break

        # Skip comparing base run with itself
        if k == base_k:
            continue

        # Path to comparison run output
        next_path = f"../02-run-rk/RK{RK_VALUE}/output/r{k:02d}/log-buff-v.npy"

        # Skip if comparison file is missing

        try:
            next_arr = load_array_with_fallback(next_path)
        except FileNotFoundError:
            print(f"Skipping missing compare file: {next_path}")
            continue

        next_t = next_arr[:, 0]
        next_v = next_arr[:, 1]

        # Get multiplier and timestep for comparison run
        next_multiplier = multipliers[k - 1]
        next_dt = rk_dt[k - 1]

        # ====================================================
        # Ensure Time Alignment via Integer Stride
        # ====================================================

        # Only compare runs where next_multiplier is an integer multiple
        # of base_multiplier (ensures exact time alignment)
        if next_multiplier % base_multiplier != 0:
            continue

        # Compute stride for downsampling base signal
        stride = next_multiplier // base_multiplier

        # Downsample base run to match coarser timestep
        base_t_samp = base_t[::stride]
        base_v_samp = base_v[::stride]

        # ====================================================
        # Trim Arrays to Equal Length
        # ====================================================

        # Ensure all arrays have the same length before comparison
        n = min(len(base_t_samp), len(next_t), len(base_v_samp), len(next_v))

        base_t_samp = base_t_samp[:n]
        base_v_samp = base_v_samp[:n]

        next_t = next_t[:n]
        next_v = next_v[:n]

        # ====================================================
        # Compute Error
        # ====================================================

        # Absolute error between base and comparison voltages
        abs_err = np.abs(base_v_samp - next_v)

        # Combine time and error into output array
        out = np.column_stack([next_t, abs_err])

        # ====================================================
        # Save Results
        # ====================================================

        # Output filename encodes dt_s comparison
        out_name = f"RK{RK_VALUE}_{base_dt:.6g}_vs_{next_dt:.6g}.npy"

        # Save error data to group directory
        np.save(os.path.join(group_dir, out_name), out)

    # Move to next group
    group_idx += 1

# ============================================================
# Done
# ============================================================

print("Done.")