import pandas as pd
import numpy as np

# ============================================================
# Load Input Data
# ============================================================

# Read the input CSV containing base configurations
data = pd.read_csv("./test.csv")

# ============================================================
# Define Target Time Resolution
# ============================================================

# Desired average timestep across all simulations
DT_S_AVG = 0.1

# Find the maximum simulation duration in the dataset
max_dur_s = data["dur_s"].max()

# Compute the maximum number of points so all configs align
# (ensures consistent number of timesteps across configs)
max_points = int(max_dur_s / DT_S_AVG)

# Adjust each row’s dt_s so that all runs have the same number of points
data["dt_s"] = data["dur_s"] / max_points

# ============================================================
# Generate dt_s Sweep (13 values per configuration)
# ============================================================

rows = []  # will store expanded configurations

# Iterate through each original configuration
for _, row in data.iterrows():

    # Base timestep for this configuration
    base_dt = float(row["dt_s"])

    # Maximum multiplier so that dt_s * c ≈ 1 second
    c = (1 / base_dt)

    # Generate 13 evenly spaced multipliers between 1 and c
    multipliers = np.linspace(1, c, 13)

    # Create 13 new rows (one per multiplier)
    for i, k in enumerate(multipliers):

        # Copy original row so we don’t overwrite it
        new_row = row.copy()

        # Scale timestep using multiplier
        dt_val = base_dt * k

        # Only round the LAST value in each group of 13
        # (ensures clean endpoint like dt_s ≈ 1.0)
        if i == 12:
            dt_val = round(dt_val, 1)
        
        # SET Q0 to FULL CHARGE
        new_row["q0_c"] = new_row["c_f"] * new_row["vhi"]

        # Store updated timestep and multiplier
        new_row["dt_s"] = dt_val
        new_row["multiplier"] = k

        # Append to list of new configurations
        rows.append(new_row)

# ============================================================
# Save Expanded Dataset
# ============================================================

# Convert list of rows into a DataFrame
rows = pd.DataFrame(rows)

# Write new configuration file for RK simulations
rows.to_csv("./rk_configs.csv", index=False)