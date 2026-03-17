import numpy as np
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# Read CSV WITH header row
cfg = pd.read_csv("./rk_configs.csv")

num_runs = len(cfg)
group_idx = 0

multipliers = cfg["sweep"].astype(int).values
rk_dt = cfg["dt_s"].values

group_idx = 0

for base_k in range(1, num_runs + 1, 13):

    group_name = f"{group_idx:03d}_cfg"
    group_dir = os.path.join("outputs", group_name)
    os.makedirs(group_dir, exist_ok=True)

    base_path = f"./RK1/r{base_k:02d}/log-buff-v.npy"

    if not os.path.exists(base_path):
        print(f"Skipping missing base file: {base_path}")
        group_idx += 1
        continue

    base_arr = np.load(base_path)
    base_t = base_arr[:, 0]
    base_v = base_arr[:, 1]

    base_multiplier = multipliers[base_k - 1]
    base_dt = rk_dt[base_k - 1]

    for offset in range(1, 14):

        k = base_k + (offset - 1)

        if k > num_runs:
            break

        if k == base_k:
            continue

        next_path = f"./RK1/r{k:02d}/log-buff-v.npy"

        if not os.path.exists(next_path):
            print(f"Skipping missing compare file: {next_path}")
            continue

        next_arr = np.load(next_path)
        next_t = next_arr[:, 0]
        next_v = next_arr[:, 1]

        next_multiplier = multipliers[k - 1]
        next_dt = rk_dt[k - 1]

        if next_multiplier % base_multiplier != 0:
            continue

        stride = next_multiplier // base_multiplier

        base_t_samp = base_t[::stride]
        base_v_samp = base_v[::stride]

        n = min(len(base_t_samp), len(next_t), len(base_v_samp), len(next_v))

        base_t_samp = base_t_samp[:n]
        base_v_samp = base_v_samp[:n]

        next_t = next_t[:n]
        next_v = next_v[:n]

        abs_err = np.abs(base_v_samp - next_v)

        out = np.column_stack([next_t, abs_err])

        out_name = f"RK1_{base_dt:.6g}_vs_{next_dt:.6g}.npy"

        np.save(os.path.join(group_dir, out_name), out)

    group_idx += 1

print("Done.")