import pandas as pd
import numpy as np

data = pd.read_csv("./test.csv")

DT_S_AVG = 0.1
max_dur_s = data["dur_s"].max()
max_points = int(max_dur_s / DT_S_AVG)

data["dt_s"] = data["dur_s"] / max_points

rows = []

for _, row in data.iterrows():
    base_dt = float(row["dt_s"])
    c = (1 / base_dt)

    multipliers = np.linspace(1, c, 13, dtype=int)

    for i, k in enumerate(multipliers):
        new_row = row.copy()

        dt_val = base_dt * k

        # Only round the LAST one in each group of 13
        if i == 12:
            dt_val = round(dt_val, 1)

        new_row["dt_s"] = dt_val
        new_row["multiplier"] = k

        rows.append(new_row)

rows = pd.DataFrame(rows)

rows.to_csv("./rk_configs.csv", index=False)