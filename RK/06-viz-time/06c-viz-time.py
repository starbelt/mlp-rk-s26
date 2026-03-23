import csv
import numpy as np
import matplotlib.pyplot as plt

FILE1 = "results/timing_results.csv"
FILE2 = "../01-gen-rk-data/rk_configs.csv"

avg_times = []
dt_s_list = []
dur_s_list = []

with open(FILE1, newline="") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader):
        if i >= 13:
            break

        avg_times.append(float(row["avg_total_runtime_s"]))
        dt_s_list.append(float(row["dt_s"]))

with open(FILE2, newline="") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader):
        if i >= 13:
            break

        dur_s_list.append(float(row["dur_s"]))

avg_times = np.array(avg_times)
dt_s_list = np.array(dt_s_list)
dur_s_list = np.array(dur_s_list)

avg_time_slopes = avg_times / dt_s_list

for i in range(len(avg_time_slopes)):
    # Pick one config 
    dt = dt_s_list[i]
    dur = dur_s_list[i]
    slope = avg_time_slopes[i]

    # Create time vector from 0 → dur
    x = np.arange(0, dur, dt)

    # Compute runtime estimate
    y = slope * x

    # Plot
    plt.plot(x, y)
    plt.axhline(y = 1.077911578550e-06)
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Estimated Runtime (s)")
    plt.title("Runtime vs Simulation Time")
    plt.grid()
plt.show()