import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def plot_rk_envelope(ax, dt_s_list, dur_s_list, avg_time_slopes, color, label, hatch, multiplier=1.0):
    """
    Plot one RK runtime envelope on the given axis.
    """
    slope_max = np.max(avg_time_slopes)
    slope_min = np.min(avg_time_slopes)

    dt = dt_s_list[0]
    dur = dur_s_list[0]

    x = np.arange(0, dur, dt)
    y_max = slope_max * x
    y_min = slope_min * x

    ax.plot(x, y_max * multiplier, alpha=0.7, color=color, label=label)
    ax.plot(x, y_min * multiplier, alpha=0.7, color=color)

    ax.fill_between(
        x, y_min * multiplier, y_max * multiplier,
        facecolor="none",
        edgecolor=color,
        hatch=hatch,
        linewidth=0.0
    )


def plot_mlp_lines(ax, values, color, label, multiplier=1.0):
    """
    Plot a group of horizontal MLP lines on the given axis.
    Only the first line gets a legend label.
    """
    for i, val in enumerate(values):
        ax.axhline(
            y=val * multiplier,
            color=color,
            linestyle="--",
            label=label if i == 0 else None
        )


# ============================================================
# MAIN SCRIPT
# ============================================================

config = 2  # Change this value to plot different configurations (1, 2, 3, ...)
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

FILE1 = "results/RK1_timing_results.csv"
FILE2 = "results/RK2_timing_results.csv"
FILE3 = "results/RK4_timing_results.csv"
FILE4 = "../01-gen-rk-data/rk_configs.csv"

avg_times_rk1 = []
avg_times_rk2 = []
avg_times_rk4 = []

dt_s_list_rk1 = []
dt_s_list_rk2 = []
dt_s_list_rk4 = []

dur_s_list = []

entries = 13 * config
start = max(0, entries - 13)

MLP32 = []
MLP64 = []
MLP128 = []
MLP256 = []
MLP512 = []
MLP1024 = []

# ============================================================
# READ MLP DATA
# ============================================================
for i in range(2, 8):
    FILE32 = f"../../MLP/04-tst-mlp/mlp-0032-results/mlp-0032-{i}/mlp-0032-{i}-tst-summary.csv"
    FILE64 = f"../../MLP/04-tst-mlp/mlp-0064-results/mlp-0064-{i}/mlp-0064-{i}-tst-summary.csv"
    FILE128 = f"../../MLP/04-tst-mlp/mlp-0128-results/mlp-0128-{i}/mlp-0128-{i}-tst-summary.csv"
    FILE256 = f"../../MLP/04-tst-mlp/mlp-0256-results/mlp-0256-{i}/mlp-0256-{i}-tst-summary.csv"
    FILE512 = f"../../MLP/04-tst-mlp/mlp-0512-results/mlp-0512-{i}/mlp-0512-{i}-tst-summary.csv"
    FILE1024 = f"../../MLP/04-tst-mlp/mlp-1024-results/mlp-1024-{i}/mlp-1024-{i}-tst-summary.csv"

    with open(FILE32, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP32.append(float(row["time_per_prediction"]))
                break

    with open(FILE64, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP64.append(float(row["time_per_prediction"]))
                break

    with open(FILE128, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP128.append(float(row["time_per_prediction"]))
                break

    with open(FILE256, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP256.append(float(row["time_per_prediction"]))
                break

    with open(FILE512, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP512.append(float(row["time_per_prediction"]))
                break

    with open(FILE1024, newline="") as f:
        reader = csv.DictReader(f)
        for j, row in enumerate(reader):
            if j == config - 1:
                MLP1024.append(float(row["time_per_prediction"]))
                break

# ============================================================
# READ RK DATA
# ============================================================
with open(FILE1, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < start:
            continue
        if i >= entries:
            break
        avg_times_rk1.append(float(row["avg_per_step_runtime_s"]))
        dt_s_list_rk1.append(float(row["dt_s"]))

with open(FILE2, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < start:
            continue
        if i >= entries:
            break
        avg_times_rk2.append(float(row["avg_per_step_runtime_s"]))
        dt_s_list_rk2.append(float(row["dt_s"]))

with open(FILE3, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < start:
            continue
        if i >= entries:
            break
        avg_times_rk4.append(float(row["avg_per_step_runtime_s"]))
        dt_s_list_rk4.append(float(row["dt_s"]))

with open(FILE4, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < start:
            continue
        if i >= entries:
            break
        dur_s_list.append(float(row["dur_s"]))

avg_times_rk1 = np.array(avg_times_rk1)
avg_times_rk2 = np.array(avg_times_rk2)
avg_times_rk4 = np.array(avg_times_rk4)

dt_s_list_rk1 = np.array(dt_s_list_rk1)
dt_s_list_rk2 = np.array(dt_s_list_rk2)
dt_s_list_rk4 = np.array(dt_s_list_rk4)

dur_s_list = np.array(dur_s_list)

avg_time_slopes_rk1 = avg_times_rk1 / dt_s_list_rk1
avg_time_slopes_rk2 = avg_times_rk2 / dt_s_list_rk2
avg_time_slopes_rk4 = avg_times_rk4 / dt_s_list_rk4

# ============================================================
# LOAD STATE DATA FOR X WINDOW
# ============================================================
state_path = f"../02-run-rk/RK1/r{config:02d}/log-states.npy"

if not os.path.exists(state_path):
    print(f"missing {state_path}")

data = np.load(state_path)
values = np.array([x[0] for x in data])
labels = np.array([x[1] for x in data])

vhi_vals = values[labels == 'VHI']
vlo_vals = values[labels == 'VLO']

x_start = vhi_vals[0]
x_end = vlo_vals[0]

# ============================================================
# CREATE FIGURE
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

# ============================================================
# PLOT RK ENVELOPES ON AX1
# ============================================================
plot_rk_envelope(ax1, dt_s_list_rk4, dur_s_list, avg_time_slopes_rk4, "red", "RK4", "||")
plot_rk_envelope(ax1, dt_s_list_rk2, dur_s_list, avg_time_slopes_rk2, "blue", "RK2", "\\\\")
plot_rk_envelope(ax1, dt_s_list_rk1, dur_s_list, avg_time_slopes_rk1, "green", "RK1", "///")

# ============================================================
# AX1 FORMATTING
# ============================================================
ax1.set_xlim(left=x_start, right=x_end)
ax1.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x - x_start:.0f}")
)

plot_mlp_lines(ax1, MLP32, "#aa681d", "MLP32")
plot_mlp_lines(ax1, MLP64, "#1f77b4", "MLP64")
plot_mlp_lines(ax1, MLP128, "#b4b21f", "MLP128")
plot_mlp_lines(ax1, MLP256, "#1fb43d", "MLP256")
plot_mlp_lines(ax1, MLP512, "#6a379a", "MLP512")
plot_mlp_lines(ax1, MLP1024, "#ff0000", "MLP1024")

ax1.set_xlabel("Simulation Time (s)")
ax1.set_ylabel("Estimated Runtime (s)")
ax1.set_title(f"Runtime vs Simulation Time (RK and MLP) - Config ({config:03d})", fontsize=10)
ax1.grid()
ax1.legend()

# ============================================================
# AX2 FORMATTING
# ============================================================
min_MLP = min(min(MLP32), min(MLP64), min(MLP128), min(MLP256), min(MLP512), min(MLP1024))
max_MLP = max(max(MLP32), max(MLP64), max(MLP128), max(MLP256), max(MLP512), max(MLP1024))

multiplier = 1e5

ax2.set_xlim(left=0, right=max(MLP1024)/min(avg_time_slopes_rk1))
#ax2.xaxis.set_major_formatter(
 #   FuncFormatter(lambda x, _: f"{x - x_start:.0f}")
#)

plot_rk_envelope(ax2, dt_s_list_rk4, dur_s_list, avg_time_slopes_rk4, "red", "RK4", "||", multiplier)
plot_rk_envelope(ax2, dt_s_list_rk2, dur_s_list, avg_time_slopes_rk2, "blue", "RK2", "\\\\", multiplier)
plot_rk_envelope(ax2, dt_s_list_rk1, dur_s_list, avg_time_slopes_rk1, "green", "RK1", "///", multiplier)

plot_mlp_lines(ax2, MLP32, "#aa681d", "MLP32", multiplier)
plot_mlp_lines(ax2, MLP64, "#1f77b4", "MLP64", multiplier)
plot_mlp_lines(ax2, MLP128, "#b4b21f", "MLP128", multiplier)
plot_mlp_lines(ax2, MLP256, "#1fb43d", "MLP256", multiplier)
plot_mlp_lines(ax2, MLP512, "#6a379a", "MLP512", multiplier)
plot_mlp_lines(ax2, MLP1024, "#ff0000", "MLP1024", multiplier)

ax2.set_xlabel("Simulation Time (s)")
ax2.set_ylabel(r"Estimated Runtime ($\mu$s)")
ax2.set_ylim(min_MLP * multiplier, max_MLP * multiplier + 0.5)
ax2.set_title(f"Runtime vs Simulation Time (MLP Zoomed) - Config ({config:03d})", fontsize=10)
ax2.grid()

# ============================================================
# AX3 FORMATTING
# ============================================================
min_MLP = min(min(MLP32), min(MLP64), min(MLP128), min(MLP256), min(MLP512), min(MLP1024))
max_MLP = max(max(MLP32), max(MLP64), max(MLP128), max(MLP256), max(MLP512), max(MLP1024))

multiplier = 1e5

ax3.set_xlim(left=0, right=max(MLP1024)/max(avg_time_slopes_rk1))
#ax3.xaxis.set_major_formatter(
 #   FuncFormatter(lambda x, _: f"{x - x_start:.0f}")
#)

plot_rk_envelope(ax3, dt_s_list_rk4, dur_s_list, avg_time_slopes_rk4, "red", "RK4", "||", multiplier)
plot_rk_envelope(ax3, dt_s_list_rk2, dur_s_list, avg_time_slopes_rk2, "blue", "RK2", "\\\\", multiplier)
plot_rk_envelope(ax3, dt_s_list_rk1, dur_s_list, avg_time_slopes_rk1, "green", "RK1", "///", multiplier)

plot_mlp_lines(ax3, MLP32, "#aa681d", "MLP32", multiplier)
plot_mlp_lines(ax3, MLP64, "#1f77b4", "MLP64", multiplier)
plot_mlp_lines(ax3, MLP128, "#b4b21f", "MLP128", multiplier)
plot_mlp_lines(ax3, MLP256, "#1fb43d", "MLP256", multiplier)
plot_mlp_lines(ax3, MLP512, "#6a379a", "MLP512", multiplier)
plot_mlp_lines(ax3, MLP1024, "#ff0000", "MLP1024", multiplier)

ax3.set_xlabel("Simulation Time (s)")
ax3.set_ylabel(r"Estimated Runtime ($\mu$s)")
ax3.set_ylim(min_MLP * multiplier, max_MLP * multiplier + 0.5)
ax3.set_title(f"Runtime vs Simulation Time (MLP Zoomed) - Config ({config:03d})", fontsize=10)
ax3.grid()

# ============================================================
# SAVE FIGURE
# ============================================================
plt.tight_layout()
plt.savefig(
    f"./plots/Execution_Time_Comparison_MLP_Configuration_{config:03d}.png",
    dpi=300
)
plt.close()