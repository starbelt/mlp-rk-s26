import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Set your 13 CSV file paths here
# =========================
TS_1  = r"RK1/r01/step-times.csv"
TS_2  = r"RK1/r02/step-times.csv"
TS_3  = r"RK1/r03/step-times.csv"
TS_4  = r"RK1/r04/step-times.csv"
TS_5  = r"RK1/r05/step-times.csv"
TS_6  = r"RK1/r06/step-times.csv"
TS_7  = r"RK1/r07/step-times.csv"
TS_8  = r"RK1/r08/step-times.csv"
TS_9  = r"RK1/r09/step-times.csv"
TS_10 = r"RK1/r10/step-times.csv"
TS_11 = r"RK1/r11/step-times.csv"
TS_12 = r"RK1/r12/step-times.csv"
TS_13 = r"../MLP/04-tst-mlp/mlp-1024-results/mlp-1024-2-000-prediction-times.csv"

csv_paths = [
    #TS_1,
    TS_2,
    TS_3,
    TS_4,
    TS_5,
    TS_6,
    TS_7,
    TS_8,
    TS_9,
    TS_10,
    TS_11,
    TS_12,
    #TS_13,
]

# =========================
# Plot all CSVs
# =========================
plt.figure(figsize=(10, 6))

for i, path in enumerate(csv_paths, start=1):
    df = pd.read_csv(path)

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    plt.plot(x, y, label=f"CSV {i}")

plt.axhline(
    y=8.781763975593e-06,
    linestyle="--",
    label="Average Time for MLP Prediction"
)

plt.xlabel("Time")
plt.ylabel("Execution Time (s)")
plt.title("RK Time Steps VS MLP Prediction Times")
plt.legend()
plt.grid(True)
plt.tight_layout()

# =========================
# Save figure as PNG
# =========================
plt.savefig("rk_vs_mlp_runtime.png", dpi=300)

plt.show()