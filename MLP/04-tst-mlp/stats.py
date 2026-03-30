import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

df = pd.read_csv("./mlp-0032-results/mlp-0032-2/mlp-0032-2-tst-summary.csv")

configs = df["configuration"].astype(int).to_numpy()
times = df["time_per_prediction"].to_numpy()

mean = df["time_per_prediction"].mean()
std  = df["time_per_prediction"].std()

plt.figure(figsize=(8, 5))

plt.plot(configs, times, marker="o", label="Time per Prediction")

# Mean line
plt.axhline(mean, color="red", linestyle="--", label="Mean")

# Shaded std region
plt.fill_between(
    configs,
    mean - std,
    mean + std,
    alpha=0.2,
    color="red",
    label="±1 Std Dev"
)

plt.xlabel("Configuration")
plt.ylabel("Time per Prediction (s)")
plt.title("Prediction Time by Configuration")
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 5))

x = np.linspace(min(times), max(times), 100)
y = norm.pdf(x, mean, std)


# histogram
plt.hist(times, bins=30, density=True, alpha=0.6, label="Data")

# bell curve
plt.plot(x, y, color="red", label="Normal Fit")

plt.xlabel("Time per Prediction (s)")
plt.ylabel("Density")
plt.title("Bell Curve (Normal Distribution Fit)")
plt.legend()

plt.show()

plt.show()