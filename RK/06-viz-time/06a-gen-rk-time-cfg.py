import pandas as pd

INPUT_CSV  = "../01-gen-rk-data/rk_configs.csv"
OUTPUT_CSV = "./rk_configs_execution_test.csv"

df = pd.read_csv(INPUT_CSV)

print("Columns:", df.columns.tolist())

df["dur_s"] = df["dt_s"]

df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Saved to {OUTPUT_CSV}")