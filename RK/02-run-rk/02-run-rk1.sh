# Configuration
# ============================================================

# Path to the CSV file containing RK simulation configurations
file="../01-gen-rk-data/rk_configs.csv"

# ============================================================
# Determine Number of Configurations
# ============================================================

# Count total lines in the file, subtract 1 to exclude header row
num_rows=$(($(wc -l < "$file") - 1))

# ============================================================
# Navigate to Output Directory
# ============================================================

# Change into the RK1 directory where outputs will be stored
cd RK4

# ============================================================
# Run Simulations for Each Configuration
# ============================================================

# Loop over each row index in the CSV file
for ((i=0; i<num_rows; i++))
do
    # Run the RK1 simulation script for the given configuration row
    # --input   : path to the configuration CSV
    # --row_idx : which row (configuration) to run
    # --out_base: output directory (current directory ./)
    python3 sim_esys_cap_rk4.py \
        --input "../$file" \
        --row_idx "$i" \
        --out_base ./output
done