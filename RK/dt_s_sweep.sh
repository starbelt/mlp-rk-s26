#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./dt_s_sweep.sh <row_idx> [input_csv]
#
# Example:
#   ./dt_s_sweep.sh 0 rk_configs.csv

if [ $# -lt 1 ]; then
    echo "Usage: $0 <row_idx> [input_csv]"
    exit 1
fi

ROW_IDX="$1"
INPUT_CSV="${2:-RK/rk_configs.csv}"

SCRIPT="RK1/sim_esys_cap_rk1.py"
BASE_OUT="outputs/rk1_dt_s_sweep"

# ---- read dt_s from the specified row of the CSV (robust parsing) ----
DT_BASE="$(
python3 - "$ROW_IDX" "$INPUT_CSV" <<'PY'
import sys, csv

row_idx = int(sys.argv[1])
path = sys.argv[2]

with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

if row_idx < 0 or row_idx >= len(rows):
    raise SystemExit(f"row_idx out of range: {row_idx} (rows={len(rows)})")

dt_s = rows[row_idx].get("dt_s", "")
if dt_s == "":
    raise SystemExit("Missing dt_s in selected row")

print(float(dt_s))
PY
)"

echo "Row index: $ROW_IDX"
echo "Input CSV: $INPUT_CSV"
echo "Base dt_s (from CSV): $DT_BASE"
echo

mkdir -p "$BASE_OUT"

ROW_DIR="${BASE_OUT}/row_${ROW_IDX}"
mkdir -p "$ROW_DIR"

N=12
END_VAL=1

for i in $(seq 0 $((N-1))); do

    DT_VAL=$(echo "$DT_BASE + ($i/($N-1)) * ($END_VAL - $DT_BASE)" | bc -l)

    DT_LABEL="$(printf "%.9f" "$DT_VAL")"

    VARIANT_DIR="${ROW_DIR}/dt_${DT_LABEL}"
    mkdir -p "$VARIANT_DIR"

    echo "Running i=$i  dt_s=$DT_VAL"
    echo " -> out_base: $VARIANT_DIR"

    python3 "$SCRIPT" \
        --input "$INPUT_CSV" \
        --row_idx "$ROW_IDX" \
        --dt_s "$DT_VAL" \
        --out_base "$VARIANT_DIR"
done

echo
echo "Sweep complete. Outputs in: $ROW_DIR"