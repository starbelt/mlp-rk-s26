#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_sweep_with_mlp.sh <row_idx> <mlp_cfg> <mlp_pth> <mlp_norm> [input_csv] [stride]
#
# Example:
#   ./plot_sweep_with_mlp.sh 0 \
#       mlp-1024-cfg.json \
#       mlp.pt \
#       mlp-norm.pt \
#       rk_configs.csv \
#       5

if [ $# -lt 3 ]; then
    echo "Usage:"
    echo "  $0 <row_idx> <mlp_cfg> <mlp_pth> <mlp_norm> [input_csv] [stride]"
    exit 1
fi

ROW_IDX="$1"
MLP_CFG="../MLP/mlp-cfg/mlp-0256/mlp-0256-5/mlp-0256-5.json"
MLP_PTH="../MLP/mlp-cfg/mlp-0256/mlp-0256-5/mlp-0256-5.pt"
MLP_NORM="../MLP/mlp-cfg/mlp-0256/mlp-0256-5/mlp-0256-5-norm.pt"

INPUT_CSV="${2:-rk_configs.csv}"
STRIDE="${3:-1}"

BASE_DIR="outputs/rk1_dt_s_sweep/row_${ROW_IDX}"
OUT_FILE="${BASE_DIR}/rk1_node_with_mlp.pdf"

echo "----------------------------------------"
echo "Plotting dt_s sweep with MLP overlay"
echo "Row index:  $ROW_IDX"
echo "Input CSV:  $INPUT_CSV"
echo "Stride:     $STRIDE"
echo "Output:     $OUT_FILE"
echo "----------------------------------------"
echo

python3 rk_plot.py \
  --base_dir "$BASE_DIR" \
  --which node \
  --stride "$STRIDE" \
  --out "$OUT_FILE" \
  --mlp_cfg "$MLP_CFG" \
  --mlp_pth "$MLP_PTH" \
  --mlp_norm "$MLP_NORM" \
  --input_csv "$INPUT_CSV" \
  --row_idx "$ROW_IDX"

echo
echo "Done."

