#!/bin/bash

# -------- USER SETTINGS --------
NUM=1024
MLP_ROOT="../mlp-cfg/mlp-$NUM"
DATA_DIR="../03-split-data"
OUT_ROOT="./mlp-$NUM-results"
SCRIPT="tst_mlp.py"

mkdir -p "$OUT_ROOT"

# -------- LOOP OVER CONFIG FOLDERS --------
for CFG_DIR in "$MLP_ROOT"/*/; do
    echo "Processing folder: $CFG_DIR"

    # find files
    JSON_FILE=$(find "$CFG_DIR" -name "*.json" | head -n 1)
    PT_FILE=$(find "$CFG_DIR" -name "*.pt" ! -name "*norm.pt" | head -n 1)
    NORM_FILE=$(find "$CFG_DIR" -name "*norm.pt" | head -n 1)

    # skip if anything missing
    if [[ -z "$JSON_FILE" || -z "$PT_FILE" || -z "$NORM_FILE" ]]; then
        echo "Skipping $CFG_DIR (missing files)"
        continue
    fi

    # extract config name
    CFG_NAME=$(basename "$CFG_DIR")

    # output directory per config
    OUT_DIR="$OUT_ROOT/$CFG_NAME"
    mkdir -p "$OUT_DIR"

    echo "Running:"
    echo "  JSON: $JSON_FILE"
    echo "  PT:   $PT_FILE"
    echo "  NORM: $NORM_FILE"

    python3 "$SCRIPT" \
        "$JSON_FILE" \
        "$PT_FILE" \
        "$NORM_FILE" \
        "$DATA_DIR" \
        "$OUT_DIR"

    echo "Done: $CFG_NAME"
    echo "----------------------------------"
done