#!/bin/bash

file="rk_configs.csv"

# number of rows excluding header
num_rows=$(($(wc -l < "$file") - 1))

cd RK1

for ((i=0; i<num_rows; i++))
do
    python3 sim_esys_cap_rk1.py --input "../$file" --row_idx "$i" --out_base ./
done