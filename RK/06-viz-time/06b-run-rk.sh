#!/bin/bash

rk_num=4
file="./rk_configs_execution.csv"
out_file="results/RK${rk_num}_timing_results.csv"

rm -f "$out_file"
echo "$rk_num"

# number of rows excluding header
num_rows=$(( $(wc -l < "$file") - 1 ))

seq 0 $((num_rows - 1)) | python3 -c "
from tqdm import tqdm
import subprocess
import sys

rk_num = ${rk_num}
out_file = '${out_file}'
input_file = '${file}'

for i in tqdm(map(int, sys.stdin), total=${num_rows}):
    subprocess.run([
        'python3', f'rk{rk_num}-benchmark.py',
        '--input', input_file,
        '--row_idx', str(i),
        '--num_runs', '100',
        '--output', out_file
    ], check=True)
"