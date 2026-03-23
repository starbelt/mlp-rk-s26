#!/bin/bash

file="rk_configs_execution.csv"
rm -f results/timing_results.csv

# number of rows excluding header
num_rows=$(($(wc -l < "$file") - 1))

seq 0 $((num_rows-1)) | python3 -c "
from tqdm import tqdm
import subprocess, sys

for i in tqdm(list(map(int, sys.stdin))):
    subprocess.run([
        'python3', 'rk1-benchmark.py',
        '--input', 'rk_configs_execution.csv',
        '--row_idx', str(i),
        '--num_runs', '100',
        '--output', 'results/timing_results.csv'
    ])
"

