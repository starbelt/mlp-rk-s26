#!/bin/bash

for N in 32 64 128 256 512 1024
do
  NODES=$(printf "%04d" $N)  # zero-pad to 4 digits

  for LAYERS in 2 3 4 5 6 7
  do
    ROOT="../../MLP/mlp-cfg/mlp-$NODES/mlp-$NODES-$LAYERS/mlp-$NODES-$LAYERS"

    echo "Running NODES=$NODES, LAYERS=$LAYERS"

    python3 mlp_accuracy.py \
        "$ROOT.json" \
        "$ROOT.pt" \
        "$ROOT-norm.pt" \
        ../../MLP/03-split-data/tst/ \
        ./
  done
done