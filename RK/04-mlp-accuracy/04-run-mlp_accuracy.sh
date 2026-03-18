#!/bin/bash

NODES=0256
LAYERS=6
ROOT="../../MLP/mlp-cfg/mlp-$NODES/mlp-$NODES-$LAYERS/mlp-$NODES-$LAYERS"

python3 mlp_accuracy.py \
    $ROOT.json\
    $ROOT.pt\
    $ROOT-norm.pt\
    ../../MLP/02-data/\
    ./